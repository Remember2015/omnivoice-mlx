"""End-to-end OmniVoice voice cloning on MLX: reference prompt, duration
estimate, prompt ids, unmasking, codec decode, official post-processing.

Follows ``OmniVoice.generate`` / ``create_voice_clone_prompt`` (official) for
the single-utterance voice-clone path: reference preprocessing with the
official parameters (RMS boost to 0.1, silence removal 200/100/200 ms, clip to a
hop multiple, punctuation appended to ``ref_text``), duration from the rule
estimator against the reference's own speaking rate, post-processing 500/100/100
ms silence removal, RMS matching and 0.1 s fade + pad.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import audio as A
from .codec import Codec
from .duration import RuleDurationEstimator
from .model import OmniVoice, load_model
from .sampler import Prompt, SamplerConfig, UnmaskStats, unmask, unmask_batch
from .text import TextTokenizer, add_punctuation, chunk_text_punctuation, prompt_text_ids


@dataclass
class VoicePrompt:
    ref_tokens: mx.array  # [Tr, C] int32
    ref_text: str
    ref_rms: float

    def save(self, path: str | Path) -> None:
        np.savez(str(path), tokens=np.asarray(self.ref_tokens), ref_text=self.ref_text,
                 ref_rms=self.ref_rms)

    @classmethod
    def load(cls, path: str | Path) -> "VoicePrompt":
        d = np.load(str(path))
        return cls(mx.array(d["tokens"].astype(np.int32)), str(d["ref_text"]), float(d["ref_rms"]))


@dataclass
class Timing:
    prompt_ms: float = 0.0
    unmask_ms: float = 0.0
    decode_ms: float = 0.0
    post_ms: float = 0.0

    @property
    def synth_ms(self) -> float:
        return self.unmask_ms + self.decode_ms


@dataclass
class GenResult:
    audio: np.ndarray      # post-processed (or raw) waveform, 24 kHz
    raw_seconds: float     # T * hop / sr, the generated length before post-processing
    tokens: mx.array       # [T, C]
    T: int
    prompt_len: int
    timing: Timing
    stats: UnmaskStats = field(default_factory=UnmaskStats)

    @property
    def rtf(self) -> float:
        return (self.timing.synth_ms / 1000.0) / self.raw_seconds


class OmniVoiceTTS:
    def __init__(self, model_dir: str | Path, *, codec_dir: str | Path | None = None,
                 dtype: str = "bfloat16", bits: int = 0, group_size: int = 64,
                 quantize_embed: bool = False, quantize_heads: bool = False,
                 head_dtype: str = "float32", codec: Codec | None = None, fuse: bool = True,
                 cache_limit_mb: int = 512, custom_gemm: bool = False):
        self.model_dir = Path(model_dir)
        t0 = time.perf_counter()
        self.model: OmniVoice = load_model(self.model_dir, dtype=dtype, bits=bits, group_size=group_size,
                                           quantize_embed=quantize_embed, quantize_heads=quantize_heads,
                                           head_dtype=head_dtype, fuse=fuse, custom_gemm=custom_gemm)
        self.load_model_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.codec = codec if codec is not None else Codec(codec_dir or self.model_dir)
        self.load_codec_s = time.perf_counter() - t0
        self.tok = TextTokenizer(self.model_dir)
        self.estimator = RuleDurationEstimator()
        self.sample_rate = self.model.config.sample_rate
        self.hop = self.model.config.hop_length
        # MLX keeps freed buffers in a cache with no limit by default. After one large
        # batch the cache holds gigabytes of odd-sized buffers and every later run —
        # even a single sentence — is ~50 % slower (README §16); a bounded cache keeps
        # the allocator fast. 512 MB covers a step's transients at any batch size tried.
        mx.set_cache_limit(cache_limit_mb * 1024 * 1024)
        mx.clear_cache()

    # ---- reference --------------------------------------------------------
    def make_prompt(self, wav_path: str | Path, ref_text: str, *, preprocess: bool = True,
                    release_encoder: bool = False) -> VoicePrompt:
        wav, sr = A.load_mono(wav_path)
        if sr != self.sample_rate:
            wav = A.sinc_resample(wav, sr, self.sample_rate)
        rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
        if 0 < rms < 0.1:
            wav = wav * (0.1 / rms)
        if preprocess:
            wav = A.remove_silence(wav, self.sample_rate, mid_sil=200, lead_sil=100, trail_sil=200)
            if len(wav) == 0:
                raise ValueError("reference audio is empty after silence removal")
        clip = len(wav) % self.hop
        if clip:
            wav = wav[:-clip]
        tokens = self.codec.encode(wav)
        if release_encoder:
            self.codec.release_encoder()  # the 700 MB encode branch, back to ~0.9 GB resident
        if preprocess:
            ref_text = add_punctuation(ref_text)
        return VoicePrompt(tokens, ref_text, rms)

    # ---- prompt -----------------------------------------------------------
    def estimate_tokens(self, text: str, prompt: VoicePrompt | None, speed: float = 1.0) -> int:
        if prompt is None or not prompt.ref_text:
            est = self.estimator.estimate_duration(text, "Nice to meet you.", 25)
        else:
            est = self.estimator.estimate_duration(text, prompt.ref_text, int(prompt.ref_tokens.shape[0]))
        if speed > 0 and speed != 1.0:
            est = est / speed
        return max(1, int(est))

    def build_prompt(self, text: str, prompt: VoicePrompt | None, *, language: str | None = None,
                     instruct: str | None = None, denoise: bool = True) -> Prompt:
        C = self.model.config.num_audio_codebook
        style_ids, text_ids = prompt_text_ids(
            self.tok, text, ref_text=prompt.ref_text if prompt else None, language=language,
            instruct=instruct, denoise=denoise and prompt is not None)
        n_text = len(style_ids) + len(text_ids)
        text_block = mx.array(style_ids + text_ids, dtype=mx.int32)[:, None]
        text_block = mx.broadcast_to(text_block, (n_text, C))
        parts = [text_block]
        if prompt is not None:
            parts.append(prompt.ref_tokens.astype(mx.int32))
        ids = mx.concatenate(parts, axis=0)
        audio_mask = mx.concatenate([mx.zeros((n_text,), dtype=mx.bool_),
                                     mx.ones((ids.shape[0] - n_text,), dtype=mx.bool_)])
        return Prompt(ids, audio_mask)

    # ---- generation -------------------------------------------------------
    def normalize(self, text: str) -> str:
        """Spoken form of ``text`` (numbers, dates, units...) via omnivoice_mlx.textnorm;
        README §17: raw text is read wrong 8/34 times, normalised 2/34."""
        from .textnorm import normalize
        return normalize(text)

    def generate(self, text: str, prompt: VoicePrompt | None, *, language: str | None = "zh",
                 sampler: SamplerConfig | None = None, duration: float | None = None,
                 speed: float = 1.0, postprocess: bool = True, seed: int | None = None,
                 normalize: bool = False) -> GenResult:
        sampler = sampler or SamplerConfig()
        if normalize:
            text = self.normalize(text)
        timing = Timing()
        t0 = time.perf_counter()
        T = int(max(1, duration * self.model.config.frame_rate)) if duration else \
            self.estimate_tokens(text, prompt, speed)
        p = self.build_prompt(text, prompt, language=language)
        mx.eval(p.ids, p.audio_mask)
        timing.prompt_ms = (time.perf_counter() - t0) * 1000

        stats = UnmaskStats()
        t0 = time.perf_counter()
        tokens = unmask(self.model, p, T, sampler, seed=seed, stats=stats)
        timing.unmask_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        wav = np.asarray(self.codec.decode(tokens))
        timing.decode_ms = (time.perf_counter() - t0) * 1000
        raw_seconds = T * self.hop / self.sample_rate

        t0 = time.perf_counter()
        if postprocess:
            wav = self.post_process(wav, prompt.ref_rms if prompt else None)
        timing.post_ms = (time.perf_counter() - t0) * 1000
        return GenResult(wav, raw_seconds, tokens, T, p.length, timing, stats)

    def generate_batch(self, texts: list[str], prompt: VoicePrompt | None, *, language: str | None = "zh",
                       sampler: SamplerConfig | None = None, speed: float = 1.0, postprocess: bool = True,
                       seed: int | None = None, max_batch: int = 4, sort_by_length: bool = True,
                       normalize: bool = False) -> list[GenResult]:
        """Several utterances through packed unmasking loops (throughput mode).

        Sorted by estimated length and cut into buckets of ``max_batch`` (default 4,
        the user's pick: README §16 — the RTF plateau starts at B=4, B=8 costs
        +0.6 GB peak for −4 %); every bucket is one loop in which the transformer
        sees all its utterances each step. Results come back in
        input order. ``timing.unmask_ms`` of each result is its bucket's total
        split by the item's share of target tokens; ``decode_ms`` is its own.
        """
        sampler = sampler or SamplerConfig()
        if normalize:
            texts = [self.normalize(t) for t in texts]
        t0 = time.perf_counter()
        prompts = [self.build_prompt(t, prompt, language=language) for t in texts]
        Ts = [self.estimate_tokens(t, prompt, speed) for t in texts]
        mx.eval(*[p.ids for p in prompts])
        prompt_ms = (time.perf_counter() - t0) * 1000
        order = sorted(range(len(texts)), key=lambda i: -Ts[i]) if sort_by_length else list(range(len(texts)))
        results: list[GenResult | None] = [None] * len(texts)
        for b in range(0, len(order), max(1, max_batch)):
            idx = order[b:b + max(1, max_batch)]
            stats = UnmaskStats()
            t0 = time.perf_counter()
            toks = unmask_batch(self.model, [(prompts[i], Ts[i]) for i in idx], sampler, seed=seed, stats=stats)
            unmask_ms = (time.perf_counter() - t0) * 1000
            total_T = sum(Ts[i] for i in idx)
            for i, tk in zip(idx, toks):
                t0 = time.perf_counter()
                wav = np.asarray(self.codec.decode(tk))
                decode_ms = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                if postprocess:
                    wav = self.post_process(wav, prompt.ref_rms if prompt else None)
                timing = Timing(prompt_ms / len(texts), unmask_ms * Ts[i] / total_T, decode_ms,
                                (time.perf_counter() - t0) * 1000)
                results[i] = GenResult(wav, Ts[i] * self.hop / self.sample_rate, tk, Ts[i], prompts[i].length,
                                       timing, stats)
        return results  # type: ignore[return-value]

    def generate_long(self, text: str, prompt: VoicePrompt | None, *, language: str | None = "zh",
                      sampler: SamplerConfig | None = None, speed: float = 1.0,
                      chunk_duration: float = 15.0, chunk_threshold: float = 30.0,
                      batch_chunks: bool = True, seed: int | None = None) -> GenResult:
        """Official long-text path: text whose estimate exceeds ``chunk_threshold`` s
        is cut at punctuation into ~``chunk_duration`` s pieces, each synthesised
        against the same reference, cross-faded with 0.1 s gaps, post-processed
        once. The official generates a single item's chunks one after another;
        with ``batch_chunks`` they go through one packed loop (generate_batch).
        """
        sampler = sampler or SamplerConfig()
        T_est = self.estimate_tokens(text, prompt, speed)
        if T_est <= chunk_threshold * self.model.config.frame_rate:
            return self.generate(text, prompt, language=language, sampler=sampler, speed=speed, seed=seed)
        chunk_len = int(chunk_duration * self.model.config.frame_rate / (T_est / len(text)))
        chunks = chunk_text_punctuation(text, chunk_len, min_chunk_len=3)
        if batch_chunks:
            parts = self.generate_batch(chunks, prompt, language=language, sampler=sampler, speed=speed,
                                        postprocess=False, seed=seed)
        else:
            parts = [self.generate(c, prompt, language=language, sampler=sampler, speed=speed,
                                   postprocess=False, seed=seed) for c in chunks]
        t0 = time.perf_counter()
        wav = A.cross_fade_chunks([r.audio for r in parts], self.sample_rate)
        wav = self.post_process(wav, prompt.ref_rms if prompt else None)
        timing = Timing(sum(r.timing.prompt_ms for r in parts), sum(r.timing.unmask_ms for r in parts),
                        sum(r.timing.decode_ms for r in parts), (time.perf_counter() - t0) * 1000)
        stats = UnmaskStats(max(r.stats.steps for r in parts), sum(r.stats.forwards for r in parts),
                            sum(r.stats.tokens_through_backbone for r in parts))
        tokens = mx.concatenate([r.tokens for r in parts], axis=0)
        return GenResult(wav, sum(r.raw_seconds for r in parts), tokens, int(tokens.shape[0]),
                         parts[0].prompt_len, timing, stats)

    def post_process(self, wav: np.ndarray, ref_rms: float | None, *, remove_sil: bool = True,
                     pad_duration: float = 0.1, fade_duration: float = 0.1) -> np.ndarray:
        if remove_sil:
            wav = A.remove_silence(wav, self.sample_rate, mid_sil=500, lead_sil=100, trail_sil=100)
        if ref_rms is not None and ref_rms < 0.1:
            wav = wav * ref_rms / 0.1
        elif ref_rms is None:
            peak = float(np.abs(wav).max()) if len(wav) else 0.0
            if peak > 1e-6:
                wav = wav / peak * 0.5
        return A.fade_and_pad(wav, self.sample_rate, pad_duration, fade_duration)
