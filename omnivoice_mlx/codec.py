"""Higgs-audio v2 tokenizer (25 Hz, 8 RVQ codebooks, 24 kHz), mlx-audio's port vendored
into ``omnivoice_mlx.higgs`` so the runtime needs neither mlx-audio nor transformers.

Two halves with very different weight:

* decode (tokens -> waveform): quantizer + fc2 + acoustic decoder, 90 MB fp32,
  45 MB fp16 — what every synthesis needs. Loaded at construction, in
  ``decode_dtype``, from ``<model_dir>/codec-decoder-<dtype>.safetensors``
  when ``scripts/convert.py --codec`` wrote one, else filtered out of the full
  ``audio_tokenizer/model.safetensors``.
* encode (reference audio -> tokens): acoustic encoder + HuBERT semantic
  branch + fusion, the other 700 MB. Only ``make_prompt`` needs it, once per
  voice, so it is loaded on first use (fp32, from the full file) and can be
  dropped again with ``release_encoder()``; ``VoicePrompt.save`` lets a
  deployment skip it entirely.

Numerics: mlx-audio's Snake runs in float32 whatever the weight dtype, so the
fp16 decoder is safe (README §17 measures the difference against fp32).
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

DECODE_PREFIXES = ("acoustic_decoder.", "quantizer.", "fc2.")
_DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def slim_decoder_path(model_dir: Path, dtype: str) -> Path:
    return Path(model_dir) / f"codec-decoder-{dtype}.safetensors"


def decoder_weights(model_dir: Path, dtype: str) -> dict[str, mx.array]:
    """The decode-path tensors, sanitized to MLX layout, cast to ``dtype``."""
    from .higgs import HiggsAudioTokenizer

    slim = slim_decoder_path(model_dir, dtype)
    if slim.exists():
        return dict(mx.load(str(slim)))
    full = Path(model_dir) / "audio_tokenizer" / "model.safetensors"
    if not full.exists():
        raise FileNotFoundError(f"{full} not found (k2-fsa/OmniVoice ships it)")
    raw = {k: v for k, v in mx.load(str(full)).items() if k.startswith(DECODE_PREFIXES)}
    sanitized = HiggsAudioTokenizer.sanitize(None, raw)  # type: ignore[arg-type]
    return {k: v.astype(_DTYPES[dtype]) for k, v in sanitized.items()}


def write_slim_decoder(model_dir: Path, dtype: str = "float16") -> Path:
    """Write ``codec-decoder-<dtype>.safetensors`` next to the model weights."""
    w = decoder_weights(model_dir, dtype)
    out = slim_decoder_path(model_dir, dtype)
    mx.save_safetensors(str(out), w)
    return out


class Codec:
    def __init__(self, model_dir: str | Path, *, decode_dtype: str = "float16"):
        import json

        from .higgs import HiggsAudioConfig, HiggsAudioTokenizer

        self.model_dir = Path(model_dir)
        cfg_path = self.model_dir / "audio_tokenizer" / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"{cfg_path} not found")
        self.config = HiggsAudioConfig.from_dict(json.loads(cfg_path.read_text()))
        self.sample_rate = int(self.config.sample_rate)
        self.hop_length = 960
        self.decode_dtype = decode_dtype

        dec = HiggsAudioTokenizer(self.config)      # decode modules only (no _init_encode_modules)
        del dec["acoustic_encoder"]                  # constructed by __init__, never used for decode
        dec.load_weights(list(decoder_weights(self.model_dir, decode_dtype).items()))
        mx.eval(dec.parameters())
        self._dec = dec
        self._enc = None

    # ---- decode ----------------------------------------------------------
    def decode(self, tokens: mx.array) -> mx.array:
        """[T, 8] int32 -> [T * 960] float32 (lazy)."""
        z = self._dec.quantizer.decode(tokens[None])            # [1, T, 1024]
        z = self._dec.fc2(z.astype(_DTYPES[self.decode_dtype]))
        return self._dec.acoustic_decoder(z)[0, :, 0].astype(mx.float32)

    # ---- encode ----------------------------------------------------------
    def _encoder(self):
        if self._enc is None:
            from .higgs import HiggsAudioTokenizer

            self._enc = HiggsAudioTokenizer.from_pretrained(str(self.model_dir))
        return self._enc

    def encode(self, wav24k: np.ndarray) -> mx.array:
        """mono float32 at 24 kHz, length a multiple of 960 -> [T, 8] int32 (loads the encoder on first use)."""
        x = mx.array(np.asarray(wav24k, dtype=np.float32))[None, :, None]
        tokens = self._encoder().encode(x)[0]
        mx.eval(tokens)
        return tokens

    def release_encoder(self) -> None:
        """Drop the 700 MB encode branch (reload happens on the next encode)."""
        self._enc = None
        mx.clear_cache()

    @property
    def encoder_loaded(self) -> bool:
        return self._enc is not None
