"""Pseudo-streaming: clause by clause.

The model cannot stream inside a clause (every position is half-revealed
until the last step and the reveal order is not left-to-right), but a clause
is done in 0.15–0.35 s and plays for 1–3 s, so a text can be cut at
punctuation and synthesised one clause ahead of playback: the first clause is
the only wait, everything after it is ready before the player needs it.

``generate_stream`` yields ``StreamPiece`` objects in text order; ``ready_ms``
is the wall time since the call at which the piece's audio existed, so a
caller (or ``bench/demo_stream.py``) can check the playback cursor never
overtakes generation.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterator

import mlx.core as mx
import numpy as np

from .pipeline import OmniVoiceTTS, VoicePrompt
from .sampler import SamplerConfig

_CLAUSE_END = "。！？；!?;"
_PAUSE = "，,："


def split_clauses(text: str, *, min_chars: int = 6, max_chars: int = 40) -> list[str]:
    """Cut at sentence punctuation, also at commas once a piece is long enough;
    glue pieces shorter than ``min_chars`` onto their neighbour."""
    pieces: list[str] = []
    cur = ""
    for ch in text.strip():
        cur += ch
        if ch in _CLAUSE_END or (ch in _PAUSE and len(cur) >= min_chars) or len(cur) >= max_chars:
            pieces.append(cur)
            cur = ""
    if cur.strip():
        pieces.append(cur)
    merged: list[str] = []
    for p in pieces:
        if merged and len(re.sub(r"[^\w]", "", p)) < min_chars:
            merged[-1] += p
        else:
            merged.append(p)
    return [p.strip() for p in merged if p.strip()]


@dataclass
class StreamPiece:
    index: int
    text: str
    audio: np.ndarray       # 24 kHz float32, post-processed
    seconds: float          # len(audio) / 24000
    synth_ms: float         # unmask + decode for this piece
    ready_ms: float         # wall time since generate_stream() was called


def generate_stream(tts: OmniVoiceTTS, text: str, prompt: VoicePrompt | None, *, language: str | None = "zh",
                    sampler: SamplerConfig | None = None, pad_duration: float = 0.05,
                    fade_duration: float = 0.05, seed: int | None = None,
                    continuity: bool = False, context_clauses: int = 1) -> Iterator[StreamPiece]:
    """Yield the clauses of ``text`` as they are synthesised, in order.

    ``continuity``: synthesise clause i against the reference *plus the previous
    ``context_clauses`` clauses' own tokens and text*, so the model hears where the
    sentence came from and carries pitch / speed across the cut instead of
    restarting from the reference every clause (the official long-text path
    does the same with chunk 0 as reference). Measured (README §17): it makes
    things WORSE — UTMOS 2.95 → 2.65 over 22 clauses, one clause down to 1.29;
    the model's own output is a poor reference. Off by default, kept for study.
    """
    sampler = sampler or SamplerConfig()
    t_start = time.perf_counter()
    history: list[tuple[str, mx.array]] = []   # (clause text, its generated tokens)
    for i, clause in enumerate(split_clauses(text)):
        p = prompt
        if continuity and prompt is not None and history:
            ctx = history[-context_clauses:]
            p = VoicePrompt(mx.concatenate([prompt.ref_tokens] + [t for _, t in ctx], axis=0),
                            " ".join([prompt.ref_text] + [c for c, _ in ctx]), prompt.ref_rms)
        r = tts.generate(clause, p, language=language, sampler=sampler, postprocess=False, seed=seed)
        wav = tts.post_process(r.audio, prompt.ref_rms if prompt else None,
                               pad_duration=pad_duration, fade_duration=fade_duration)
        history.append((clause, r.tokens))
        yield StreamPiece(i, clause, wav, len(wav) / tts.sample_rate, r.timing.synth_ms,
                          (time.perf_counter() - t_start) * 1000)
