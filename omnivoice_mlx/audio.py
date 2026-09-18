"""Waveform utilities with the official (pydub / torchaudio) semantics, in numpy.

* ``sinc_resample``: torchaudio ``resample`` (sinc_interp_hann), as in mlx-audio.
* silence detection: pydub's ``detect_silence`` / ``detect_nonsilent`` /
  ``split_on_silence`` / ``detect_leading_silence`` on int16-quantised RMS
  (dBFS relative to 32768), 10 ms steps where the official passes them.
* ``remove_silence`` / ``trim_long_audio`` / ``fade_and_pad``: the official
  ``omnivoice/utils/audio.py`` functions, channels dropped (mono float32 [T]).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf


def load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1).astype(np.float32), int(sr)


def write_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    sf.write(str(path), np.clip(audio, -1.0, 1.0).astype(np.float32), sr, subtype="PCM_16")


def sinc_resample(waveform: np.ndarray, orig_freq: int, new_freq: int,
                  lowpass_filter_width: int = 6, rolloff: float = 0.99) -> np.ndarray:
    if orig_freq == new_freq:
        return waveform.astype(np.float32)
    g = math.gcd(int(orig_freq), int(new_freq))
    orig_r, new_r = orig_freq // g, new_freq // g
    base_freq = min(orig_r, new_r) * rolloff
    width = math.ceil(lowpass_filter_width * orig_r / base_freq)
    idx = np.arange(-width, width + orig_r, dtype=np.float64)[None, :] / orig_r
    t = np.arange(0, -new_r, -1, dtype=np.float64)[:, None] / new_r + idx
    t *= base_freq
    t = np.clip(t, -lowpass_filter_width, lowpass_filter_width)
    window = np.cos(t * np.pi / lowpass_filter_width / 2) ** 2
    t_pi = t * np.pi
    kernel = np.where(t_pi == 0, 1.0, np.sin(t_pi) / t_pi)
    kernel = (kernel * window * (base_freq / orig_r)).astype(np.float32)
    length = len(waveform)
    padded = np.pad(waveform.astype(np.float32), (width, width + orig_r))
    out_len = math.ceil(length * new_r / orig_r)
    result = np.zeros(out_len, dtype=np.float32)
    for phase in range(new_r):
        conv = np.convolve(padded, kernel[phase, ::-1], mode="valid")[::orig_r]
        n = min(len(conv), math.ceil((out_len - phase) / new_r))
        result[phase:phase + n * new_r:new_r] = conv[:n]
    return result


# --- pydub-compatible silence detection ---------------------------------------

def _ms(n_samples: int, sr: int) -> int:
    return round(1000 * (n_samples / sr))


def _sample(ms: int, sr: int) -> int:
    return int(ms * (sr / 1000.0))


def _pcm16(audio: np.ndarray) -> np.ndarray:
    return (audio * 32767.0).clip(-32768, 32767).astype(np.int16)


def _rms(pcm: np.ndarray, start_ms: int, end_ms: int, sr: int) -> float:
    s, e = _sample(start_ms, sr), min(len(pcm), _sample(end_ms, sr))
    if e <= s:
        return 0.0
    w = pcm[s:e].astype(np.float64)
    return float(np.sqrt(np.mean(w * w)))


def detect_silence(audio: np.ndarray, sr: int, min_silence_len: int = 1000,
                   silence_thresh: float = -16.0, seek_step: int = 1) -> list[tuple[int, int]]:
    seg_len = _ms(len(audio), sr)
    if seg_len < min_silence_len:
        return []
    pcm = _pcm16(audio)
    thresh = (10 ** (silence_thresh / 20.0)) * 32768.0
    last = seg_len - min_silence_len
    starts = list(range(0, last + 1, seek_step))
    if last % seek_step:
        starts.append(last)
    silence_starts = [s for s in starts if _rms(pcm, s, s + min_silence_len, sr) <= thresh]
    if not silence_starts:
        return []
    ranges = []
    prev = silence_starts.pop(0)
    cur_start = prev
    for s in silence_starts:
        continuous = s == prev + seek_step
        has_gap = s > prev + min_silence_len
        if not continuous and has_gap:
            ranges.append((cur_start, prev + min_silence_len))
            cur_start = s
        prev = s
    ranges.append((cur_start, prev + min_silence_len))
    return ranges


def detect_nonsilent(audio: np.ndarray, sr: int, min_silence_len: int = 1000,
                     silence_thresh: float = -16.0, seek_step: int = 1) -> list[tuple[int, int]]:
    seg_len = _ms(len(audio), sr)
    if seg_len == 0:
        return []
    silent = detect_silence(audio, sr, min_silence_len, silence_thresh, seek_step)
    if not silent:
        return [(0, seg_len)]
    if silent[0][0] == 0 and silent[0][1] == seg_len:
        return []
    prev_end = 0
    out = []
    for s, e in silent:
        out.append((prev_end, s))
        prev_end = e
    if silent[-1][1] != seg_len:
        out.append((prev_end, seg_len))
    if out and out[0] == (0, 0):
        out.pop(0)
    return out


def detect_leading_silence(audio: np.ndarray, sr: int, silence_threshold: float = -50.0,
                           chunk_size: int = 10) -> int:
    """Milliseconds of leading silence (pydub semantics: 10 ms chunks, dBFS)."""
    pcm = _pcm16(audio)
    total = _ms(len(audio), sr)
    trim = 0
    while trim < total:
        r = _rms(pcm, trim, trim + chunk_size, sr)
        dbfs = 20 * math.log10(r / 32768.0) if r > 0 else -float("inf")
        if dbfs >= silence_threshold:
            break
        trim += chunk_size
    return min(trim, total)


def _slice(audio: np.ndarray, sr: int, start_ms: int, end_ms: int) -> np.ndarray:
    return audio[max(0, _sample(start_ms, sr)):min(len(audio), _sample(end_ms, sr))]


def split_on_silence(audio: np.ndarray, sr: int, min_silence_len: int, silence_thresh: float,
                     keep_silence: int, seek_step: int) -> list[np.ndarray]:
    ranges = [(s - keep_silence, e + keep_silence)
              for s, e in detect_nonsilent(audio, sr, min_silence_len, silence_thresh, seek_step)]
    for i in range(len(ranges) - 1):
        last_end, next_start = ranges[i][1], ranges[i + 1][0]
        if next_start < last_end:
            mid = (last_end + next_start) // 2
            ranges[i] = (ranges[i][0], mid)
            ranges[i + 1] = (mid, ranges[i + 1][1])
    seg_len = _ms(len(audio), sr)
    return [_slice(audio, sr, max(s, 0), min(e, seg_len)) for s, e in ranges]


def remove_silence(audio: np.ndarray, sr: int, mid_sil: int = 300, lead_sil: int = 100,
                   trail_sil: int = 300) -> np.ndarray:
    """Official ``remove_silence``: drop middle silences > mid_sil ms, keep lead/trail ms at the edges."""
    x = np.asarray(audio, dtype=np.float32)
    if mid_sil > 0:
        pieces = split_on_silence(x, sr, min_silence_len=mid_sil, silence_thresh=-50,
                                  keep_silence=mid_sil, seek_step=10)
        x = np.concatenate(pieces) if pieces else x[:0]
    # remove_silence_edges: detect_leading_silence on both ends, threshold -50 dBFS
    start = max(0, detect_leading_silence(x, sr, -50.0) - lead_sil)
    x = x[_sample(start, sr):]
    rev = x[::-1]
    start = max(0, detect_leading_silence(rev, sr, -50.0) - trail_sil)
    x = rev[_sample(start, sr):][::-1]
    return np.ascontiguousarray(x, dtype=np.float32)


def trim_long_audio(audio: np.ndarray, sr: int, max_duration: float = 15.0,
                    min_duration: float = 3.0, trim_threshold: float = 20.0) -> np.ndarray:
    if len(audio) / sr <= trim_threshold:
        return audio
    nonsilent = detect_nonsilent(audio, sr, min_silence_len=100, silence_thresh=-40, seek_step=10)
    if not nonsilent:
        return audio
    max_ms, min_ms = int(max_duration * 1000), int(min_duration * 1000)
    best = 0
    for s, e in nonsilent:
        if best < s <= max_ms:
            best = s
        if e > max_ms:
            break
    if best < min_ms:
        best = min(max_ms, _ms(len(audio), sr))
    return _slice(audio, sr, 0, best)


def fade_and_pad(audio: np.ndarray, sr: int, pad_duration: float = 0.1,
                 fade_duration: float = 0.1) -> np.ndarray:
    if len(audio) == 0:
        return audio
    out = audio.astype(np.float32).copy()
    k = min(int(fade_duration * sr), len(out) // 2)
    if k > 0:
        out[:k] *= np.linspace(0, 1, k, dtype=np.float32)
        out[-k:] *= np.linspace(1, 0, k, dtype=np.float32)
    pad = int(pad_duration * sr)
    if pad > 0:
        z = np.zeros(pad, dtype=np.float32)
        out = np.concatenate([z, out, z])
    return out


def cross_fade_chunks(chunks: list[np.ndarray], sr: int, silence_duration: float = 0.3) -> np.ndarray:
    """Official ``cross_fade_chunks`` (mono): fade out / silence / fade in at every boundary."""
    if len(chunks) == 1:
        return chunks[0]
    total_n = int(silence_duration * sr)
    fade_n = total_n // 3
    merged = chunks[0].astype(np.float32).copy()
    for chunk in chunks[1:]:
        fo = min(fade_n, len(merged))
        if fo > 0:
            merged[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32)
        nxt = chunk.astype(np.float32).copy()
        fi = min(fade_n, len(nxt))
        if fi > 0:
            nxt[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)
        merged = np.concatenate([merged, np.zeros(fade_n, dtype=np.float32), nxt])
    return merged
