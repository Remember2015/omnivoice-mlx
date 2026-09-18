"""Pseudo-streaming demo: a paragraph clause by clause, with the timeline a
player would see (first audio, and whether generation ever falls behind
playback).

    bench/benchlock.sh -- .venv/bin/python bench/demo_stream.py --model models/mlx-q8-fp16 --variant s8-kv4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from bench import parse_variant  # noqa: E402
from bench_long import PARAGRAPH  # noqa: E402
from cases import REF_TEXT, REF_WAV  # noqa: E402
from omnivoice_mlx import OmniVoiceTTS  # noqa: E402
from omnivoice_mlx.audio import write_wav  # noqa: E402
from omnivoice_mlx.stream import generate_stream, split_clauses  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/mlx-q8-fp16"))
    ap.add_argument("--variant", default="s16-kv8")
    ap.add_argument("--text", default=PARAGRAPH)
    ap.add_argument("--out", default=str(ROOT / "out/stream"))
    ap.add_argument("--continuity", action="store_true", help="feed the previous clause as extra reference (worse, README §17)")
    ap.add_argument("--clauses-dir", default=None, help="also write every clause as <dir>/cNN.wav")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = parse_variant(args.variant)

    tts = OmniVoiceTTS(args.model)
    vp = tts.make_prompt(REF_WAV, REF_TEXT)
    tts.generate("你好。", vp, sampler=cfg, postprocess=False)  # warm-up
    clauses = split_clauses(args.text)
    print(f"{len(args.text)} chars -> {len(clauses)} clauses: {[len(c) for c in clauses]}")

    pieces = []
    play_cursor_ms = None   # wall time at which playback of everything so far would end
    first_ms = None
    underruns = 0
    for p in generate_stream(tts, args.text, vp, sampler=cfg, continuity=args.continuity):
        pieces.append(p)
        if args.clauses_dir:
            Path(args.clauses_dir).mkdir(parents=True, exist_ok=True)
            write_wav(Path(args.clauses_dir) / f"c{p.index:02d}.wav", p.audio, 24000)
        if first_ms is None:
            first_ms = p.ready_ms
            play_cursor_ms = p.ready_ms
        gap = p.ready_ms - play_cursor_ms  # > 0 means the player ran dry before this piece arrived
        if gap > 0:
            underruns += 1
        play_cursor_ms = max(play_cursor_ms, p.ready_ms) + p.seconds * 1000
        print(f"  #{p.index:<2} ready {p.ready_ms:7.0f} ms  synth {p.synth_ms:5.0f} ms  plays {p.seconds:4.2f}s  "
              f"slack {-gap:7.0f} ms  | {p.text}")
    total_audio = sum(p.seconds for p in pieces)
    total_synth = sum(p.synth_ms for p in pieces) / 1000
    print(f"\nfirst audio {first_ms:.0f} ms; {len(pieces)} pieces, {total_audio:.1f}s audio, synth {total_synth:.2f}s "
          f"(RTF {total_synth / total_audio:.3f}); playback would finish at {play_cursor_ms / 1000:.1f}s; underruns {underruns}")
    suffix = "-cont" if args.continuity else ""
    write_wav(out / f"stream-{args.variant}{suffix}.wav", np.concatenate([p.audio for p in pieces]), 24000)


if __name__ == "__main__":
    main()
