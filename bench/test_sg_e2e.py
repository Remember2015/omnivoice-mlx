"""End-to-end effect of the custom simdgroup GEMM (kernels_sg via
SmallMQuantizedLinear): exactness against the MLX path and interleaved timing,
two models in one process sharing the codec.

    bench/benchlock.sh -- .venv/bin/python bench/test_sg_e2e.py
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from cases import CASES, REF_TEXT, REF_WAV, sentence_set  # noqa: E402
from omnivoice_mlx import OmniVoiceTTS, SamplerConfig  # noqa: E402
from omnivoice_mlx.sampler import unmask  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/mlx-q8-fp16"))
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    base = OmniVoiceTTS(args.model, custom_gemm=False)
    sg = OmniVoiceTTS(args.model, custom_gemm=True, codec=base.codec)
    vp = base.make_prompt(REF_WAV, REF_TEXT)
    det = SamplerConfig(position_temperature=0.0)
    print("== exactness (q8-fp16, deterministic), custom kernel vs MLX qmm")
    for name, text in CASES:
        p = base.build_prompt(text, vp, language="zh")
        T = base.estimate_tokens(text, vp)
        a = np.asarray(unmask(base.model, p, T, det))
        b = np.asarray(unmask(sg.model, p, T, det))
        print(f"  {name:<9} T={T:>3} token agreement {(a == b).mean():.4f}")

    cfg = SamplerConfig()
    for tts in (base, sg):
        tts.generate("你好。", vp, sampler=cfg, postprocess=False)
    mx.clear_cache()
    print(f"\n== timing, {args.runs} runs interleaved (default sampler {cfg.tag()})")
    rows = []
    for run in range(args.runs):
        for name, text in CASES:
            for label, tts in (("mlx", base), ("custom", sg)):
                r = tts.generate(text, vp, sampler=cfg, postprocess=False)
                rows.append((label, name, r.timing.unmask_ms, r.rtf, r.timing.synth_ms, r.raw_seconds))
    print(f"  {'':<8} " + " ".join(f"{c[0]:>12}" for c in CASES) + "   pooled RTF")
    for label in ("mlx", "custom"):
        cells = [statistics.median(r[2] for r in rows if r[0] == label and r[1] == c) for c, _ in CASES]
        sel = [r for r in rows if r[0] == label]
        pooled = sum(r[4] for r in sel) / 1000 / sum(r[5] for r in sel)
        print(f"  {label:<8} " + " ".join(f"{c:9.0f} ms" for c in cells) + f"   {pooled:.3f}")

    asr = sentence_set("asr")
    print(f"\n== 20-sentence set, pooled RTF, {args.runs} runs interleaved")
    tot = {"mlx": [0.0, 0.0], "custom": [0.0, 0.0]}
    for run in range(args.runs):
        for _, text in asr:
            for label, tts in (("mlx", base), ("custom", sg)):
                r = tts.generate(text, vp, sampler=cfg, postprocess=False)
                tot[label][0] += r.timing.synth_ms / 1000
                tot[label][1] += r.raw_seconds
    for label in ("mlx", "custom"):
        print(f"  {label:<8} RTF {tot[label][0] / tot[label][1]:.4f}")


if __name__ == "__main__":
    main()
