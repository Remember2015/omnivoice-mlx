"""Fused projections + batch-2 cached steps: exactness (fp32, same seed) and
speed (q8-fp16, four combinations interleaved in one process).

    bench/benchlock.sh -- .venv/bin/python bench/test_fastpath.py
"""
from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from cases import CASES, REF_TEXT, REF_WAV  # noqa: E402
from omnivoice_mlx import OmniVoiceTTS, SamplerConfig  # noqa: E402
from omnivoice_mlx.sampler import unmask  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/mlx-q8-fp16"))
    ap.add_argument("--raw", default=str(ROOT / "models/k2-fsa-OmniVoice"))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--skip-exact", action="store_true")
    args = ap.parse_args()

    if not args.skip_exact:
        print("== exactness, fp32, seed 3, s16-kv8 (fast vs segment path) and s16 (fused vs unfused)")
        a = OmniVoiceTTS(args.raw, dtype="float32", fuse=True)
        b = OmniVoiceTTS(args.raw, dtype="float32", fuse=False, codec=a.codec)
        vp = a.make_prompt(REF_WAV, REF_TEXT)
        for name, text in CASES:
            p = a.build_prompt(text, vp, language="zh")
            T = a.estimate_tokens(text, vp)
            outs = {}
            for label, tts, cfg in [("fused+fast", a, SamplerConfig(16, cache_refresh=8, fast_path=True)),
                                    ("fused+slow", a, SamplerConfig(16, cache_refresh=8, fast_path=False)),
                                    ("unfused+slow", b, SamplerConfig(16, cache_refresh=8, fast_path=False)),
                                    ("fused s16", a, SamplerConfig(16, cache_refresh=0)),
                                    ("unfused s16", b, SamplerConfig(16, cache_refresh=0))]:
                outs[label] = np.asarray(unmask(tts.model, p, T, cfg, seed=3))
            ref = outs["unfused+slow"]
            print(f"  {name:<9} T={T:>3}  fused+fast vs unfused+slow {(outs['fused+fast'] == ref).mean():.4f}  "
                  f"fused+slow vs unfused+slow {(outs['fused+slow'] == ref).mean():.4f}  "
                  f"fused vs unfused (no cache) {(outs['fused s16'] == outs['unfused s16']).mean():.4f}")
        del a, b
        mx.clear_cache()

    print("\n== speed, q8-fp16, interleaved")
    fused = OmniVoiceTTS(args.model, fuse=True)
    plain = OmniVoiceTTS(args.model, fuse=False, codec=fused.codec)
    vp = fused.make_prompt(REF_WAV, REF_TEXT)
    combos = {"fused+fast": (fused, SamplerConfig(fast_path=True)),
              "fused+slow": (fused, SamplerConfig(fast_path=False)),
              "plain+fast": (plain, SamplerConfig(fast_path=True)),
              "plain+slow": (plain, SamplerConfig(fast_path=False))}
    for tts, cfg in combos.values():
        tts.generate("你好。", vp, sampler=cfg, postprocess=False)
    mx.clear_cache()
    rows = []
    for run in range(args.runs):
        for name, text in CASES:
            for label, (tts, cfg) in combos.items():
                r = tts.generate(text, vp, sampler=cfg, postprocess=False)
                rows.append((label, name, r.timing.unmask_ms / r.stats.forwards, r.rtf, r.timing.synth_ms, r.raw_seconds))
    print(f"  {'combo':<12} " + " ".join(f"{c[0]:>10}" for c in CASES) + "   RTF pooled")
    for label in combos:
        cells = [statistics.median(r[2] for r in rows if r[0] == label and r[1] == c) for c, _ in CASES]
        sel = [r for r in rows if r[0] == label]
        pooled = sum(r[4] for r in sel) / 1000 / sum(r[5] for r in sel)
        print(f"  {label:<12} " + " ".join(f"{c:10.1f}" for c in cells) + f"   {pooled:.3f}   (ms/step)")


if __name__ == "__main__":
    main()
