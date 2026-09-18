"""Several model flavours in ONE process, interleaved per sentence, so their
timings share the machine state (the only way to compare flavours on this box).

    bench/benchlock.sh -- .venv/bin/python bench/bench_models.py --tag flavours \
        --spec fp16=models/k2-fsa-OmniVoice:float16 q8fp16=models/mlx-q8-fp16 q4all=models/mlx-q4-all \
        --variants s32 s8 --runs 3

spec: name=dir[:dtype[:bits]] (dtype/bits only matter for the raw k2-fsa checkpoint;
converted dirs carry their own). The codec is loaded once and shared.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from bench import parse_variant  # noqa: E402
from cases import REF_TEXT, REF_WAV, sentence_set  # noqa: E402
from omnivoice_mlx import OmniVoiceTTS  # noqa: E402
from omnivoice_mlx.model import model_bytes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["s32", "s8"])
    ap.add_argument("--set", default="cases", choices=["cases", "asr", "all"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=str(ROOT / "out"))
    args = ap.parse_args()
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    models: dict[str, OmniVoiceTTS] = {}
    codec = None
    for spec in args.spec:
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        path, dtype = parts[0], (parts[1] if len(parts) > 1 else "bfloat16")
        bits = int(parts[2]) if len(parts) > 2 else 0
        tts = OmniVoiceTTS(path, dtype=dtype, bits=bits, codec=codec)
        codec = tts.codec
        models[name] = tts
        print(f"[{name}] {path} dtype={dtype} bits={bits}: {model_bytes(tts.model) / 1e6:.0f} MB, "
              f"load {tts.load_model_s:.1f}s", flush=True)
    mx.clear_cache()
    vp = next(iter(models.values())).make_prompt(REF_WAV, REF_TEXT)
    variants = {v: parse_variant(v) for v in args.variants}
    cases = sentence_set(args.set)
    for tts in models.values():
        for cfg in variants.values():
            tts.generate("你好。", vp, sampler=cfg, postprocess=False)
    mx.clear_cache()
    print(f"resident after warm-up {mx.get_active_memory() / 1e9:.2f} GB", flush=True)

    rows = []
    for run in range(args.runs):
        for name, text in cases:
            for mname, tts in models.items():
                for vname, cfg in variants.items():
                    r = tts.generate(text, vp, sampler=cfg, postprocess=False)
                    rows.append(dict(model=mname, variant=vname, case=name, run=run, T=r.T,
                                     unmask_ms=r.timing.unmask_ms, decode_ms=r.timing.decode_ms,
                                     synth_ms=r.timing.synth_ms, raw_s=r.raw_seconds, rtf=r.rtf,
                                     forwards=r.stats.forwards))
                    print(f"  r{run} {mname:<8} {vname:<6} {name:<9} unmask {r.timing.unmask_ms:6.0f} "
                          f"RTF {r.rtf:.3f} {r.timing.unmask_ms / max(r.stats.forwards, 1):5.1f} ms/step", flush=True)

    print(f"\n=== {args.tag}: median over {args.runs} runs ===")
    print(f"{'model':<8} {'variant':<6} " + " ".join(f"{c[0]:>9}" for c in cases) + "   RTF pooled")
    for mname in models:
        for vname in variants:
            cells = []
            for cname, _ in cases:
                sel = [r for r in rows if r["model"] == mname and r["variant"] == vname and r["case"] == cname]
                cells.append(statistics.median(r["unmask_ms"] / r["forwards"] for r in sel))
            allv = [r for r in rows if r["model"] == mname and r["variant"] == vname]
            pooled = sum(r["synth_ms"] for r in allv) / 1000 / sum(r["raw_s"] for r in allv)
            print(f"{mname:<8} {vname:<6} " + " ".join(f"{c:9.1f}" for c in cells) + f"   {pooled:.3f}")
    print(f"peak memory {mx.get_peak_memory() / 1e9:.2f} GB")
    (out / f"{args.tag}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
