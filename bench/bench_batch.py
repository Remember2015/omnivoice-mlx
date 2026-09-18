"""Throughput mode: the 20-sentence set through generate_batch with buckets of B
(sorted by length, padded rows, one masked SDPA per layer), fast path vs the
per-segment path, with GPU busy-ness and the achieved backbone GEMM rate.

    bench/benchlock.sh -- .venv/bin/python bench/bench_batch.py --model models/mlx-q8-fp16 --tag batch-q8 \
        --batch 1 2 4 8 16 --variants s16-kv8-ue3 s16-kv8-ue3-slow

Pooled RTF = total (unmask + decode) time / total raw audio of the 20 sentences.
Wavs of the largest B (first run) go to out/<tag>/ for scoring.
"""
from __future__ import annotations

import argparse
import json
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
from gpu_util import Sampler  # noqa: E402
from omnivoice_mlx import OmniVoiceTTS  # noqa: E402
from omnivoice_mlx.audio import write_wav  # noqa: E402

PARAMS = 440e6  # backbone linear parameters: GEMM FLOPs per row ≈ 2 * PARAMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/mlx-q8-fp16"))
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--head-dtype", default="float32")
    ap.add_argument("--variants", nargs="+", default=["s16-kv8-ue3"])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--set", default="asr", choices=["cases", "asr", "all"])
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=str(ROOT / "out"))
    args = ap.parse_args()
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    tts = OmniVoiceTTS(args.model, dtype=args.dtype, head_dtype=args.head_dtype)
    vp = tts.make_prompt(REF_WAV, REF_TEXT)
    cases = sentence_set(args.set)
    texts = [t for _, t in cases]
    variants = {v: parse_variant(v) for v in args.variants}
    for cfg in variants.values():
        tts.generate_batch(["你好。", "你好吗？", "今天天气不错。"], vp, sampler=cfg, postprocess=False, max_batch=3)
    mx.clear_cache()

    rows = []
    for run in range(args.runs):
        for vname, cfg in variants.items():
            for B in args.batch:
                with Sampler() as gpu:
                    t_all = time.perf_counter()
                    res = tts.generate_batch(texts, vp, sampler=cfg, max_batch=B)
                    wall = time.perf_counter() - t_all
                synth_ms = sum(r.timing.synth_ms for r in res)
                unmask_ms = sum(r.timing.unmask_ms for r in res)
                raw_s = sum(r.raw_seconds for r in res)
                # stats objects are shared per bucket: count each bucket once
                seen, tokens, fwds = set(), 0, 0
                for r in res:
                    if id(r.stats) not in seen:
                        seen.add(id(r.stats))
                        tokens += r.stats.tokens_through_backbone
                        fwds += r.stats.forwards
                tflops = tokens * 2 * PARAMS / (unmask_ms / 1000) / 1e12
                row = dict(variant=vname, B=B, run=run, n=len(texts), synth_s=synth_ms / 1000, raw_s=raw_s,
                           rtf=synth_ms / 1000 / raw_s, wall_s=wall, tokens_per_forward=tokens / max(fwds, 1),
                           buckets=len(seen), tflops=tflops, gpu_busy=gpu.summary(), peak_gb=mx.get_peak_memory() / 1e9)
                rows.append(row)
                if run == 0 and B == max(args.batch):
                    for (name, _), r in zip(cases, res):
                        write_wav(out / f"{args.tag}-{vname}-b{B}-{name}.wav", r.audio, 24000)
                print(f"  r{run} {vname:<16} B={B:<2} {raw_s:5.1f}s audio: synth {synth_ms / 1000:6.2f}s wall {wall:6.2f}s  "
                      f"pooled RTF {row['rtf']:.3f}  {row['tokens_per_forward']:5.0f} rows/fwd  {tflops:4.1f} TFLOPS  "
                      f"GPU {gpu.summary()[:12]}  peak {row['peak_gb']:.2f} GB", flush=True)
    print(f"\n=== {args.tag}: best-of-{args.runs} pooled RTF ===")
    for vname in variants:
        print(f"{vname:<16} " + "  ".join(f"B={B}: {min(r['rtf'] for r in rows if r['variant'] == vname and r['B'] == B):.3f}"
                                         for B in args.batch))
    (out / f"{args.tag}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
