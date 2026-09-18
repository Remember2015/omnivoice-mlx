"""RTF bench for the MLX port: one model, several sampler variants, interleaved.

    bench/benchlock.sh -- .venv/bin/python bench/bench.py --tag bf16 \
        --variants s32 s16 s8 s16-kv4 s8-cfg0.5 --set cases --runs 3

Every sentence is synthesised once per variant per round, variants in
round-robin, so the variants see the same machine state (absolute numbers on
this box drift 10-40 % with load; only same-process interleaved numbers are
comparable). Reports per variant: median synth ms (unmask + codec decode),
RTF against the raw generated length, tokens through the backbone, and writes
``out/<tag>/<tag>-<variant>-<case>.wav`` (first round) for whatever scores them.

Variant syntax: ``s<steps>[-cfg<frac>][-kv<refresh>]``, e.g. ``s16-kv4-cfg0.5``.
"""
from __future__ import annotations

import argparse
import json
import re
import resource
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from cases import REF_TEXT, REF_WAV, sentence_set  # noqa: E402
from omnivoice_mlx import OmniVoiceTTS, SamplerConfig  # noqa: E402
from omnivoice_mlx.audio import write_wav  # noqa: E402


def parse_variant(s: str) -> SamplerConfig:
    m = re.fullmatch(r"s(\d+)((?:-(?:cfg[\d.]+|kv\d+|th[\d.]+|ue\d+|slow|sync))*)", s)
    if not m:
        raise SystemExit(f"bad variant {s!r}; expected s<steps>[-cfg<frac>][-kv<n>][-th<prob>][-sync]")
    cfg = SamplerConfig(num_steps=int(m.group(1)))
    for opt in m.group(2).split("-"):
        if opt.startswith("cfg"):
            cfg.cfg_until = float(opt[3:])
        elif opt.startswith("kv"):
            cfg.cache_refresh = int(opt[2:])
        elif opt.startswith("th"):
            cfg.conf_threshold = float(opt[2:])
        elif opt.startswith("ue"):
            cfg.uncond_every = int(opt[2:])
        elif opt == "slow":
            cfg.fast_path = False
        elif opt == "sync":
            cfg.sync_every_step = True
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/k2-fsa-OmniVoice"))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--bits", type=int, default=0)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--quantize-embed", action="store_true")
    ap.add_argument("--quantize-heads", action="store_true")
    ap.add_argument("--head-dtype", default="float32")
    ap.add_argument("--variants", nargs="+", default=["s32", "s16", "s8"])
    ap.add_argument("--set", default="cases", choices=["cases", "asr", "all"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--seed", type=int, default=None, help="fixed seed per sentence (default: random)")
    ap.add_argument("--no-wav", action="store_true")
    ap.add_argument("--wav-all-runs", action="store_true", help="also keep runs > 0 (under out/<tag>/r<run>/)")
    args = ap.parse_args()

    variants = {v: parse_variant(v) for v in args.variants}
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    tts = OmniVoiceTTS(args.model, dtype=args.dtype, bits=args.bits, group_size=args.group_size,
                       quantize_embed=args.quantize_embed, quantize_heads=args.quantize_heads,
                       head_dtype=args.head_dtype)
    from omnivoice_mlx.model import model_bytes
    print(f"[{args.tag}] model {args.dtype} bits={args.bits} head={args.head_dtype}: "
          f"{model_bytes(tts.model) / 1e6:.0f} MB, load {tts.load_model_s:.1f}s, codec {tts.load_codec_s:.1f}s, "
          f"mlx {mx.__version__}", flush=True)
    vp = tts.make_prompt(REF_WAV, REF_TEXT)
    print(f"reference: {vp.ref_tokens.shape[0]} tokens ({vp.ref_tokens.shape[0] / 25:.2f}s)", flush=True)

    cases = sentence_set(args.set)
    # warm-up: kernels / first touch, every variant once on a short sentence
    for cfg in variants.values():
        tts.generate("你好。", vp, sampler=cfg, postprocess=False)
    mx.clear_cache()
    base_mem = mx.get_active_memory()

    rows: list[dict] = []
    for run in range(args.runs):
        for name, text in cases:
            for vname, cfg in variants.items():
                r0 = resource.getrusage(resource.RUSAGE_SELF)
                t0 = time.perf_counter()
                res = tts.generate(text, vp, sampler=cfg, seed=args.seed, postprocess=True)
                wall = time.perf_counter() - t0
                r1 = resource.getrusage(resource.RUSAGE_SELF)
                cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
                rows.append(dict(tag=args.tag, variant=vname, case=name, run=run, chars=len(text),
                                 P=res.prompt_len, T=res.T, raw_s=res.raw_seconds,
                                 out_s=len(res.audio) / 24000, unmask_ms=res.timing.unmask_ms,
                                 decode_ms=res.timing.decode_ms, post_ms=res.timing.post_ms,
                                 synth_ms=res.timing.synth_ms, wall_ms=wall * 1000, rtf=res.rtf,
                                 cpu_pct=cpu / wall * 100, forwards=res.stats.forwards,
                                 tokens=res.stats.tokens_through_backbone))
                if not args.no_wav and (run == 0 or args.wav_all_runs):
                    d = out if run == 0 else out / f"r{run}"
                    d.mkdir(exist_ok=True)
                    write_wav(d / f"{args.tag}-{vname}-{name}.wav", res.audio, 24000)
                print(f"  r{run} {vname:<14} {name:<9} P={res.prompt_len:>3} T={res.T:>3} raw {res.raw_seconds:5.2f}s "
                      f"unmask {res.timing.unmask_ms:6.0f} decode {res.timing.decode_ms:4.0f} ms  "
                      f"RTF {res.rtf:.3f}  {res.timing.unmask_ms / max(res.stats.steps, 1):5.1f} ms/step", flush=True)

    print(f"\n=== {args.tag}: median over {args.runs} runs, per variant ===")
    print(f"{'variant':<14} {'case':<9} {'T':>4} {'unmask':>7} {'decode':>6} {'RTF':>6} {'ms/step':>8} {'tok/step':>8} {'CPU%':>5}")
    summary = {}
    for vname in variants:
        for name, _ in cases:
            sel = [r for r in rows if r["variant"] == vname and r["case"] == name]
            med = lambda k: statistics.median(r[k] for r in sel)  # noqa: E731
            steps = max(variants[vname].num_steps, 1)
            print(f"{vname:<14} {name:<9} {sel[0]['T']:>4} {med('unmask_ms'):7.0f} {med('decode_ms'):6.0f} "
                  f"{med('rtf'):6.3f} {med('unmask_ms') / sel[0]['forwards']:8.1f} "
                  f"{sel[0]['tokens'] / sel[0]['forwards']:8.0f} {med('cpu_pct'):5.0f}")
        allv = [r for r in rows if r["variant"] == vname]
        rtf_all = sum(r["synth_ms"] for r in allv) / 1000 / sum(r["raw_s"] for r in allv)
        summary[vname] = dict(rtf_pooled=rtf_all, median_rtf=statistics.median(r["rtf"] for r in allv))
        print(f"{vname:<14} {'POOLED':<9} {'':>4} {'':>7} {'':>6} {rtf_all:6.3f}")
    print(f"peak memory {mx.get_peak_memory() / 1e9:.2f} GB, resident after warm-up {base_mem / 1e9:.2f} GB")
    (out / f"{args.tag}-{args.set}.json").write_text(json.dumps(dict(rows=rows, summary=summary, args=vars(args)),
                                                               ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
