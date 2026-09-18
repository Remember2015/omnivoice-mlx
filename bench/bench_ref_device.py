"""RTF bench for the OFFICIAL torch implementation, CPU vs MPS, interleaved.

    bench/benchlock.sh -- .venv-ref/bin/python bench/bench_ref_device.py \
        --devices cpu mps --steps 8 32 --runs 3 --tag ref-dev

One process loads the official model once per device, then walks
(case x device x steps) round-robin so every variant sees the same machine
state (absolute numbers on this box drift 10-40 % with load).

Timing matches bench.py: synth = unmask + codec decode, RTF against the raw
generated length (T x 40 ms), post-processing excluded. MPS work is queued
asynchronously, so the clock is stopped after ``torch.mps.synchronize()``.

Note on MPS: the official loader keeps the Higgs codec on CPU there (the
tokenizer has a conv with > 65536 output channels, unsupported by MPS), so the
decode column on the mps rows is CPU time by construction. PYTORCH_ENABLE_MPS_FALLBACK
is deliberately NOT set: an unsupported op must raise, not silently run on CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
    raise SystemExit("unset PYTORCH_ENABLE_MPS_FALLBACK: a silent CPU fallback would fake the mps numbers")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from cases import REF_TEXT, REF_WAV, sentence_set  # noqa: E402


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/k2-fsa-OmniVoice"))
    ap.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    ap.add_argument("--dtypes", nargs="+", default=["float32"], choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--steps", nargs="+", type=int, default=[8, 32])
    ap.add_argument("--set", default="cases", choices=["cases", "asr", "all"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=str(ROOT / "out"))
    args = ap.parse_args()

    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    from omnivoice import OmniVoice, OmniVoiceGenerationConfig

    models, prompts = {}, {}
    for dev in args.devices:
        for dt in args.dtypes:
            key = (dev, dt)
            t0 = time.perf_counter()
            m = OmniVoice.from_pretrained(args.model, device_map=dev, torch_dtype=getattr(torch, dt))
            m.eval()
            load_s = time.perf_counter() - t0
            got = str(m.device)
            if not got.startswith(dev):
                raise SystemExit(f"asked for {dev}, model landed on {got}")
            pdt = {str(p.dtype) for p in m.parameters()}
            print(f"[{dev}/{dt}] loaded in {load_s:.1f}s  backbone {got} params {sorted(pdt)} "
                  f"{sum(p.numel() for p in m.parameters()) / 1e6:.0f}M, codec on {m.audio_tokenizer.device} "
                  f"{next(m.audio_tokenizer.parameters()).dtype}, torch {torch.__version__}", flush=True)
            models[key] = m
            prompts[key] = m.create_voice_clone_prompt(ref_audio=str(REF_WAV), ref_text=REF_TEXT)

    variants = [(dev, dt, st) for dev in args.devices for dt in args.dtypes for st in args.steps]
    cases = sentence_set(args.set)

    for dev, dt, steps in variants:  # warm-up: lazy kernels / first touch
        m, vp = models[(dev, dt)], prompts[(dev, dt)]
        gen = OmniVoiceGenerationConfig(num_step=steps)
        with torch.no_grad():
            task = m._preprocess_all(text="你好。", language="zh", voice_clone_prompt=vp)
            m._generate_iterative(task, gen)
        sync(dev)
    print("warm-up done\n", flush=True)

    rows: list[dict] = []
    for run in range(args.runs):
        for name, text in cases:
            for dev, dt, steps in variants:
                m, vp = models[(dev, dt)], prompts[(dev, dt)]
                gen = OmniVoiceGenerationConfig(num_step=steps)
                torch.manual_seed(0)
                with torch.no_grad():
                    task = m._preprocess_all(text=text, language="zh", voice_clone_prompt=vp)
                    sync(dev)
                    t0 = time.perf_counter()
                    tokens = m._generate_iterative(task, gen)[0]
                    sync(dev)
                    unmask_ms = (time.perf_counter() - t0) * 1000
                    t0 = time.perf_counter()
                    audio = m._decode_and_post_process(tokens.detach(), vp.ref_rms, gen)
                    sync(dev)
                    decode_ms = (time.perf_counter() - t0) * 1000
                T = int(task.target_lens[0])
                raw_s = T * 960 / m.sampling_rate
                rtf = (unmask_ms + decode_ms) / 1000 / raw_s
                tag_v = f"{dev}-{dt[:2]}{dt[-2:]}-s{steps}"
                rows.append(dict(device=dev, dtype=dt, steps=steps, variant=tag_v, case=name, run=run,
                                 T=T, raw_s=raw_s, out_s=len(audio) / m.sampling_rate,
                                 unmask_ms=unmask_ms, decode_ms=decode_ms, rtf=rtf))
                print(f"  r{run} {tag_v:<18} {name:<9} T={T:>3} raw {raw_s:5.2f}s  "
                      f"unmask {unmask_ms:7.0f} decode {decode_ms:6.0f} ms  RTF {rtf:6.3f}  "
                      f"{unmask_ms / steps:6.1f} ms/step", flush=True)

    print(f"\n=== {args.tag}: median over {args.runs} runs ===")
    print(f"{'variant':<18} {'case':<9} {'T':>4} {'unmask':>8} {'decode':>7} {'RTF':>7} {'ms/step':>8}")
    summary = {}
    for dev, dt, steps in variants:
        v = f"{dev}-{dt[:2]}{dt[-2:]}-s{steps}"
        for name, _ in cases:
            sel = [r for r in rows if r["variant"] == v and r["case"] == name]
            med = lambda k: statistics.median(r[k] for r in sel)  # noqa: E731
            print(f"{v:<18} {name:<9} {sel[0]['T']:>4} {med('unmask_ms'):8.0f} {med('decode_ms'):7.0f} "
                  f"{med('rtf'):7.3f} {med('unmask_ms') / steps:8.1f}")
        allv = [r for r in rows if r["variant"] == v]
        pooled = sum(r["unmask_ms"] + r["decode_ms"] for r in allv) / 1000 / sum(r["raw_s"] for r in allv)
        summary[v] = dict(rtf_pooled=pooled, median_rtf=statistics.median(r["rtf"] for r in allv),
                          unmask_ms=statistics.median(r["unmask_ms"] for r in allv),
                          decode_ms=statistics.median(r["decode_ms"] for r in allv))
        print(f"{v:<18} {'POOLED':<9} {'':>4} {'':>8} {'':>7} {pooled:7.3f}")
    (out / f"{args.tag}-{args.set}.json").write_text(
        json.dumps(dict(rows=rows, summary=summary, args=vars(args)), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
