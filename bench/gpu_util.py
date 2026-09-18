"""GPU busy-ness while synthesising, from the accelerator's own counters
(``ioreg -c IOAccelerator`` → "Device Utilization %", no sudo needed), next to
the achieved GEMM rate (FLOPs pushed through the backbone / time) so the two
kinds of "utilisation" are not confused: busy % says whether the GPU had work
queued, TFLOPS says how well that work used the ALUs.

    bench/benchlock.sh -- .venv/bin/python bench/gpu_util.py --variants s16-kv8-ue3 s32 --seconds 8
"""
from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from cases import CASES, REF_TEXT, REF_WAV  # noqa: E402

_KEYS = ("Device Utilization %", "Renderer Utilization %", "Tiler Utilization %")
_RE = re.compile(r'"(Device|Renderer|Tiler) Utilization %"=(\d+)')


def read_util() -> dict[str, int]:
    out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
    return {f"{k} Utilization %": int(v) for k, v in _RE.findall(out)}


class Sampler:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.samples: list[dict[str, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(read_util())
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()

    def summary(self, key: str = "Device Utilization %") -> str:
        v = [s[key] for s in self.samples if key in s]
        if not v:
            return "no samples"
        return f"mean {statistics.mean(v):5.1f} %  median {statistics.median(v):3.0f}  min {min(v):3d}  max {max(v):3d}  (n={len(v)})"


def idle_baseline(seconds: float = 3.0) -> str:
    with Sampler() as s:
        time.sleep(seconds)
    return s.summary()


def run_omnivoice(args):
    import mlx.core as mx

    from bench import parse_variant
    from omnivoice_mlx import OmniVoiceTTS

    tts = OmniVoiceTTS(args.model)
    vp = tts.make_prompt(REF_WAV, REF_TEXT)
    params = 440e6  # backbone linear params (28 layers), the GEMM FLOPs per token ≈ 2 * params
    for v in args.variants:
        cfg = parse_variant(v)
        tts.generate("你好。", vp, sampler=cfg, postprocess=False)
        mx.synchronize()
        tokens = unmask_ms = 0.0
        n = 0
        with Sampler() as s:
            t_end = time.perf_counter() + args.seconds
            while time.perf_counter() < t_end:
                for _, text in CASES:
                    r = tts.generate(text, vp, sampler=cfg, postprocess=False)
                    tokens += r.stats.tokens_through_backbone
                    unmask_ms += r.timing.unmask_ms
                    n += 1
        tflops = tokens * 2 * params / (unmask_ms / 1000) / 1e12
        print(f"[omnivoice {v:<12}] {n} sentences  GPU busy: {s.summary()}  |  backbone GEMM {tflops:4.1f} TFLOPS achieved "
              f"({tflops / 13.6 * 100:3.0f} % of 13.6 peak)", flush=True)
    if args.paragraph:
        from bench_long import PARAGRAPH
        from omnivoice_mlx.text import chunk_text_punctuation
        chunk = chunk_text_punctuation(PARAGRAPH, 92, 3)[0]
        cfg = parse_variant(args.variants[0])
        tokens = unmask_ms = 0.0
        n = 0
        with Sampler() as s:
            t_end = time.perf_counter() + args.seconds
            while time.perf_counter() < t_end:
                r = tts.generate(chunk, vp, sampler=cfg, postprocess=False)
                tokens += r.stats.tokens_through_backbone
                unmask_ms += r.timing.unmask_ms
                n += 1
        tflops = tokens * 2 * params / (unmask_ms / 1000) / 1e12
        print(f"[omnivoice {args.variants[0]} 15s-chunk T={r.T}] {n} chunks  GPU busy: {s.summary()}  |  GEMM {tflops:4.1f} TFLOPS "
              f"({tflops / 13.6 * 100:3.0f} %)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/mlx-q8-fp16"))
    ap.add_argument("--variants", nargs="+", default=["s16-kv8-ue3", "s32"])
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--paragraph", action="store_true")
    args = ap.parse_args()
    print(f"idle baseline: {idle_baseline()}", flush=True)
    run_omnivoice(args)


if __name__ == "__main__":
    main()
