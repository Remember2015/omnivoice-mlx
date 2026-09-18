"""mlx-audio's OmniVoice port under the same conditions as bench.py: same
reference tokens (ours, encoded once), same target length (passed as
``duration_s`` so its own estimator is bypassed), same sentences, steps and
timing definition (unmask + codec decode over the raw generated length).

    bench/benchlock.sh -- .venv/bin/python bench/bench_mlxaudio.py --model models/mlxaudio-bf16 --tag mlxaudio-bf16
"""
from __future__ import annotations

import argparse
import json
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
from omnivoice_mlx.audio import write_wav  # noqa: E402
from omnivoice_mlx.duration import RuleDurationEstimator  # noqa: E402
from omnivoice_mlx.text import add_punctuation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/mlxaudio-bf16"))
    ap.add_argument("--steps", type=int, nargs="+", default=[32, 16, 8])
    ap.add_argument("--set", default="cases", choices=["cases", "asr", "all"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tag", default="mlxaudio-bf16")
    ap.add_argument("--out", default=str(ROOT / "out"))
    args = ap.parse_args()
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    from mlx_audio.tts.models.omnivoice.utils import create_voice_clone_prompt
    from mlx_audio.tts.utils import load_model

    t0 = time.perf_counter()
    model = load_model(Path(args.model))
    print(f"[{args.tag}] loaded {args.model} in {time.perf_counter() - t0:.1f}s", flush=True)
    ref_tokens = create_voice_clone_prompt(str(REF_WAV), tokenizer=model.audio_tokenizer, max_duration_s=15.0)
    mx.eval(ref_tokens)
    ref_text = add_punctuation(REF_TEXT)
    est = RuleDurationEstimator()
    print(f"reference: {ref_tokens.shape[0]} tokens", flush=True)

    def gen(text, steps):
        T = max(1, int(est.estimate_duration(text, ref_text, int(ref_tokens.shape[0]))))
        t0 = time.perf_counter()
        res = next(model.generate(text=text, language="zh", ref_tokens=ref_tokens, ref_text=REF_TEXT,
                                  num_steps=steps, duration_s=T / 25.0))
        audio = np.asarray(res.audio, dtype=np.float32).reshape(-1)
        total = time.perf_counter() - t0
        return audio, total, res.processing_time_seconds, T

    cases = sentence_set(args.set)
    for s in args.steps:
        gen("你好。", s)
    rows = []
    for run in range(args.runs):
        for name, text in cases:
            for steps in args.steps:
                audio, total, unmask_s, T = gen(text, steps)
                raw = T * 960 / 24000
                rows.append(dict(tag=args.tag, variant=f"s{steps}", case=name, run=run, T=T, raw_s=raw,
                                 unmask_ms=unmask_s * 1000, decode_ms=(total - unmask_s) * 1000,
                                 synth_ms=total * 1000, rtf=total / raw))
                if run == 0:
                    write_wav(out / f"{args.tag}-s{steps}-{name}.wav", audio, 24000)
                print(f"  r{run} s{steps:<3} {name:<9} T={T:>3} raw {raw:5.2f}s unmask {unmask_s * 1000:6.0f} "
                      f"decode {(total - unmask_s) * 1000:4.0f} ms  RTF {total / raw:.3f}  "
                      f"{unmask_s * 1000 / steps:5.1f} ms/step", flush=True)
    print(f"\n=== {args.tag}: median over {args.runs} runs ===")
    for steps in args.steps:
        for name, _ in cases:
            sel = [r for r in rows if r["variant"] == f"s{steps}" and r["case"] == name]
            print(f"s{steps:<3} {name:<9} T={sel[0]['T']:>3} unmask {statistics.median(r['unmask_ms'] for r in sel):6.0f} "
                  f"decode {statistics.median(r['decode_ms'] for r in sel):4.0f} RTF {statistics.median(r['rtf'] for r in sel):.3f} "
                  f"{statistics.median(r['unmask_ms'] for r in sel) / steps:5.1f} ms/step")
        allv = [r for r in rows if r["variant"] == f"s{steps}"]
        print(f"s{steps:<3} POOLED RTF {sum(r['synth_ms'] for r in allv) / 1000 / sum(r['raw_s'] for r in allv):.3f}")
    print(f"peak memory {mx.get_peak_memory() / 1e9:.2f} GB")
    (out / f"{args.tag}-{args.set}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
