"""Long text (a ~1 minute paragraph): official chunked path, chunks sequential
vs all chunks in one packed loop.

    bench/benchlock.sh -- .venv/bin/python bench/bench_long.py --dtype float16 --tag long-fp16 \
        --variants s32 s16-kv8 s8-kv4 --runs 2
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
from cases import REF_TEXT, REF_WAV  # noqa: E402
from omnivoice_mlx import OmniVoiceTTS  # noqa: E402
from omnivoice_mlx.audio import write_wav  # noqa: E402
from omnivoice_mlx.text import chunk_text_punctuation  # noqa: E402

PARAGRAPH = (
    "各位早上好，先说一下今天的安排。上午九点半在三楼会议室开季度复盘，市场、产品和研发三个组各准备十分钟的汇报，"
    "重点讲一下上个季度没有完成的目标和原因，不用铺陈成绩。十一点以后是自由讨论，张经理会把预算的初步方案发到群里，"
    "大家提前看一眼。中午食堂有新的窗口，据说是川菜，想尝试的同事可以去试试，不过据前天去过的人说辣得有点狠。"
    "下午两点，法务的同事过来讲新的数据合规要求，主要涉及用户录音的存储和删除周期，做语音功能的同学务必参加。"
    "四点之前请把本周的周报交上来，格式还是老样子，一页纸以内，写清楚做了什么、卡在哪里、需要谁帮忙。"
    "最后提醒一句，周五下午公司组织体检，早上不要吃早饭，记得带上身份证。今天就这些，散会。"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/k2-fsa-OmniVoice"))
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--variants", nargs="+", default=["s32", "s16-kv8", "s8-kv4"])
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=str(ROOT / "out"))
    args = ap.parse_args()
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    tts = OmniVoiceTTS(args.model, dtype=args.dtype)
    vp = tts.make_prompt(REF_WAV, REF_TEXT)
    T_est = tts.estimate_tokens(PARAGRAPH, vp)
    chunk_len = int(15.0 * 25 / (T_est / len(PARAGRAPH)))
    chunks = chunk_text_punctuation(PARAGRAPH, chunk_len, 3)
    print(f"paragraph {len(PARAGRAPH)} chars, estimated {T_est / 25:.1f}s -> {len(chunks)} chunks "
          f"({[len(c) for c in chunks]} chars)", flush=True)
    variants = {v: parse_variant(v) for v in args.variants}
    tts.generate_batch(["你好。", "你好吗？"], vp, sampler=variants[args.variants[0]], postprocess=False)
    mx.clear_cache()

    rows = []
    for run in range(args.runs):
        for vname, cfg in variants.items():
            for batched in (False, True):
                t0 = time.perf_counter()
                r = tts.generate_long(PARAGRAPH, vp, sampler=cfg, batch_chunks=batched)
                wall = time.perf_counter() - t0
                rows.append(dict(variant=vname, batched=batched, run=run, raw_s=r.raw_seconds, out_s=len(r.audio) / 24000,
                                 synth_ms=r.timing.synth_ms, wall_s=wall, rtf=r.rtf, forwards=r.stats.forwards,
                                 peak_gb=mx.get_peak_memory() / 1e9))
                if run == 0:
                    write_wav(out / f"{args.tag}-{vname}-{'batch' if batched else 'seq'}.wav", r.audio, 24000)
                print(f"  r{run} {vname:<8} {'batched' if batched else 'sequential':<10} raw {r.raw_seconds:5.1f}s out {len(r.audio) / 24000:5.1f}s "
                      f"synth {r.timing.synth_ms / 1000:6.2f}s wall {wall:6.2f}s RTF {r.rtf:.3f} fwd {r.stats.forwards} "
                      f"peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
    print(f"\n=== {args.tag}: median over {args.runs} runs ===")
    for vname in variants:
        for batched in (False, True):
            sel = [r for r in rows if r["variant"] == vname and r["batched"] == batched]
            print(f"{vname:<8} {'batched' if batched else 'sequential':<10} RTF {statistics.median(r['rtf'] for r in sel):.3f} "
                  f"wall {statistics.median(r['wall_s'] for r in sel):5.2f}s")
    (out / f"{args.tag}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
