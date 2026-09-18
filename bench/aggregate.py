"""Markdown tables from bench.py JSONs.

    .venv/bin/python bench/aggregate.py out/bf16 out/fp16 ...            # timing (cases): median per variant/case
    .venv/bin/python bench/aggregate.py --quality out/bf16-asr out/...   # CER / sim, from a quality.json you produced

``--quality`` reads ``<dir>/quality.json``: a list of
``{"dir", "tag", "name", "cer", "sim"}`` rows. Producing it is up to you — this
repository ships no recogniser or speaker model (see the README's 方法).
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def timing(dirs: list[Path]) -> None:
    print("| model | variant | opener ms/step | sentence ms/step | long ms/step | RTF opener | RTF sentence | RTF long | RTF pooled |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for d in dirs:
        for js in sorted(d.glob("*-cases.json")):
            data = json.loads(js.read_text())
            rows = data["rows"]
            tag = rows[0]["tag"]
            variants = list(dict.fromkeys(r["variant"] for r in rows))
            for v in variants:
                sel = [r for r in rows if r["variant"] == v]
                cells, rtfs = [], []
                for case in ("opener", "sentence", "long"):
                    s = [r for r in sel if r["case"] == case]
                    if not s:
                        cells.append("-"); rtfs.append("-"); continue
                    ms = statistics.median(r["unmask_ms"] / max(r["forwards"], 1) for r in s)
                    cells.append(f"{ms:.1f}")
                    rtfs.append(f"{statistics.median(r['rtf'] for r in s):.3f}")
                pooled = sum(r["synth_ms"] for r in sel) / 1000 / sum(r["raw_s"] for r in sel)
                print(f"| {tag} | {v} | " + " | ".join(cells) + " | " + " | ".join(rtfs) + f" | {pooled:.3f} |")


def quality(dirs: list[Path]) -> None:
    per = defaultdict(list)
    for d in dirs:
        q = d / "quality.json"
        if not q.exists():
            continue
        for r in json.loads(q.read_text()):
            per[(Path(r["dir"]).name, r["tag"])].append(r)
    print("| run | variant | n | CER mean | CER max | >20% | sim mean | sim min | 字/s |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for (dn, tag), rows in sorted(per.items()):
        cers = [r["cer"] for r in rows]
        sims = [r["sim"] for r in rows]
        rate = sum(r["chars"] for r in rows) / sum(r["audio_s"] for r in rows)
        print(f"| {dn} | {tag} | {len(rows)} | {statistics.mean(cers) * 100:.2f}% | {max(cers) * 100:.1f}% | "
              f"{sum(c > 0.2 for c in cers)} | {statistics.mean(sims):.3f} | {min(sims):.3f} | {rate:.2f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--quality", action="store_true")
    args = ap.parse_args()
    dirs = [Path(p) for p in args.dirs]
    (quality if args.quality else timing)(dirs)


if __name__ == "__main__":
    main()
