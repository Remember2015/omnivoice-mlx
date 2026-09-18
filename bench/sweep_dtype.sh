#!/usr/bin/env bash
# dtype / quantisation sweep: timing on the 3 cases (3 runs), then quality on the
# 20-sentence set (1 run) for every model flavour, all serialised through benchlock.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOCK="bench/benchlock.sh --load 6 --"
V="s32 s16 s8"
mkdir -p out
{
  echo "### timing (cases x3) $(date)"
  $LOCK $PY bench/bench.py --tag bf16 --dtype bfloat16 --variants $V --set cases --runs 3
  $LOCK $PY bench/bench.py --tag fp16 --dtype float16 --variants $V --set cases --runs 3
  $LOCK $PY bench/bench.py --tag fp32 --dtype float32 --variants $V --set cases --runs 3
  $LOCK $PY bench/bench.py --tag q8 --model models/mlx-q8 --variants $V --set cases --runs 3
  $LOCK $PY bench/bench.py --tag q4 --model models/mlx-q4 --variants $V --set cases --runs 3
  echo "### quality (asr set x1) $(date)"
  $LOCK $PY bench/bench.py --tag bf16-asr --dtype bfloat16 --variants $V --set asr --runs 1
  $LOCK $PY bench/bench.py --tag fp16-asr --dtype float16 --variants $V --set asr --runs 1
  $LOCK $PY bench/bench.py --tag q8-asr --model models/mlx-q8 --variants $V --set asr --runs 1
  $LOCK $PY bench/bench.py --tag q4-asr --model models/mlx-q4 --variants $V --set asr --runs 1
  $LOCK $PY bench/bench.py --tag fp32-asr --dtype float32 --variants $V --set asr --runs 1
  # wavs for each tag land in out/<tag>-asr/ — score them with your own harness
  echo "### done $(date)"
} 2>&1 | grep --line-buffered -v 'Warning\|kernel = \|Loading weights'
