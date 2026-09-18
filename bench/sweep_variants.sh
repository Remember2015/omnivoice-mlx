#!/usr/bin/env bash
# Sampler-variant sweep on one model flavour: timing on the 3 cases (3 runs,
# interleaved in one process) then quality on the 20-sentence set (1 run).
#   DTYPE=bfloat16 TAG=bf16 bench/sweep_variants.sh [extra bench args]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOCK="bench/benchlock.sh --load 6 --"
DTYPE=${DTYPE:-bfloat16}
TAG=${TAG:-bf16}
MODEL=${MODEL:-models/k2-fsa-OmniVoice}
V=${VARIANTS:-"s32 s16 s8 s16-kv2 s16-kv4 s16-kv8 s8-kv2 s8-kv4 s16-cfg0.5 s16-cfg0.75 s8-cfg0.5 s16-kv4-cfg0.5 s32-th0.9 s16-th0.9 s16-th0.95 s16-kv4-th0.9"}
mkdir -p out
{
  echo "### variants timing (cases x3) $TAG $(date)"
  $LOCK $PY bench/bench.py --tag "var-$TAG" --model "$MODEL" --dtype "$DTYPE" --variants $V --set cases --runs 3 "$@"
  echo "### variants quality (asr x1) $TAG $(date)"
  $LOCK $PY bench/bench.py --tag "var-$TAG-asr" --model "$MODEL" --dtype "$DTYPE" --variants $V --set asr --runs 1 "$@"
  echo "### eval $(date)"
  # wavs land in out/var-$TAG-asr/ — score them with your own harness
  echo "### done $(date)"
} 2>&1 | grep --line-buffered -v 'Warning\|kernel = \|Loading weights'
