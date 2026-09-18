#!/usr/bin/env bash
# Run a heavy (model-loading, GPU) command only when the machine is idle and no
# other benchmark holds the lock. Every benchmark / parity run that loads a model
# goes through this, so concurrent engineers serialise instead of skewing each
# other's timings, and a machine the owner is using for something else is left
# alone until it quiets down.
#
#   bench/benchlock.sh [--max-wait SECS] [--load MAX_LOAD1] -- <command...>
#
# Idle = 1-minute load average below MAX_LOAD1 (default 6.0 on this 12-core M2
# Max; a proxy daemon and a couple of helpers idle at ~1.5 on their own) AND no
# Python process other than ourselves using more than 30 % CPU (an MLX TTS loop
# sits at ~36 %; every model job here is Python).
# The lock is .cache/bench.lock (mkdir is atomic); a lock older than 45 min is
# treated as abandoned, and so is one whose owner pid is gone.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCK="$ROOT/.cache/bench.lock"
MAX_WAIT=2700
MAX_LOAD=6.0
while [ $# -gt 0 ]; do
  case "$1" in
    --max-wait) MAX_WAIT="$2"; shift 2 ;;
    --load) MAX_LOAD="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "benchlock: unknown option $1" >&2; exit 2 ;;
  esac
done
[ $# -gt 0 ] || { echo "benchlock: no command" >&2; exit 2; }
mkdir -p "$ROOT/.cache"

waited=0
# 1. the lock
while ! mkdir "$LOCK" 2>/dev/null; do
  owner_pid=$(awk '{print $1}' "$LOCK/owner" 2>/dev/null || true)
  if [ -d "$LOCK" ] && { [ -z "$owner_pid" ] || ! kill -0 "$owner_pid" 2>/dev/null || \
       [ "$(( $(date +%s) - $(stat -f %m "$LOCK") ))" -gt 2700 ]; }; then
    echo "benchlock: removing stale lock ($(cat "$LOCK/owner" 2>/dev/null))" >&2
    rm -rf "$LOCK"; continue
  fi
  [ "$waited" -lt "$MAX_WAIT" ] || { echo "benchlock: gave up waiting for the lock" >&2; exit 75; }
  sleep 15; waited=$((waited + 15))
done
echo "$$ $(date '+%F %T') $*" > "$LOCK/owner"
trap 'rm -rf "$LOCK"' EXIT INT TERM

# 2. the machine
while :; do
  load1=$(sysctl -n vm.loadavg | awk '{print $2}')
  busy=$(ps -Ao pid=,%cpu=,command= | awk -v me=$$ '$1!=me && $2>30 && tolower($3) ~ /python/ {n++} END{print n+0}')
  if awk -v l="$load1" -v m="$MAX_LOAD" 'BEGIN{exit !(l<m)}' && [ "$busy" -eq 0 ]; then break; fi
  [ "$waited" -lt "$MAX_WAIT" ] || { echo "benchlock: machine never went idle (load $load1, busy $busy)" >&2; exit 75; }
  echo "benchlock: waiting, load1=$load1 busy_procs=$busy" >&2
  sleep 20; waited=$((waited + 20))
done
echo "benchlock: idle (load1=$load1), running: $*" >&2
# Not `exec`: that would replace this shell and skip the EXIT trap, leaving the
# lock behind for everyone else until the stale timer (found the hard way).
set +e
"$@"
rc=$?
exit "$rc"
