#!/bin/bash
set -euo pipefail
cd /home/campus/zhangweihao/FLASH1
NAME=${1:-dp-failslow-$(date -u +%Y%m%dT%H%M%SZ)}
ITERS=${2:-400}
export GREYHOUND_HIDDEN_SIZE=${GREYHOUND_HIDDEN_SIZE:-2048}
export GREYHOUND_SEQ_LENGTH=${GREYHOUND_SEQ_LENGTH:-128}
export GREYHOUND_VERIFY_DP=${GREYHOUND_VERIFY_DP:-1}
[[ "$NAME" =~ ^[a-zA-Z0-9_-]+$ ]]
mkdir -p "results/$NAME"
python3 -u inject_clock.py "$NAME" > "results/$NAME/injection-monitor.log" 2>&1 &
GUARD=$!
trap 'kill -TERM "$GUARD" 2>/dev/null || true; wait "$GUARD" 2>/dev/null || true' EXIT
set +e
bash run-experiment.sh dp "$NAME" "$ITERS"
RET=$?
if [[ "$RET" -ne 0 ]]; then kill -TERM "$GUARD" 2>/dev/null || true; fi
wait "$GUARD"
exit "$RET"
