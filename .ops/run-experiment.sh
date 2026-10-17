#!/bin/bash
set -euo pipefail
ROOT=/home/campus/zhangweihao/FLASH1
cd "$ROOT"
MODE=${1:-basic}
NAME=${2:-${MODE}-$(date -u +%Y%m%dT%H%M%SZ)}
ITERS=${3:-200}
[[ "$NAME" =~ ^[a-zA-Z0-9_-]+$ ]]
if [[ "$MODE" == basic ]]; then LAUNCHER=run_training.py; elif [[ "$MODE" == dp ]]; then LAUNCHER=run_training_dp.py; else exit 2; fi
if [[ -e "results/$NAME/run-config.txt" ]]; then echo "Experiment already exists: $NAME" >&2; exit 2; fi
mkdir -p "results/$NAME/trainlog" "checkpoints/$NAME"
ss -lnt | awk '{print $4}' | grep -qE '(^|:)6379$|(^|:)6000$' && { echo 'Training port busy'; exit 2; }
nvidia-smi --query-gpu=index,uuid,name,memory.total,clocks.current.graphics,clocks.max.graphics --format=csv > "results/$NAME/hardware.csv"
./docker-gh inspect tianyuanwu/greyhound:ae > "results/$NAME/image-inspect.json"
printf 'mode=%s\niters=%s\nsource_commit=d641b3997c17de700e1ef7d37e353418df7e336d\nstart_utc=%s\n' "$MODE" "$ITERS" "$(date -u --iso-8601=seconds)" > "results/$NAME/run-config.txt"
printf 'hidden_size_override=%s\nseq_length_override=%s\n' "${GREYHOUND_HIDDEN_SIZE:-1024}" "${GREYHOUND_SEQ_LENGTH:-32}" >> "results/$NAME/run-config.txt"
set +e
./docker-gh run --name "greyhound-$NAME" --init --privileged --gpus '"device=0,1,2,3"' --runtime nvidia \
  --shm-size 32G --network host \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e PYTHONUNBUFFERED=1 \
  -e GREYHOUND_HIDDEN_SIZE="${GREYHOUND_HIDDEN_SIZE:-1024}" -e GREYHOUND_SEQ_LENGTH="${GREYHOUND_SEQ_LENGTH:-32}" \
  -e GREYHOUND_VERIFY_DP="${GREYHOUND_VERIFY_DP:-0}" \
  -e PIP_NO_INDEX=1 -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
  -v "$ROOT/Greyhound:/workspace/Greyhound" \
  -v "$ROOT/dataset:/workspace/dataset:ro" \
  -v "$ROOT/results:/repro" \
  -v "$ROOT/checkpoints/$NAME:/workspace/checkpoints" \
  -w /workspace/Greyhound --entrypoint bash tianyuanwu/greyhound:ae \
  -c "python -m pip install --no-index --no-deps --force-reinstall detector/dist/control_plane-1.0-py3-none-any.whl && python -u $LAUNCHER --logdir /repro/$NAME/trainlog --iter $ITERS" > "results/$NAME/console.log" 2>&1
RET=$?
printf '%s\n' "$RET" > "results/$NAME/exit-code.txt"
printf 'end_utc=%s\n' "$(date -u --iso-8601=seconds)" >> "results/$NAME/run-config.txt"
exit "$RET"
