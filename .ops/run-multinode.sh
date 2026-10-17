#!/usr/bin/env bash
set -euo pipefail

ROOT=${FLASH1_ROOT:-/home/campus/zhangweihao/FLASH1}
cd "$ROOT"

NAME=${1:?Usage: run-multinode.sh NAME [ITERS] (set MASTER_ADDR and NODE_RANK per node)}
ITERS=${2:-${GREYHOUND_ITERS:-200}}
MASTER_ADDR=${MASTER_ADDR:-}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-${RANK:-}}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
GPU_DEVICES=${GREYHOUND_GPU_DEVICES:-0,1,2,3}
CHECKPOINT_NAME=${GREYHOUND_CHECKPOINT_NAME:-$NAME}

TP=${GREYHOUND_TP:-1}
PP=${GREYHOUND_PP:-1}
NUM_LAYERS=${GREYHOUND_NUM_LAYERS:-32}
MICRO_BATCH_SIZE=${GREYHOUND_MICRO_BATCH_SIZE:-4}
GLOBAL_BATCH_SIZE=${GREYHOUND_GLOBAL_BATCH_SIZE:-128}
SEQ_LENGTH=${GREYHOUND_SEQ_LENGTH:-32}
HIDDEN_SIZE=${GREYHOUND_HIDDEN_SIZE:-1024}
SAVE_PATH=${GREYHOUND_SAVE:-/workspace/checkpoints}
LOAD_PATH=${GREYHOUND_LOAD:-/workspace/checkpoints}
SAVE_INTERVAL=${GREYHOUND_SAVE_INTERVAL:-10000}
EXIT_INTERVAL=${GREYHOUND_EXIT_INTERVAL:-0}
REDIS_PORT=${GREYHOUND_REDIS_PORT:-6379}
REDIS_TIMEOUT=${GREYHOUND_REDIS_TIMEOUT:-120}
CONTROL_PLANE_WHL_PATH=${CONTROL_PLANE_WHL_PATH:-/workspace/Greyhound/detector/dist/control_plane-1.0-py3-none-any.whl}
PROBE=${GREYHOUND_PROBE:-1}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
NCCL_NET=${NCCL_NET:-Socket}
NCCL_DEBUG=${NCCL_DEBUG:-WARN}
GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}
VERIFY_DP=${GREYHOUND_VERIFY_DP:-0}
DELAY_FILE=${GREYHOUND_DELAY_FILE:-/repro/$NAME/delay-ms}
PP_SPLIT=${GREYHOUND_PP_SPLIT:-}
EXTRA_ARGS=${GREYHOUND_EXTRA_ARGS:-}
HANDLE_RESTART=${GREYHOUND_HANDLE_RESTART:-0}
REPARTITION_METADATA=${GREYHOUND_REPARTITION_METADATA:-}
NO_FAILSLOW=${GREYHOUND_NO_FAILSLOW:-0}
NO_FAILSLOW=${NO_FAILSLOW,,}
case "$NO_FAILSLOW" in
  1|true|yes|on) NO_FAILSLOW=1 ;;
  *) NO_FAILSLOW=0 ;;
esac
FAILSLOW_AWARE=${GREYHOUND_FAILSLOW_AWARE:-1}
FAILSLOW_AWARE=${FAILSLOW_AWARE,,}
if [[ "$FAILSLOW_AWARE" =~ ^(0|false|no|off)$ ]]; then
  NO_FAILSLOW=1
fi

[[ "$NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid experiment name: $NAME" >&2; exit 2; }
[[ "$CHECKPOINT_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || {
  echo "Invalid checkpoint name: $CHECKPOINT_NAME" >&2
  exit 2
}
[[ "$MASTER_ADDR" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
  echo 'MASTER_ADDR must be the master node IPv4 address (for example 10.10.4.1)' >&2
  exit 2
}
[[ "$NNODES" =~ ^[1-9][0-9]*$ ]] || { echo 'NNODES must be a positive integer' >&2; exit 2; }
[[ "$NODE_RANK" =~ ^[0-9]+$ ]] && (( NODE_RANK < NNODES )) || {
  echo "Set NODE_RANK to an integer from 0 through $((NNODES - 1))" >&2
  exit 2
}
[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] && (( NPROC_PER_NODE <= 4 )) || {
  echo 'NPROC_PER_NODE must be between 1 and 4' >&2
  exit 2
}
[[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || { echo 'MASTER_PORT must be a port number' >&2; exit 2; }
(( MASTER_PORT >= 1 && MASTER_PORT <= 65535 )) || { echo 'MASTER_PORT must be between 1 and 65535' >&2; exit 2; }
[[ "$REDIS_PORT" =~ ^[0-9]+$ ]] && (( REDIS_PORT >= 1 && REDIS_PORT <= 65535 )) || {
  echo 'GREYHOUND_REDIS_PORT must be between 1 and 65535' >&2
  exit 2
}
[[ "$ITERS" =~ ^[1-9][0-9]*$ ]] || { echo 'ITERS must be a positive integer' >&2; exit 2; }

RESULT_DIR="results/$NAME"
CHECKPOINT_DIR="checkpoints/$CHECKPOINT_NAME"
if [[ -e "$RESULT_DIR/run-config.txt" ]]; then
  echo "Experiment already exists on this node: $NAME" >&2
  exit 2
fi
mkdir -p "$RESULT_DIR/trainlog" "$CHECKPOINT_DIR"

if [[ "$NODE_RANK" == 0 ]]; then
  ss -lnt | awk 'NR > 1 {print $4}' | grep -qE "(:|\\])($REDIS_PORT|$MASTER_PORT)$" && {
    echo "Training port $REDIS_PORT or $MASTER_PORT is busy on the master node" >&2
    exit 2
  }
fi

nvidia-smi --query-gpu=index,uuid,name,memory.total,clocks.current.graphics,clocks.max.graphics --format=csv > "$RESULT_DIR/hardware.csv"
./docker-gh inspect tianyuanwu/greyhound:ae > "$RESULT_DIR/image-inspect.json"
{
  printf 'mode=multinode\nname=%s\ncheckpoint_name=%s\niters=%s\nmaster_addr=%s\nmaster_port=%s\nnnodes=%s\nnode_rank=%s\nnproc_per_node=%s\n' \
    "$NAME" "$CHECKPOINT_NAME" "$ITERS" "$MASTER_ADDR" "$MASTER_PORT" "$NNODES" "$NODE_RANK" "$NPROC_PER_NODE"
  printf 'tp=%s\npp=%s\nnum_layers=%s\nmicro_batch_size=%s\nglobal_batch_size=%s\nseq_length=%s\nhidden_size=%s\n' \
    "$TP" "$PP" "$NUM_LAYERS" "$MICRO_BATCH_SIZE" "$GLOBAL_BATCH_SIZE" "$SEQ_LENGTH" "$HIDDEN_SIZE"
  printf 'save=%s\nload=%s\nsave_interval=%s\nexit_interval=%s\nno_failslow=%s\nprobe=%s\nverify_dp=%s\nhandle_restart=%s\npp_split=%s\nrepartition_metadata=%s\nextra_args=%s\nnccl_path=%s\nnccl_socket_ifname=%s\nnccl_ib_disable=%s\nnccl_net=%s\nnccl_debug=%s\ngloo_socket_ifname=%s\ndelay_file=%s\nstart_utc=%s\n' \
    "$SAVE_PATH" "$LOAD_PATH" "$SAVE_INTERVAL" "$EXIT_INTERVAL" "$NO_FAILSLOW" "$PROBE" "$VERIFY_DP" "$HANDLE_RESTART" \
    "$PP_SPLIT" "$REPARTITION_METADATA" "$EXTRA_ARGS" "${NCCL_PATH:-}" "$NCCL_SOCKET_IFNAME" "$NCCL_IB_DISABLE" \
    "$NCCL_NET" "$NCCL_DEBUG" "$GLOO_SOCKET_IFNAME" "$DELAY_FILE" "$(date -u --iso-8601=seconds)"
} > "$RESULT_DIR/run-config.txt"

PYTHON_ARGS=(
  --logdir "/repro/$NAME/trainlog"
  --iter "$ITERS"
  --master-addr "$MASTER_ADDR"
  --master-port "$MASTER_PORT"
  --nnodes "$NNODES"
  --node-rank "$NODE_RANK"
  --nproc-per-node "$NPROC_PER_NODE"
  --tp "$TP"
  --pp "$PP"
  --num-layers "$NUM_LAYERS"
  --micro-batch-size "$MICRO_BATCH_SIZE"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --seq-length "$SEQ_LENGTH"
  --hidden-size "$HIDDEN_SIZE"
  --save "$SAVE_PATH"
  --load "$LOAD_PATH"
  --save-interval "$SAVE_INTERVAL"
  --exit-interval "$EXIT_INTERVAL"
  --redis-port "$REDIS_PORT"
  --redis-timeout "$REDIS_TIMEOUT"
)
if [[ -n "$PP_SPLIT" ]]; then
  PYTHON_ARGS+=(--pp-split "$PP_SPLIT")
fi
if [[ "$NO_FAILSLOW" =~ ^(1|true|yes|on)$ ]]; then
  PYTHON_ARGS+=(--no-failslow)
fi

printf -v PYTHON_ARGS_QUOTED '%q ' "${PYTHON_ARGS[@]}"
printf -v CONTROL_WHEEL_QUOTED '%q' "$CONTROL_PLANE_WHL_PATH"
CONTAINER_COMMAND="python -m pip install --no-index --no-deps --force-reinstall $CONTROL_WHEEL_QUOTED && python -u run_training_dp.py $PYTHON_ARGS_QUOTED"

set +e
DOCKER_ARGS=(
  run --name "greyhound-$NAME-node$NODE_RANK" --init --privileged
)
if [[ -n "${NCCL_PATH:-}" ]]; then
  DOCKER_ARGS+=(-e "NCCL_PATH=$NCCL_PATH")
fi
DOCKER_ARGS+=(
  --gpus "\"device=$GPU_DEVICES\"" --runtime nvidia --shm-size 32G --network host
  -e "CUDA_VISIBLE_DEVICES=$GPU_DEVICES" -e PYTHONUNBUFFERED=1
  -e "GREYHOUND_NO_FAILSLOW=$NO_FAILSLOW"
  -e "GREYHOUND_FAILSLOW_AWARE=$FAILSLOW_AWARE"
  -e "GREYHOUND_PROBE=$PROBE"
  -e "GREYHOUND_VERIFY_DP=$VERIFY_DP"
  -e "GREYHOUND_VERIFY_RESTORE=${GREYHOUND_VERIFY_RESTORE:-0}"
  -e "GREYHOUND_HANDLE_RESTART=$HANDLE_RESTART"
  -e "GREYHOUND_REPARTITION_METADATA=$REPARTITION_METADATA"
  -e "GREYHOUND_DELAY_FILE=$DELAY_FILE"
  -e "GREYHOUND_PP_SPLIT=$PP_SPLIT"
  -e "GREYHOUND_EXTRA_ARGS=$EXTRA_ARGS"
  -e "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
  -e "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
  -e "NCCL_NET=$NCCL_NET"
  -e "NCCL_DEBUG=$NCCL_DEBUG"
  -e "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
  -e "REDIS_PORT=$REDIS_PORT"
  -e "CONTROL_PLANE_WHL_PATH=$CONTROL_PLANE_WHL_PATH"
  -e PIP_NO_INDEX=1 -e PIP_DISABLE_PIP_VERSION_CHECK=1
  -v "$ROOT/Greyhound:/workspace/Greyhound"
  -v "$ROOT/dataset:/workspace/dataset:ro"
  -v "$ROOT/results:/repro"
  -v "$ROOT/$CHECKPOINT_DIR:/workspace/checkpoints"
  -w /workspace/Greyhound --entrypoint bash tianyuanwu/greyhound:ae
  -c "$CONTAINER_COMMAND"
)
./docker-gh "${DOCKER_ARGS[@]}" > "$RESULT_DIR/console.log" 2>&1
RET=$?
printf '%s\n' "$RET" > "$RESULT_DIR/exit-code.txt"
printf 'end_utc=%s\n' "$(date -u --iso-8601=seconds)" >> "$RESULT_DIR/run-config.txt"
exit "$RET"
