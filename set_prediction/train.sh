#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

: "${QASA_CHECKPOINT_PATH:?Set QASA_CHECKPOINT_PATH to a trained QASA checkpoint}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATA_PATH="${DATA_PATH:-./data/COCO2017}"
LOG_ROOT="${LOG_ROOT:-./logs/set_prediction}"
OUTPUT_PATH="${OUTPUT_PATH:-$LOG_ROOT/probe_best.pth}"
LOG_FILE="$LOG_ROOT/probe_$(date +%y%m%d-%H%M%S).log"
mkdir -p "$LOG_ROOT"

ARGS=(
  --coco_root "$DATA_PATH"
  --qasa_checkpoint "$QASA_CHECKPOINT_PATH"
  --year "${COCO_YEAR:-2017}"
  --image_size "${IMAGE_SIZE:-224}"
  --train_size "${TRAIN_SIZE:-10000}"
  --val_size "${VAL_SIZE:-1000}"
  --batch_size "${BATCH_SIZE:-64}"
  --num_workers "${NUM_WORKERS:-4}"
  --probe_type "${PROBE_TYPE:-linear}"
  --max_steps "${MAX_STEPS:-15000}"
  --out "$OUTPUT_PATH"
)

if [[ -n "${WHICH_ENCODER:-}" ]]; then
  ARGS+=(--which_encoder "$WHICH_ENCODER")
fi
if [[ -n "${QASA_ARGS_JSON:-}" ]]; then
  ARGS+=(--qasa_args_json "$QASA_ARGS_JSON")
fi

nohup python -m set_prediction.set_prediction "${ARGS[@]}" >"$LOG_FILE" 2>&1 &
echo "Set prediction started: pid=$! log=$LOG_FILE"
echo "Best probe: $OUTPUT_PATH"
