#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
DATA_PATH="${DATA_PATH:-./data/COCO2017}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./logs/qasa/coco/teacher/251025-013949/checkpoint.pt.tar}"
LOG_ROOT="${LOG_ROOT:-./logs/qasa_eval/coco}"
mkdir -p "$LOG_ROOT/run_logs"
LOG_FILE="$LOG_ROOT/run_logs/eval_$(date +%y%m%d-%H%M%S).log"

nohup python qasa_eval.py \
  --dataset coco \
  --data_path "$DATA_PATH" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --log_path "$LOG_ROOT" \
  --eval_batch_size 32 \
  --which_encoder dino_vitb16 \
  --num_slots 33 \
  --num_iterations 3 \
  --num_dec_blocks 4 \
  --num_heads 6 \
  --train_permutations random \
  --eval_permutations all \
  --livis False \
  --pre_argmax_slot_maxnorm False \
  --eval_gate_slots_attn False \
  >"$LOG_FILE" 2>&1 &

echo "Evaluation started: pid=$! log=$LOG_FILE"
