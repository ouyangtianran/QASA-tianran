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
  --use_conditional_slot_pruning \
  --gate_layers all \
  --cov_rho 0.8 \
  --cov_tau 0.5 \
  --cov_novelty_alpha 0.3 \
  --pre_argmax_slot_maxnorm False \
  --eval_gate_slots_attn True \
  >"$LOG_FILE" 2>&1 &

echo "Evaluation started: pid=$! log=$LOG_FILE"
