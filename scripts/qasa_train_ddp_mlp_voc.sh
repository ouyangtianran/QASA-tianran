#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29503}"
DATA_PATH="${DATA_PATH:-./data/VOCdevkit/VOC2012}"
LOG_ROOT="${LOG_ROOT:-./logs/qasa/voc}"
mkdir -p "$LOG_ROOT/run_logs"
LOG_FILE="$LOG_ROOT/run_logs/train_$(date +%y%m%d-%H%M%S).log"

nohup torchrun \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  qasa_train_ddp.py \
  --dataset voc \
  --data_path "$DATA_PATH" \
  --epochs 600 \
  --batch_size 32 \
  --image_size 224 \
  --val_image_size 224 \
  --val_mask_size 128 \
  --num_slots 20 \
  --train_permutations random \
  --eval_permutations standard \
  --dec_type mlp \
  --use_conditional_slot_pruning \
  --gate_eps 1e-6 \
  --gate_warmup 10 \
  --cov_rho 0.8 \
  --cov_tau 0.5 \
  --cov_novelty_alpha 0.3 \
  --num_workers 8 \
  --log_path "$LOG_ROOT/teacher" \
  >"$LOG_FILE" 2>&1 &

echo "Training started: pid=$! log=$LOG_FILE"
