#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"
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
  --batch_size 256 \
  --image_size 224 \
  --val_image_size 224 \
  --val_mask_size 128 \
  --num_slots 20 \
  --train_permutations random \
  --eval_permutations standard \
  --use_conditional_slot_pruning \
  --gate_layers all \
  --gate_warmup 10 \
  --cov_rho 0.8 \
  --cov_tau 0.4 \
  --cov_novelty_alpha 0.3 \
  --which_encoder dinov2_vits14 \
  --num_workers 8 \
  --log_path "$LOG_ROOT/teacher" \
  >"$LOG_FILE" 2>&1 &

echo "Training started: pid=$! log=$LOG_FILE"
