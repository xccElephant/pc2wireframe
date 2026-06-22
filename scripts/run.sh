#!/usr/bin/env bash
# Two-stage PC2Wireframe training driver:
#   stage 1 = corner Rectified Flow (PTv3 encoder + RF velocity net)
#   stage 2 = edge predictor (vertices + latent z -> connectivity + curves)
set -euo pipefail

cd "$(dirname "$0")/.."

DATA=configs/data.yaml

# ----- stage 1: corner RF (single GPU) -----
train() {
  python -m src.main fit --config "$DATA" --config configs/rf.yaml
}

# ----- stage 1: corner RF (8x A800 DDP) -----
train_ddp() {
  python -m src.main fit --config "$DATA" --config configs/rf_ddp.yaml
}

# ----- stage 2: edge predictor (single GPU) -----
train_edge() {
  python -m src.main fit --config configs/edge.yaml
}

# ----- stage 2: edge predictor (8x A800 DDP) -----
train_edge_ddp() {
  python -m src.main fit \
    --config configs/edge.yaml --config configs/edge_ddp.yaml
}

# ----- stage-1 inference (corner cloud + dedup'd vertices; pass CKPT=<rf.ckpt>) -----
predict() {
  python -m src.main predict --config "$DATA" --config configs/rf.yaml \
    --ckpt_path "${CKPT:?set CKPT to the trained RF checkpoint}"
}

# Run a target by name, e.g.  bash scripts/run.sh train
"${1:-train}"
