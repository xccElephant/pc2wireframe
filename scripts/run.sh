#!/usr/bin/env bash
# Rectified-Flow PC2Wireframe training driver -- a single trainable model
# (PTv3 encoder + RF velocity net), single config, single stage.
set -euo pipefail

cd "$(dirname "$0")/.."

DATA=configs/data.yaml

# ----- train (single GPU) -----
train() {
  python -m src.main fit --config "$DATA" --config configs/rf.yaml
}

# ----- train (8x A800 DDP) -----
train_ddp() {
  python -m src.main fit --config "$DATA" --config configs/rf_ddp.yaml
}

# ----- wireframe grouper (learned reconstruction; trains on GT point sets) -----
train_grouper() {
  python -m src.main fit \
    --config configs/grouper_data.yaml --config configs/grouper.yaml
}

# ----- wireframe grouper (8x A800 DDP) -----
train_grouper_ddp() {
  python -m src.main fit \
    --config configs/grouper_data.yaml --config configs/grouper_ddp.yaml
}

# ----- inference / submission (pass CKPT=<rf.ckpt>) -----
predict() {
  python -m src.main predict --config "$DATA" --config configs/rf.yaml \
    --ckpt_path "${CKPT:?set CKPT to the trained RF checkpoint}"
}

# Run a target by name, e.g.  bash scripts/run.sh train
"${1:-train}"
