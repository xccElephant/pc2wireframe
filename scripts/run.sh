#!/usr/bin/env bash
# VQVAE WireframeAE PC2Wireframe training driver -- one trainable end-to-end
# discrete autoencoder (frozen PTv3 encoder + multi-scale compressors +
# per-scale ResidualVQ + graph decoder).
set -euo pipefail

cd "$(dirname "$0")/.."

DATA=configs/data.yaml

# ----- train (single GPU) -----
train() {
  python -m src.main fit --config "$DATA" --config configs/vqvae.yaml
}

# ----- train (8x A800 DDP) -----
train_ddp() {
  python -m src.main fit --config "$DATA" --config configs/vqvae_ddp.yaml
}

# ----- inference / submission export (pass CKPT=<vqvae.ckpt>) -----
export_submission() {
  python scripts/export_submission.py \
    --ckpt "${CKPT:?set CKPT to the trained VQVAE checkpoint}" \
    --out-dir logs/submission
}

# Run a target by name, e.g.  bash scripts/run.sh train
"${1:-train}"
