#!/usr/bin/env bash
# Single-stage WireframeAE PC2Wireframe training driver -- one trainable
# autoencoder (frozen PTv3 encoder + compressor + WireframeAE decoder).
set -euo pipefail

cd "$(dirname "$0")/.."

DATA=configs/data.yaml

# ----- train (single GPU) -----
train() {
  python -m src.main fit --config "$DATA" --config configs/ae.yaml
}

# ----- train (8x A800 DDP) -----
train_ddp() {
  python -m src.main fit --config "$DATA" --config configs/ae_ddp.yaml
}

# ----- inference / submission export (pass CKPT=<ae.ckpt>) -----
export_submission() {
  python scripts/export_submission.py \
    --ckpt "${CKPT:?set CKPT to the trained WireframeAE checkpoint}" \
    --out-dir logs/submission
}

# Run a target by name, e.g.  bash scripts/run.sh train
"${1:-train}"
