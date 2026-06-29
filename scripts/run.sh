#!/usr/bin/env bash
# Staged PC2Wireframe training driver (PTv3 encoder + edge-query DETR decoder).
#
# Two independent stages, each with its own config. Stage 2 needs the stage-1
# curve-VAE checkpoint (frozen). Pass CURVE_VAE_CKPT=<path to stage-1 ckpt>.
set -euo pipefail

cd "$(dirname "$0")/.."

DATA=configs/data.yaml

# ----- stage 1: curve VAE (single GPU) -----
stage1() {
  python -m src.main fit --config "$DATA" --config configs/curve_vae.yaml
}

# ----- stage 1: curve VAE (8x A800 DDP) -----
stage1_ddp() {
  python -m src.main fit --config "$DATA" --config configs/curve_vae_ddp.yaml
}

# ----- stage 2: point cloud -> wireframe (curve VAE frozen, single GPU) -----
# Pass CURVE_VAE_CKPT=<path to stage-1 ckpt>.
stage2() {
  python -m src.main fit --config "$DATA" --config configs/pc2wireframe.yaml \
    --model.curve_vae_ckpt "${CURVE_VAE_CKPT:?set CURVE_VAE_CKPT to the stage-1 checkpoint}"
}

# ----- stage 2: point cloud -> wireframe (8x A800 DDP) -----
stage2_ddp() {
  python -m src.main fit --config "$DATA" --config configs/pc2wireframe_ddp.yaml \
    --model.curve_vae_ckpt "${CURVE_VAE_CKPT:?set CURVE_VAE_CKPT to the stage-1 checkpoint}"
}

# ----- inference / submission export (pass CKPT=<pc2wireframe.ckpt>) -----
export_submission() {
  python scripts/export_submission.py \
    --ckpt "${CKPT:?set CKPT to the trained stage-2 checkpoint}" \
    --out-dir logs/submission
}

# Run a target by name, e.g.  bash scripts/run.sh stage1
"${1:-stage1}"
