#!/usr/bin/env bash
# Staged PC2Wireframe training driver.
#
# Two independent stages, each with its own config. Stage 2 needs the stage-1
# curve-VAE checkpoint (frozen). Pass CURVE_VAE_CKPT=<path to stage-1 ckpt>.
set -euo pipefail

cd "$(dirname "$0")/.."

DATA=configs/data.yaml

# ----- stage 1: curve VAE -----
stage1() {
  python -m src.main fit --config "$DATA" --config configs/curve_vae.yaml
}

# ----- stage 2: point cloud -> wireframe (curve VAE frozen) -----
# Pass CURVE_VAE_CKPT=<path to stage-1 ckpt>.
stage2() {
  python -m src.main fit --config "$DATA" --config configs/pc2wireframe.yaml \
    --model.curve_vae_ckpt "${CURVE_VAE_CKPT:?set CURVE_VAE_CKPT to the stage-1 checkpoint}"
}

# Run a single stage by name, e.g.  bash scripts/run.sh stage1
"${1:-stage1}"
