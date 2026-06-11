#!/usr/bin/env bash
# Staged PC2Wireframe training driver.
#
# Three independent stages, each with its own config. Stage 2 needs the stage-1
# curve-VAE checkpoint (frozen); stage 3 needs the stage-2 wireframe-VAE
# checkpoint and the stage-1 curve-VAE checkpoint (both frozen). Edit the
# CKPT paths below to point at the best checkpoint from the previous stage.
set -euo pipefail

cd "$(dirname "$0")/.."

DATA=configs/data.yaml

# ----- stage 1: curve VAE -----
stage1() {
  python -m src.main fit --config "$DATA" --config configs/curve_vae.yaml
}

# ----- stage 2: wireframe VAE (curve VAE frozen) -----
# Pass CURVE_VAE_CKPT=<path to stage-1 ckpt>.
stage2() {
  python -m src.main fit --config "$DATA" --config configs/wireframe_vae.yaml \
    --model.curve_vae_ckpt "${CURVE_VAE_CKPT:?set CURVE_VAE_CKPT to the stage-1 checkpoint}"
}

# ----- stage 3: point cloud -> wireframe (both VAEs frozen) -----
# Pass WIREFRAME_VAE_CKPT=<stage-2 ckpt> and CURVE_VAE_CKPT=<stage-1 ckpt>.
stage3() {
  python -m src.main fit --config "$DATA" --config configs/pc2wireframe.yaml \
    --model.wireframe_vae_ckpt "${WIREFRAME_VAE_CKPT:?set WIREFRAME_VAE_CKPT to the stage-2 checkpoint}" \
    --model.curve_vae_ckpt "${CURVE_VAE_CKPT:?set CURVE_VAE_CKPT to the stage-1 checkpoint}"
}

# Run a single stage by name, e.g.  bash scripts/run.sh stage1
"${1:-stage1}"
