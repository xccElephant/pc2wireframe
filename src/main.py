"""LightningCLI entry point for the staged PC2Wireframe pipeline.

The model (``CurveVAEModule`` for stage 1 / ``PC2WireframeModule`` for stage 2)
and datamodule are resolved from ``class_path`` in the configs::

    # stage 1: curve VAE
    python -m src.main fit \
        --config configs/data.yaml \
        --config configs/curve_vae.yaml

    # stage 2: point cloud -> wireframe (curve VAE frozen)
    python -m src.main fit \
        --config configs/data.yaml \
        --config configs/pc2wireframe.yaml \
        --model.curve_vae_ckpt <stage1.ckpt>

    # validate a checkpoint
    python -m src.main validate \
        --config configs/data.yaml \
        --config configs/pc2wireframe.yaml \
        --ckpt_path <pc2wireframe.ckpt>

Submission export uses ``scripts/export_submission.py`` (single forward:
encode -> 16x256 latent -> edge-set decode -> merge). ``--ckpt_path`` is
Lightning's own arg to resume a run / load weights for ``predict`` /
``validate``.
"""
import pyrootutils
import torch

torch.set_float32_matmul_precision("high")

pyrootutils.setup_root(
    __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
)

from pytorch_lightning.cli import LightningCLI  # noqa: E402


def main() -> None:
    LightningCLI(
        save_config_callback=None,
        parser_kwargs={"parser_mode": "yaml"},
    )


if __name__ == "__main__":
    main()
