"""LightningCLI entry point for the VQVAE WireframeAE PC2Wireframe branch.

A single trainable end-to-end discrete autoencoder (frozen PTv3 encoder +
multi-scale compressors + per-scale ResidualVQ + graph decoder) and datamodule
are resolved from ``class_path`` in the configs::

    # train (single GPU)
    python -m src.main fit \
        --config configs/data.yaml \
        --config configs/vqvae.yaml

    # train (8x A800 DDP)
    python -m src.main fit \
        --config configs/data.yaml \
        --config configs/vqvae_ddp.yaml

    # validate a checkpoint
    python -m src.main validate \
        --config configs/data.yaml \
        --config configs/vqvae.yaml \
        --ckpt_path <vqvae.ckpt>

Submission export uses ``scripts/export_submission.py`` (single forward:
encode -> RVQ indices -> decode). ``--ckpt_path`` is Lightning's own arg to
resume a run / load weights for ``predict`` / ``validate``.
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
