"""LightningCLI entry point for the Rectified-Flow PC2Wireframe branch.

A single trainable model (PTv3 encoder + RF velocity net) and datamodule are
resolved from ``class_path`` in the configs::

    # train (single GPU)
    python -m src.main fit \
        --config configs/data.yaml \
        --config configs/rf.yaml

    # train (8x A800 DDP)
    python -m src.main fit \
        --config configs/data.yaml \
        --config configs/rf_ddp.yaml

    # inference / submission
    python -m src.main predict \
        --config configs/data.yaml \
        --config configs/rf.yaml \
        --ckpt_path <rf.ckpt>

``--ckpt_path`` is Lightning's own arg to resume a run / load weights for
``predict`` / ``validate``.
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
