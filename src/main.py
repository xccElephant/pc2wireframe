"""LightningCLI entry point for the staged PC2Wireframe training.

Training is split into two independent stages, each selected by its own config
(the model class is chosen per-stage via ``class_path``)::

    # stage 1: curve VAE
    python -m src.main fit \
        --config configs/data.yaml \
        --config configs/curve_vae.yaml

    # stage 2: point-cloud -> wireframe (curve VAE frozen).
    # The frozen curve VAE is warm-started from the stage-1 checkpoint via the
    # ``curve_vae_ckpt`` model arg -- pass it on the CLI (overrides the YAML):
    python -m src.main fit \
        --config configs/data.yaml \
        --config configs/pc2wireframe.yaml \
        --model.curve_vae_ckpt <stage1.ckpt>

    # inference / submission (stage 2 checkpoint)
    python -m src.main predict \
        --config configs/data.yaml \
        --config configs/pc2wireframe.yaml \
        --ckpt_path <stage2.ckpt>

Both the model and the datamodule are resolved from ``class_path`` in the
configs, so a single entry point serves both stages.

Note on checkpoints -- there are two distinct kinds:
  * ``--model.curve_vae_ckpt`` (or the same key in the YAML): used **once at
    build time** to initialise + freeze the curve VAE from stage 1. This can
    also be hard-coded in the stage-2 config.
  * ``--ckpt_path``: Lightning's own arg to *resume* a run / load weights for
    ``predict``/``validate`` of the current stage. Do not confuse the two.
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
