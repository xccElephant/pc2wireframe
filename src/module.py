"""LightningModules for the staged PC2Wireframe training.

The pipeline is trained in **two** independent stages, each with its own config
(``configs/{curve_vae,pc2wireframe}.yaml``) and its own module:

  1. :class:`CurveVAEModule` -- train the per-curve neural parametric VAE
     (``AutoencoderKL1D``) on canonicalised GT curves.
  2. :class:`PC2WireframeModule` -- train the end-to-end point-cloud ->
     wireframe model (PTv3 encoder + latent compressor + transformer wireframe
     decoder) with the **curve VAE frozen** (loaded from stage 1). The decoder
     predicts a node set and an edge set directly; supervision is via Hungarian
     node / edge matching (coordinates, existence with Focal BCE, endpoint
     distributions, and a matched-edge curve L1 through the frozen curve VAE).

Both share the ``AdamW`` + linear-warmup/cosine-decay schedule in
:class:`_BaseModule` and are driven through ``LightningCLI`` (see
``src/main.py``); the model class is selected per-stage via ``class_path``.
"""
from __future__ import annotations

import math
from typing import Any

import pytorch_lightning as pl
import torch

from .metrics import WireframeScore
from .models.criterion import WireframeCriterion
from .models.packing import build_targets, normalized_curves_from_batch
from .models.pc2wireframe import PC2WireframeModel


# ----------------------------------------------------------------------
# default sub-config builders (overridable from YAML)
# ----------------------------------------------------------------------
def _default_pc_encoder() -> dict[str, Any]:
    return dict(
        in_channels=3,
        grid_size=0.01,
        cls_mode=False,
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        stride=(2, 2, 2, 2),
        enable_flash=True,
        # 16 * 256 = 4096 floats (competition latent budget).
        latent_num=16,
        latent_dim=256,
        compressor_heads=8,
        compressor_layers=2,
        variational=True,
    )


def _default_decoder() -> dict[str, Any]:
    return dict(
        num_node_queries=768,    # >= data.yaml max_vertices
        num_edge_queries=1024,   # >= data.yaml max_edges
        d_model=512,
        nhead=8,
        dim_ff=2048,
        node_layers=6,
        edge_layers=4,
        endpoint_dim=128,
        dropout=0.1,
    )


def _default_curve_vae() -> dict[str, Any]:
    return dict(
        latent_channels=3,
        sample_points_num=32,
        # length of down_block_types + last block_out_channels set latent_len
        # (32 / 2**3 = 4) and d_model (256); 3 channels x 4 = 12-d latent.
        down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D"),
        block_out_channels=(128, 256, 256),
        layers_per_block=2,
    )


# ----------------------------------------------------------------------
# checkpoint / freezing helpers
# ----------------------------------------------------------------------
def _load_submodule(
    dest: torch.nn.Module, ckpt_path: str, candidate_prefixes: list[str]
) -> None:
    """Load one submodule's weights out of a (Lightning) checkpoint.

    A previous-stage Lightning checkpoint stores everything under a module path
    (e.g. ``model.curve_vae.*`` or ``curve_vae.*``). We strip the first
    ``candidate_prefix`` that actually matches keys and load the rest into
    ``dest`` (non-strict, so extra/missing keys are reported not fatal).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    best: tuple[str, dict[str, torch.Tensor]] | None = None
    for pref in candidate_prefixes:
        p = (pref + ".") if pref else ""
        sub = {
            (k[len(p):] if p else k): v
            for k, v in sd.items()
            if k.startswith(p)
        }
        if sub and (best is None or len(sub) > len(best[1])):
            best = (pref, sub)
    if best is None:
        raise RuntimeError(
            f"No keys matching prefixes {candidate_prefixes!r} in {ckpt_path!r}"
        )
    missing, unexpected = dest.load_state_dict(best[1], strict=False)
    print(
        f"[load] {ckpt_path} prefix={best[0]!r} -> {dest.__class__.__name__} "
        f"(loaded={len(best[1])}, missing={len(missing)}, "
        f"unexpected={len(unexpected)})"
    )


def _freeze(module: torch.nn.Module) -> None:
    """Put ``module`` in eval mode and disable grads for all its params."""
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def _gt_wireframes(batch: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-sample GT wireframes (numpy) for metric computation."""
    from .data.dataset import unbatch_wireframe_graphs

    graphs = unbatch_wireframe_graphs(batch)
    out: list[dict[str, Any]] = []
    for g in graphs:
        out.append({
            "vertices": g["vertices"].detach().cpu().numpy(),
            "edge_index": g["edge_index"].detach().cpu().numpy(),
            "edge_points": g["edge_points"].detach().cpu().numpy(),
        })
    return out


# ----------------------------------------------------------------------
# shared base: optimizer + frozen-module bookkeeping
# ----------------------------------------------------------------------
class _BaseModule(pl.LightningModule):
    """AdamW + linear-warmup/cosine-decay, optimising only trainable params."""

    def frozen_modules(self) -> list[torch.nn.Module]:
        """Submodules that must stay in eval mode even during ``.train()``."""
        return []

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        # Keep frozen sub-VAEs deterministic (no dropout / running-stat drift).
        for m in self.frozen_modules():
            m.eval()
        return self

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
            betas=tuple(self.hparams.betas),
        )

        warmup = max(1, int(self.hparams.warmup_steps))
        try:
            est = int(self.trainer.estimated_stepping_batches)
        except (RuntimeError, AttributeError):
            est = 0
        total = max(warmup + 1, est if est > 0 else int(self.hparams.max_steps))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ):
        """Clip grads by global norm using the module's ``grad_clip`` hparam.

        This is the single source of truth for clipping; the trainer configs
        therefore leave ``gradient_clip_val`` unset. A ``grad_clip <= 0``
        disables clipping.
        """
        clip = float(getattr(self.hparams, "grad_clip", 0.0) or 0.0)
        if clip > 0.0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip,
                gradient_clip_algorithm="norm",
            )


# ----------------------------------------------------------------------
# stage 1: curve VAE
# ----------------------------------------------------------------------
class CurveVAEModule(_BaseModule):
    """Stage 1 -- train the per-curve VAE (``AutoencoderKL1D``) alone."""

    def __init__(
        self,
        curve_vae: dict[str, Any] | None = None,
        # ----- optimization -----
        lr: float = 1.0e-4,
        weight_decay: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        warmup_steps: int = 500,
        max_steps: int = 100_000,
        grad_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        from .models.vae import AutoencoderKL1D

        self.curve_vae = AutoencoderKL1D(**(curve_vae or _default_curve_vae()))

    def _step(self, batch: dict[str, Any], stage: str):
        curves = normalized_curves_from_batch(batch)  # (Esum, U, 3) or None
        if curves is None or curves.shape[0] == 0:
            return None
        # Validation uses a fixed t grid (and the posterior mode) so val/loss is
        # comparable across epochs; training samples t / the posterior.
        t = None
        if stage != "train":
            p = self.curve_vae.sample_points_num
            t = torch.linspace(0.0, 1.0, p, device=curves.device)
            t = t.unsqueeze(0).expand(curves.shape[0], -1)
        loss, parts = self.curve_vae(
            curves,
            t=t,
            sample_posterior=(stage == "train"),
            return_loss=True,
        )
        bs = int(curves.shape[0])
        self.log(f"{stage}/loss", loss, batch_size=bs, prog_bar=True, sync_dist=True)
        for key, value in parts.items():
            self.log(f"{stage}/{key}", value, batch_size=bs, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")


# ----------------------------------------------------------------------
# stage 2: point cloud -> wireframe (curve VAE frozen)
# ----------------------------------------------------------------------
class PC2WireframeModule(_BaseModule):
    """Stage 2 -- train the end-to-end point-cloud -> wireframe model.

    The PTv3 encoder + latent compressor + wireframe decoder are trained jointly;
    the curve VAE (loaded frozen from stage 1) only turns the predicted per-edge
    curve latent into a polyline. Supervision is set-prediction: Hungarian
    node / edge matching, then matched coordinate / existence / endpoint / curve
    losses (see :class:`~src.models.criterion.WireframeCriterion`).
    """

    def __init__(
        self,
        # ----- model config (nested, fully overridable from YAML) -----
        pc_encoder: dict[str, Any] | None = None,
        decoder: dict[str, Any] | None = None,
        curve_vae: dict[str, Any] | None = None,
        # ----- frozen curve VAE warm-start (stage-1 ckpt) -----
        curve_vae_ckpt: str | None = None,
        # ----- loss weights -----
        kl_weight: float = 1e-4,
        node_coord_weight: float = 5.0,
        node_exist_weight: float = 1.0,
        edge_exist_weight: float = 2.0,
        endpoint_weight: float = 1.0,
        curve_weight: float = 2.0,
        topo_degree_weight: float = 0.5,
        topo_dedup_weight: float = 0.5,
        aux_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        match_node_coord: float = 5.0,
        match_node_exist: float = 1.0,
        match_edge_exist: float = 0.5,
        match_edge_topk: int = 64,
        node_exist_pos_weight: float = 10.0,
        # ----- inference thresholds (also used for val reconstruction) -----
        vertex_threshold: float = 0.5,
        edge_threshold: float = 0.5,
        # ----- eval metric (CCD / TA / VPE -> weighted final score) -----
        eval_w_ccd: float = 0.3,
        eval_w_ta: float = 0.4,
        eval_w_vpe: float = 0.3,
        eval_ccd_tau: float = 0.1,
        eval_vpe_tau: float = 0.1,
        eval_match_thresh: float = 0.1,
        eval_num_per_edge: int = 32,
        # ----- optimization -----
        lr: float = 1.5e-4,
        weight_decay: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        warmup_steps: int = 500,
        max_steps: int = 100_000,
        grad_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = PC2WireframeModel(
            pc_encoder=pc_encoder or _default_pc_encoder(),
            decoder=decoder or _default_decoder(),
            curve_vae=curve_vae or _default_curve_vae(),
        )
        if curve_vae_ckpt:
            _load_submodule(
                self.model.curve_vae, curve_vae_ckpt,
                ["curve_vae", "model.curve_vae"],
            )
        _freeze(self.model.curve_vae)

        self.criterion = WireframeCriterion(
            node_coord_weight=node_coord_weight,
            node_exist_weight=node_exist_weight,
            match_node_coord=match_node_coord,
            match_node_exist=match_node_exist,
            node_exist_pos_weight=node_exist_pos_weight,
            edge_exist_weight=edge_exist_weight,
            endpoint_weight=endpoint_weight,
            curve_weight=curve_weight,
            topo_degree_weight=topo_degree_weight,
            topo_dedup_weight=topo_dedup_weight,
            aux_weight=aux_weight,
            match_edge_exist=match_edge_exist,
            match_edge_topk=match_edge_topk,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            num_curve_points=eval_num_per_edge,
        )

        self.val_metrics = WireframeScore(
            w_ccd=eval_w_ccd, w_ta=eval_w_ta, w_vpe=eval_w_vpe,
            ccd_tau=eval_ccd_tau, vpe_tau=eval_vpe_tau,
            match_thresh=eval_match_thresh, num_per_edge=eval_num_per_edge,
        )

    def frozen_modules(self):
        return [self.model.curve_vae]

    # ------------------------------------------------------------------
    def forward(self, point_cloud: torch.Tensor, sample: bool = False):
        return self.model(point_cloud, sample=sample)

    def _step(self, batch, stage):
        point_cloud = batch["point_cloud"]
        out = self.model(point_cloud, sample=(stage == "train"))
        targets = build_targets(batch)
        total, parts = self.criterion(out["preds"], targets, self.model.curve_vae)

        # KL on the PC-encoder latent (only when variational).
        if out["logvar"] is not None:
            mu, logvar = out["mu"], out["logvar"]
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            total = total + self.hparams.kl_weight * kl
            parts["kl"] = kl.detach()

        bs = point_cloud.shape[0]
        self.log(f"{stage}/loss", total, batch_size=bs, prog_bar=True, sync_dist=True)
        for key, value in parts.items():
            self.log(f"{stage}/{key}", value, batch_size=bs, sync_dist=True)
        return out, total

    def training_step(self, batch, batch_idx):
        _, total = self._step(batch, "train")
        return total

    def validation_step(self, batch, batch_idx):
        out, _ = self._step(batch, "val")
        preds_wf = self.model.reconstruct(
            out["preds"],
            vertex_threshold=self.hparams.vertex_threshold,
            edge_threshold=self.hparams.edge_threshold,
            num_points=self.hparams.eval_num_per_edge,
        )
        self.val_metrics.update(preds_wf, _gt_wireframes(batch))

    def on_validation_epoch_end(self) -> None:
        res = self.val_metrics.compute()
        self.log_dict(
            {f"val/{k}": v for k, v in res.items() if k != "score"},
            prog_bar=False, sync_dist=False,
        )
        self.log("val/score", res["score"], prog_bar=True, sync_dist=False)
        self.val_metrics.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """Per-shape latent + decoded wireframe for submission.

        The model runs in the per-shape *normalized* frame (see the dataset's
        unit-cube transform); predictions are mapped back to raw CAD coordinates
        with the stored ``pc_center`` / ``pc_scale`` so the submission is in the
        original coordinate system.
        """
        out = self.model(batch["point_cloud"], sample=False)
        wireframes = self.model.reconstruct(
            out["preds"],
            vertex_threshold=self.hparams.vertex_threshold,
            edge_threshold=self.hparams.edge_threshold,
            num_points=self.hparams.eval_num_per_edge,
        )
        if "pc_center" in batch:
            center = batch["pc_center"].detach().cpu().numpy()
            scale = batch["pc_scale"].detach().cpu().numpy()
            for s, wf in enumerate(wireframes):
                self._denormalize_wireframe(wf, center[s], float(scale[s]))
        return {
            "shape_id": batch.get("shape_id"),
            "z": out["z"],
            "wireframes": wireframes,
        }

    @staticmethod
    def _denormalize_wireframe(wf: dict[str, Any], center, scale) -> None:
        """Map a reconstructed wireframe from the normalized frame to raw coords.

        Inverse of the dataset unit-cube transform (``x_raw = x_norm * scale +
        center``), applied in place to vertex coordinates and curve polylines.
        """
        if wf.get("vertices") is not None and wf["vertices"].size:
            wf["vertices"] = wf["vertices"] * scale + center
        if wf.get("edge_points") is not None and wf["edge_points"].size:
            wf["edge_points"] = wf["edge_points"] * scale + center


__all__ = ["CurveVAEModule", "PC2WireframeModule"]
