"""LightningModules for the staged PC2Wireframe training.

The pipeline is trained in **two** independent stages, each with its own config
(``configs/{curve_vae,pc2wireframe}.yaml``) and its own module:

  1. :class:`CurveVAEModule` -- train the per-curve neural parametric VAE
     (``AutoencoderKL1D``) on canonicalised GT curves.
  2. :class:`PC2WireframeModule` -- train the end-to-end point-cloud ->
     wireframe model (PTv3 encoder + latent compressor + **edge-set DETR
     decoder**) with the **curve VAE frozen** (loaded from stage 1). The decoder
     emits a set of 512 edge queries; each predicts an existence confidence, two
     endpoints (``tanh`` -> ``[-1,1]``) and a 12-d curve latent. Supervision is a
     Hungarian edge-set matching (endpoint L1 + existence focal/BCE + curve loss
     through the frozen curve VAE; see
     :class:`~src.models.edge_set_criterion.EdgeSetCriterion`). Reconstruction
     thresholds the edges, merges the free endpoints into shared vertices and
     denormalises the decoded curves onto them.

Both share the ``AdamW`` + linear-warmup/cosine-decay schedule in
:class:`_BaseModule` and are driven through ``LightningCLI`` (see
``src/main.py``); the model class is selected per-stage via ``class_path``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from .metrics import WireframeScore
from .models.edge_set_criterion import EdgeSetCriterion
from .models.edge_set_decoder import EdgeSetDecoder
from .models.packing import decode_curve_latent, normalized_curves_from_batch
from .models.pc_encoder import PCEncoder
from .recon import assemble_wireframe


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
        num_edge_queries=512,    # >= data.yaml max_edges
        d_model=512,
        nhead=8,
        dim_ff=2048,
        num_layers=6,
        dropout=0.1,
        deep_supervision=True,
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
    (e.g. ``curve_vae.*`` or ``model.curve_vae.*``). We strip the first
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
    out: list[dict[str, Any]] = []
    for g in batch["gt_wireframes"]:
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
        """Clip grads by global norm using the module's ``grad_clip`` hparam."""
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
        curves = curves.to(self.device)
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

    The PTv3 encoder + latent compressor + edge-set decoder are trained jointly;
    the curve VAE (loaded frozen from stage 1) only turns a predicted per-edge
    curve latent into a polyline. Supervision is the Hungarian edge-set matching
    in :class:`~src.models.edge_set_criterion.EdgeSetCriterion`. Validation
    sweeps the ``(ethr, merge_tol)`` reconstruction grid and logs the best score.
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
        edge_exist_weight: float = 2.0,
        endpoint_weight: float = 5.0,
        curve_weight: float = 5.0,
        curve_endpoint_weight: float = 2.0,
        lat_reg_weight: float = 0.1,
        aux_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        exist_pos_weight_max: float = 40.0,
        # ----- matching costs -----
        match_endpoint: float = 5.0,
        match_exist: float = 1.0,
        match_lat: float = 1.0,
        # ----- decode (validation / predict): assemble the predicted edge set -----
        ethr: float = 0.5,
        merge_tol: float = 0.04,
        min_edges: int = 1,
        num_per_edge: int = 32,
        enms_tol: float = 0.05,
        prune_dangling: bool = False,
        # ----- eval metric (CCD / TA / VPE -> weighted final score) -----
        eval_w_ccd: float = 0.3,
        eval_w_ta: float = 0.4,
        eval_w_vpe: float = 0.3,
        eval_match_thresh: float = 0.1,
        # ----- threshold-robust checkpoint selection ((ethr, merge_tol) grid) -----
        val_ethr_grid: list[float] | None = None,
        val_merge_tol_grid: list[float] | None = None,
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
        from .models.vae import AutoencoderKL1D

        pc_cfg = pc_encoder or _default_pc_encoder()
        self.pc_encoder = PCEncoder(**pc_cfg)
        self.curve_vae = AutoencoderKL1D(**(curve_vae or _default_curve_vae()))
        self.curve_latent_dim = int(
            self.curve_vae.config.latent_channels * self.curve_vae.latent_len)

        dec_cfg = dict(decoder or _default_decoder())
        dec_cfg.setdefault("latent_token_dim", int(pc_cfg["latent_dim"]))
        dec_cfg.setdefault("num_latent_tokens", int(pc_cfg["latent_num"]))
        dec_cfg["curve_latent_dim"] = self.curve_latent_dim
        self.decoder = EdgeSetDecoder(**dec_cfg)
        self.num_per_edge = int(num_per_edge)

        if curve_vae_ckpt:
            _load_submodule(
                self.curve_vae, curve_vae_ckpt,
                ["curve_vae", "model.curve_vae"],
            )
        _freeze(self.curve_vae)

        self.criterion = EdgeSetCriterion(
            edge_exist_weight=edge_exist_weight,
            endpoint_weight=endpoint_weight,
            curve_weight=curve_weight,
            curve_endpoint_weight=curve_endpoint_weight,
            lat_reg_weight=lat_reg_weight,
            aux_weight=aux_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            exist_pos_weight_max=exist_pos_weight_max,
            match_endpoint=match_endpoint,
            match_exist=match_exist,
            match_lat=match_lat,
            num_curve_points=num_per_edge,
        )

        # Threshold-robust validation: one WireframeScore per (ethr, merge_tol).
        et_grid = list(val_ethr_grid) if val_ethr_grid else [ethr]
        mt_grid = list(val_merge_tol_grid) if val_merge_tol_grid else [merge_tol]
        grid = [(float(et), float(mt)) for et in et_grid for mt in mt_grid]
        primary = (float(ethr), float(merge_tol))
        if primary not in grid:
            grid.append(primary)
        self._val_grid = grid
        self._val_primary_key = self._grid_key(*primary)
        self.val_metrics = nn.ModuleDict({
            self._grid_key(et, mt): WireframeScore(
                w_ccd=eval_w_ccd, w_ta=eval_w_ta, w_vpe=eval_w_vpe,
                match_thresh=eval_match_thresh, num_per_edge=num_per_edge,
            )
            for et, mt in grid
        })

    def frozen_modules(self):
        return [self.curve_vae]

    @staticmethod
    def _grid_key(ethr: float, merge_tol: float) -> str:
        """Stable ModuleDict key for an (ethr, merge_tol) grid point."""
        return (f"et{int(round(float(ethr) * 1000)):04d}"
                f"_mt{int(round(float(merge_tol) * 1000)):04d}")

    # ------------------------------------------------------------------
    def forward(
        self, coord: torch.Tensor, offset: torch.Tensor, sample: bool = False
    ) -> dict[str, Any]:
        mu, logvar = self.pc_encoder(coord, offset)
        if sample and logvar is not None:
            z = self.pc_encoder.compressor.reparameterize(mu, logvar)
        else:
            z = mu
        preds = self.decoder(z)
        return {"z": z, "mu": mu, "logvar": logvar, "preds": preds}

    def _step(self, batch, stage):
        out = self.forward(
            batch["point_cloud"], batch["pc_offset"], sample=(stage == "train"))
        total, parts = self.criterion(out["preds"], batch, self.curve_vae)

        # KL on the PC-encoder latent (only when variational).
        if out["logvar"] is not None:
            mu, logvar = out["mu"], out["logvar"]
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            total = total + self.hparams.kl_weight * kl
            parts["kl"] = kl.detach()

        bs = int(batch["num_graphs"])
        self.log(f"{stage}/loss", total, batch_size=bs, prog_bar=True, sync_dist=True)
        for key, value in parts.items():
            self.log(f"{stage}/{key}", value, batch_size=bs, sync_dist=True)
        return out, total

    def training_step(self, batch, batch_idx):
        _, total = self._step(batch, "train")
        return total

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _decode_canonical(self, curve_latent: torch.Tensor) -> np.ndarray:
        """``(B, Ne, D)`` curve latents -> canonical curves ``(B, Ne, P, 3)``."""
        b, ne, _ = curve_latent.shape
        flat = curve_latent.reshape(b * ne, -1)
        curves = decode_curve_latent(
            self.curve_vae, flat, num_points=self.num_per_edge,
            pin_endpoints=True)
        return curves.reshape(b, ne, self.num_per_edge, 3).cpu().numpy()

    @torch.no_grad()
    def _assemble(
        self, preds: dict[str, torch.Tensor], *, ethr: float, merge_tol: float
    ) -> list[dict[str, np.ndarray]]:
        eprob = torch.sigmoid(preds["edge_exist_logit"]).cpu().numpy()
        endpoints = preds["endpoints"].cpu().numpy()
        curves = self._decode_canonical(preds["curve_latent"])
        return [
            assemble_wireframe(
                eprob[i], endpoints[i], curves[i],
                ethr=ethr, merge_tol=merge_tol,
                min_edges=int(self.hparams.min_edges),
                num_per_edge=self.num_per_edge,
                enms_tol=float(self.hparams.enms_tol),
                prune_dangling=bool(self.hparams.prune_dangling),
            )
            for i in range(eprob.shape[0])
        ]

    @torch.no_grad()
    def reconstruct(
        self, preds: dict[str, torch.Tensor]
    ) -> list[dict[str, np.ndarray]]:
        """Assemble each shape's wireframe at the baked-in ``(ethr, merge_tol)``."""
        return self._assemble(
            preds, ethr=float(self.hparams.ethr),
            merge_tol=float(self.hparams.merge_tol))

    # ------------------------------------------------------------------
    @staticmethod
    def _recon_stats(
        preds: list[dict[str, np.ndarray]], gts: list[dict[str, Any]]
    ) -> dict[str, float]:
        n = max(1, len(preds))
        n_nonempty = pred_v = pred_e = gt_v = gt_e = 0.0
        for p, g in zip(preds, gts):
            pe = int(np.asarray(p["edge_index"]).reshape(-1, 2).shape[0])
            pv = int(np.asarray(p["vertices"]).reshape(-1, 3).shape[0])
            ge = int(np.asarray(g["edge_index"]).reshape(-1, 2).shape[0])
            gv = int(np.asarray(g["vertices"]).reshape(-1, 3).shape[0])
            n_nonempty += 1.0 if pe > 0 else 0.0
            pred_e += pe
            pred_v += pv
            gt_e += ge
            gt_v += gv
        return {
            "recon/nonempty_frac": n_nonempty / n,
            "recon/pred_vertices": pred_v / n,
            "recon/pred_edges": pred_e / n,
            "recon/gt_vertices": gt_v / n,
            "recon/gt_edges": gt_e / n,
        }

    def validation_step(self, batch, batch_idx):
        out, _ = self._step(batch, "val")
        bs = int(batch["num_graphs"])
        gts = _gt_wireframes(batch)
        for et, mt in self._val_grid:
            preds = self._assemble(out["preds"], ethr=et, merge_tol=mt)
            key = self._grid_key(et, mt)
            self.val_metrics[key].update(preds, gts)
            if key == self._val_primary_key:
                self.log_dict(
                    self._recon_stats(preds, gts), batch_size=bs,
                    on_step=False, on_epoch=True, sync_dist=True,
                )

    def on_validation_epoch_end(self) -> None:
        best_score = float("-inf")
        best_res: dict[str, torch.Tensor] | None = None
        best_et = best_mt = None
        primary_res: dict[str, torch.Tensor] | None = None

        for et, mt in self._val_grid:
            key = self._grid_key(et, mt)
            res = self.val_metrics[key].compute()
            if key == self._val_primary_key:
                primary_res = res
            self.log(f"val/grid/score@{key}", res["score"], sync_dist=False)
            score = float(res["score"])
            if score > best_score:
                best_score, best_res = score, res
                best_et, best_mt = et, mt
            self.val_metrics[key].reset()

        if best_res is None:
            return
        if primary_res is not None:
            self.log("val/score", primary_res["score"], prog_bar=True,
                     sync_dist=False)
        self.log("val/score_best", best_score, prog_bar=True, sync_dist=False)
        self.log_dict(
            {f"val/{k}": v for k, v in best_res.items() if k != "score"},
            prog_bar=False, sync_dist=False,
        )
        self.log("val/best_ethr", float(best_et), sync_dist=False)
        self.log("val/best_merge_tol", float(best_mt), sync_dist=False)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """Per-shape 16x256 latent + decoded wireframe for submission."""
        out = self.forward(
            batch["point_cloud"], batch["pc_offset"], sample=False)
        wireframes = self.reconstruct(out["preds"])
        return {
            "shape_id": batch.get("shape_id"),
            "z": out["z"],
            "wireframes": wireframes,
        }


__all__ = ["CurveVAEModule", "PC2WireframeModule"]
