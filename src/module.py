"""LightningModule for the VQVAE PC2Wireframe branch (joint vertex+edge).

A single trainable, end-to-end discrete autoencoder driven through
``LightningCLI`` (see ``src/main.py``):

    packed point cloud (coord (P_sum, 3), offset (B,))   (native, variable size)
        -> UtoniaEncoder (frozen PTv3, multi-scale per-stage features)
        -> per-scale continuous tokens  z_s (B, N_s, 256)
        -> MultiScaleResidualVQ (per-scale ResidualVQ)
        -> per-scale quantized z_q_s + flat indices (B, T<=4096)   (submission)
        -> JointSetDecoder (vertex queries + edge queries + association matrix A)
        -> wireframe {vertices, edge_index, edge_points}

The flat **indices** (``T = sum_s N_s * n_q <= 4096``) are the competition
submission; the decoder reconstructs the wireframe from them alone (indices ->
codebooks -> z_q -> decoder). Training is a **joint** set-prediction problem
(see :class:`~src.models.joint_set_criterion.JointSetCriterion`): vertex queries
are Hungarian-matched to GT vertices (existence + coord), edge queries are
matched to GT edges on an association-aware cost (existence + a per-edge curve
VAE latent), and an explicit edge->vertex association matrix ``A`` carries the
topology. A per-curve VAE is trained **jointly** (no freezing) so each edge's
latent decodes into a precise curve; reconstruction reads endpoints as the
top-2 vertices per edge under ``A`` and denormalises the decoded curve onto
them. A VQ commitment loss (ramped in after a continuous-``z`` warmup) trains the
codebooks; the curve VAE adds a KL term.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from .metrics import WireframeScore
from .models.joint_set_criterion import JointSetCriterion
from .models.joint_set_decoder import JointSetDecoder
from .models.quantizer import MultiScaleResidualVQ
from .models.utonia_encoder import UtoniaEncoder
from .models.vae import AutoencoderKL1D
from .models.vae.curve_packing import decode_curve_latent
from .recon import assemble_wireframe


# ----------------------------------------------------------------------
# default sub-config builders (overridable from YAML)
# ----------------------------------------------------------------------
def _default_pc_encoder() -> dict[str, Any]:
    return dict(
        utonia="logs/utonia/utonia.pth",
        grid_size=0.01,
        freeze=True,
        # all 5 PTv3 encoder stages (finer -> coarser); the finest stages get
        # the most tokens so fine detail survives into the latent.
        scale_tokens=[192, 128, 96, 64, 32],
        scale_stages=[0, 1, 2, 3, 4],
        latent_dim=256,
        # must divide every used stage's enc_channels (54/108/216/432/576): 6.
        compressor_heads=6,
        compressor_layers=1,
    )


def _default_quantizer() -> dict[str, Any]:
    # (scale_tokens / dim are taken from the encoder; budget is the *index*
    # count sum_s N_s*n_q = (192+128+96+64+32)*8 = 512*8 = 4096 <= 4096.)
    # Per-scale codebook sizes shrink towards the coarse end (few tokens there
    # never fill a big codebook -> collapse); raised dead-code revival + faster
    # EMA decay keep the coarse codebooks alive.
    return dict(
        n_q=8,
        codebook_size=[8192, 4096, 2048, 1024, 512],
        kmeans_init=True,
        kmeans_iters=10,
        threshold_ema_dead_code=4,
        decay=0.95,
        commitment_weight=0.25,
    )


def _default_curve_vae() -> dict[str, Any]:
    # 3 channels x 4 latent positions = 12-d per-edge curve latent contract.
    return dict(
        latent_channels=3,
        sample_points_num=32,
        down_block_types=["DownBlock1D", "DownBlock1D", "DownBlock1D"],
        block_out_channels=[128, 256, 256],
        layers_per_block=2,
    )


def _default_joint_decoder() -> dict[str, Any]:
    return dict(
        latent_dim=256,
        num_vertex_queries=512,
        num_edge_queries=512,
        d_model=256,
        nhead=8,
        num_layers=6,
        mlp_ratio=4.0,
        dropout=0.0,
        assoc_dim=64,
        coord_tanh=True,
    )


# ----------------------------------------------------------------------
# shared base: optimizer + schedule
# ----------------------------------------------------------------------
class _BaseModule(pl.LightningModule):
    """AdamW + linear-warmup/cosine-decay, optimising only trainable params."""

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
        clip = float(getattr(self.hparams, "grad_clip", 0.0) or 0.0)
        if clip > 0.0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip,
                gradient_clip_algorithm="norm",
            )


# ----------------------------------------------------------------------
# JointWireframe module (vertex + edge queries + curve VAE, single stage)
# ----------------------------------------------------------------------
class JointWireframeModule(_BaseModule):
    """Point cloud -> RVQ indices -> joint vertex+edge decoder -> wireframe.

    A frozen Utonia + ``MultiScaleResidualVQ`` front end (so the flat-index
    submission contract is unchanged) feeds the :class:`JointSetDecoder`, which
    predicts vertices (existence + coord) and edges (existence + a curve VAE
    latent) with
    an explicit edge->vertex association matrix, and a **trainable** per-curve
    VAE turns the edge latent into a precise curve. Everything (decoder + curve
    VAE) is trained jointly end-to-end; reconstruction reads endpoints from the
    association matrix (top-2 per edge) and denormalises the decoded curve onto
    them. Validation sweeps the ``(vthr, ethr)`` threshold grid.
    """

    def __init__(
        self,
        # ----- model config (nested, fully overridable from YAML) -----
        pc_encoder: dict[str, Any] | None = None,
        quantizer: dict[str, Any] | None = None,
        decoder: dict[str, Any] | None = None,
        curve_vae: dict[str, Any] | None = None,
        # ----- loss weights (joint set criterion) -----
        w_vexist: float = 2.0,
        w_vcoord: float = 5.0,
        w_eexist: float = 2.0,
        w_curve: float = 5.0,
        w_curve_endpoint: float = 2.0,
        w_lat_reg: float = 0.1,
        w_anchor: float = 5.0,
        w_anchor_endpoint: float = 2.0,
        w_assoc: float = 2.0,
        assoc_pos_weight_max: float = 64.0,
        w_commit: float = 0.25,
        # existence loss: focal (focal_gamma>0) else calibrated BCE.
        focal_gamma: float = 0.0,
        focal_alpha: float = 0.5,
        exist_label_smoothing: float = 0.02,
        exist_pos_weight_max: float = 20.0,
        # matching costs
        match_w_vcoord: float = 1.0,
        match_w_exist: float = 0.5,
        match_w_inc: float = 1.0,
        match_w_lat: float = 1.0,
        # ----- schedules -----
        quant_warmup_steps: int = 2000,   # continuous-z warmup (no commit)
        commit_ramp_steps: int = 2000,    # commit weight 0 -> w_commit ramp
        kl_weight: float = 1e-6,          # final curve-VAE KL weight
        kl_ramp_steps: int = 2000,        # KL weight 0 -> kl_weight ramp
        match_warmup_steps: int = 2000,   # steps with w_inc = 0 in edge matching
        match_inc_ramp_steps: int = 2000,  # then w_inc 0 -> 1 ramp
        # ----- decode (validation / predict): threshold the predicted sets -----
        vthr: float = 0.5,
        ethr: float = 0.5,
        min_vertices: int = 2,
        min_edges: int = 1,
        num_per_edge: int = 32,
        # ----- eval metric -----
        eval_w_ccd: float = 0.3,
        eval_w_ta: float = 0.4,
        eval_w_vpe: float = 0.3,
        eval_match_thresh: float = 0.1,
        # ----- threshold-robust checkpoint selection (vthr, ethr grid) -----
        val_vthr_grid: list[float] | None = None,
        val_ethr_grid: list[float] | None = None,
        # ----- optimization -----
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        warmup_steps: int = 1000,
        max_steps: int = 100_000,
        grad_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = UtoniaEncoder(**(pc_encoder or _default_pc_encoder()))
        self.quantizer = MultiScaleResidualVQ(
            scale_tokens=self.encoder.scale_tokens,
            dim=self.encoder.latent_dim,
            **(quantizer or _default_quantizer()),
        )
        self.curve_vae = AutoencoderKL1D(**(curve_vae or _default_curve_vae()))
        self.curve_latent_dim = int(
            self.curve_vae.config.latent_channels * self.curve_vae.latent_len)

        dec_cfg = dict(decoder or _default_joint_decoder())
        dec_cfg.setdefault("num_scales", len(self.encoder.scale_tokens))
        dec_cfg.setdefault("latent_dim", self.encoder.latent_dim)
        dec_cfg["curve_latent_dim"] = self.curve_latent_dim
        self.decoder = JointSetDecoder(**dec_cfg)
        self.num_per_edge = int(num_per_edge)

        self.criterion = JointSetCriterion(
            w_vexist=w_vexist, w_vcoord=w_vcoord,
            w_eexist=w_eexist, w_curve=w_curve,
            w_curve_endpoint=w_curve_endpoint, w_lat_reg=w_lat_reg,
            w_anchor=w_anchor, w_anchor_endpoint=w_anchor_endpoint,
            kl_weight=kl_weight, w_assoc=w_assoc,
            assoc_pos_weight_max=assoc_pos_weight_max,
            focal_gamma=focal_gamma, focal_alpha=focal_alpha,
            exist_label_smoothing=exist_label_smoothing,
            exist_pos_weight_max=exist_pos_weight_max,
            match_w_vcoord=match_w_vcoord, match_w_exist=match_w_exist,
            match_w_inc=match_w_inc, match_w_lat=match_w_lat,
        )

        # Threshold-robust validation: one WireframeScore per (vthr, ethr) point.
        vt_grid = list(val_vthr_grid) if val_vthr_grid else [vthr]
        et_grid = list(val_ethr_grid) if val_ethr_grid else [ethr]
        grid = [(float(vt), float(et)) for vt in vt_grid for et in et_grid]
        primary = (float(vthr), float(ethr))
        if primary not in grid:
            grid.append(primary)
        self._val_grid = grid
        self._val_primary_key = self._grid_key(*primary)
        self.val_metrics = nn.ModuleDict({
            self._grid_key(vt, et): WireframeScore(
                w_ccd=eval_w_ccd, w_ta=eval_w_ta, w_vpe=eval_w_vpe,
                match_thresh=eval_match_thresh, num_per_edge=num_per_edge,
            )
            for vt, et in grid
        })

    # ------------------------------------------------------------------
    @staticmethod
    def _grid_key(vthr: float, ethr: float) -> str:
        """Stable ModuleDict key for a (vthr, ethr) grid point."""
        return (f"vt{int(round(float(vthr) * 1000)):04d}"
                f"_et{int(round(float(ethr) * 1000)):04d}")

    # ------------------------------------------------------------------
    def _commit_weight(self) -> float:
        """0 during the continuous-z warmup, then linearly ramp to ``w_commit``."""
        step = int(self.global_step)
        warm = int(self.hparams.quant_warmup_steps)
        ramp = max(1, int(self.hparams.commit_ramp_steps))
        if step < warm:
            return 0.0
        return float(self.hparams.w_commit) * min(1.0, (step - warm) / ramp)

    def _kl_weight(self) -> float:
        """Curve-VAE KL weight ramped 0 -> ``kl_weight`` over ``kl_ramp_steps``."""
        ramp = max(1, int(self.hparams.kl_ramp_steps))
        return float(self.hparams.kl_weight) * min(
            1.0, int(self.global_step) / ramp)

    def _w_inc(self) -> float:
        """Edge-match incidence weight: 0 during warmup, then ramp 0 -> 1."""
        step = int(self.global_step)
        warm = int(self.hparams.match_warmup_steps)
        ramp = max(1, int(self.hparams.match_inc_ramp_steps))
        if step < warm:
            return 0.0
        return min(1.0, (step - warm) / ramp)

    # ------------------------------------------------------------------
    def encode(self, batch: dict[str, Any]) -> list[torch.Tensor]:
        return self.encoder(batch["point_cloud"], batch["pc_offset"])

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        z_list = self.encode(batch)
        qo = self.quantizer(z_list)
        use_q = (not self.training) or (
            int(self.global_step) >= int(self.hparams.quant_warmup_steps))
        dec_in = qo["z_q"] if use_q else z_list
        out = self.decoder(dec_in)
        out["indices"] = qo["indices"]
        out["commit"] = qo["commit"]
        out["idx_list"] = qo["idx_list"]
        return out

    def _loss(
        self, out: dict[str, torch.Tensor], gt_wireframes: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        return self.criterion(
            out, gt_wireframes, self.curve_vae,
            w_inc=self._w_inc(), kl_weight=self._kl_weight())

    def _perplexity_logs(self, out: dict[str, torch.Tensor]) -> dict[str, Any]:
        pplx = MultiScaleResidualVQ.perplexity(
            out["idx_list"], self.quantizer.codebook_size)
        return {f"vq/perplexity_s{s}": p for s, p in enumerate(pplx)}

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        losses = self._loss(out, batch["gt_wireframes"])
        commit_w = self._commit_weight()
        loss = losses["loss_geom"] + commit_w * out["commit"]
        bs = batch["num_graphs"]
        self.log("train/loss", loss, batch_size=bs, prog_bar=True, sync_dist=True)
        self.log("train/loss_commit", out["commit"].detach(), batch_size=bs,
                 sync_dist=True)
        self.log("train/commit_weight", commit_w, batch_size=bs)
        self.log("train/kl_weight", self._kl_weight(), batch_size=bs)
        self.log("train/w_inc", self._w_inc(), batch_size=bs)
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items() if k != "loss_geom"},
            batch_size=bs, sync_dist=True,
        )
        self.log_dict(self._perplexity_logs(out), batch_size=bs, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _decode_curves(self, curve_latent: torch.Tensor) -> np.ndarray:
        """``(B, Ne, D)`` curve latents -> canonical curves ``(B, Ne, P, 3)``."""
        b, ne, _ = curve_latent.shape
        flat = curve_latent.reshape(b * ne, -1)
        curves = decode_curve_latent(
            self.curve_vae, flat, num_points=self.num_per_edge,
            pin_endpoints=True)                          # (B*Ne, P, 3)
        return curves.reshape(b, ne, self.num_per_edge, 3).cpu().numpy()

    @torch.no_grad()
    def _assemble(
        self, out: dict[str, torch.Tensor], *, vthr: float, ethr: float
    ) -> list[dict[str, np.ndarray]]:
        vprob = torch.sigmoid(out["vertex_exist_logit"]).cpu().numpy()
        vcoord = out["vertex_coord"].cpu().numpy()
        eprob = torch.sigmoid(out["edge_exist_logit"]).cpu().numpy()
        assoc = torch.sigmoid(out["assoc_logit"]).cpu().numpy()
        curves = self._decode_curves(out["curve_latent"])
        return [
            assemble_wireframe(
                vprob[i], vcoord[i], eprob[i], assoc[i], curves[i],
                vthr=vthr, ethr=ethr, num_per_edge=self.num_per_edge,
                min_vertices=int(self.hparams.min_vertices),
                min_edges=int(self.hparams.min_edges),
            )
            for i in range(vprob.shape[0])
        ]

    @torch.no_grad()
    def decode_to_wireframes(
        self, out: dict[str, torch.Tensor]
    ) -> list[dict[str, np.ndarray]]:
        """Assemble each shape's wireframe at the baked-in ``(vthr, ethr)``."""
        return self._assemble(
            out, vthr=float(self.hparams.vthr), ethr=float(self.hparams.ethr))

    # ------------------------------------------------------------------
    @staticmethod
    def _gt_to_numpy(gt_wireframes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "vertices": g["vertices"].detach().cpu().numpy(),
                "edge_index": g["edge_index"].detach().cpu().numpy(),
                "edge_points": g["edge_points"].detach().cpu().numpy(),
            }
            for g in gt_wireframes
        ]

    @staticmethod
    def _recon_stats(
        preds: list[dict[str, np.ndarray]],
        gts: list[dict[str, Any]],
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
        out = self.forward(batch)
        losses = self._loss(out, batch["gt_wireframes"])
        loss = losses["loss_geom"] + float(self.hparams.w_commit) * out["commit"]
        bs = batch["num_graphs"]
        self.log("val/loss", loss, batch_size=bs, prog_bar=True,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss_commit", out["commit"].detach(), batch_size=bs,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(
            {f"val/{k}": v for k, v in losses.items() if k != "loss_geom"},
            batch_size=bs, on_step=False, on_epoch=True, sync_dist=True,
        )
        self.log_dict(self._perplexity_logs(out), batch_size=bs,
                      on_step=False, on_epoch=True, sync_dist=True)

        gts = self._gt_to_numpy(batch["gt_wireframes"])
        for vt, et in self._val_grid:
            preds = self._assemble(out, vthr=vt, ethr=et)
            key = self._grid_key(vt, et)
            self.val_metrics[key].update(preds, gts)
            if key == self._val_primary_key:
                self.log_dict(
                    self._recon_stats(preds, gts), batch_size=bs,
                    on_step=False, on_epoch=True, sync_dist=True,
                )

    def on_validation_epoch_end(self) -> None:
        best_score = float("-inf")
        best_res: dict[str, torch.Tensor] | None = None
        best_vt = best_et = None
        primary_res: dict[str, torch.Tensor] | None = None

        for vt, et in self._val_grid:
            key = self._grid_key(vt, et)
            res = self.val_metrics[key].compute()
            if key == self._val_primary_key:
                primary_res = res
            self.log(f"val/grid/score@{key}", res["score"], sync_dist=False)
            score = float(res["score"])
            if score > best_score:
                best_score, best_res = score, res
                best_vt, best_et = vt, et
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
        self.log("val/best_vthr", float(best_vt), sync_dist=False)
        self.log("val/best_ethr", float(best_et), sync_dist=False)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        out = self.forward(batch)
        preds = self.decode_to_wireframes(out)
        return {
            "shape_id": batch.get("shape_id"),
            "indices": out["indices"],
            "latent": out["indices"],
            "wireframes": preds,
        }


__all__ = ["JointWireframeModule"]
