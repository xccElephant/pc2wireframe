"""LightningModule for the VQVAE PC2Wireframe branch (edge-centric).

A single trainable, end-to-end discrete autoencoder driven through
``LightningCLI`` (see ``src/main.py``):

    packed point cloud (coord (P_sum, 3), offset (B,))   (native, variable size)
        -> UtoniaEncoder (frozen PTv3, multi-scale per-stage features)
        -> per-scale continuous tokens  z_s (B, N_s, 256)
        -> MultiScaleResidualVQ (per-scale ResidualVQ)
        -> per-scale quantized z_q_s + flat indices (B, T<=4096)   (submission)
        -> EdgeSetDecoder (edge queries -> existence + ordered curve points)
        -> wireframe {vertices, edge_index, edge_points}

The flat **indices** (``T = sum_s N_s * n_q <= 4096``) are the competition
submission; the decoder reconstructs the wireframe from them alone (indices ->
codebooks -> z_q -> decoder). Training is an **edge** set-prediction problem:
the ``Q`` edge queries are Hungarian-matched to the GT edges (endpoint cost),
supervised with an existence focal loss + ordered per-point L1 + endpoint L1 +
small smoothness / segment-length regularizers (see
:class:`~src.models.edge_set_criterion.EdgeSetCriterion`). Vertices are not
predicted directly: they emerge at inference from a union-find merge of the
edge endpoints. A VQ commitment loss (ramped in after a continuous-``z`` warmup)
trains the codebooks; there is no KL term.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch

from .metrics import WireframeScore
from .models.edge_set_criterion import EdgeSetCriterion
from .models.edge_set_decoder import EdgeSetDecoder
from .models.quantizer import MultiScaleResidualVQ
from .models.utonia_encoder import UtoniaEncoder
from .recon import aggregate_wireframe


# ----------------------------------------------------------------------
# default sub-config builders (overridable from YAML)
# ----------------------------------------------------------------------
def _default_pc_encoder() -> dict[str, Any]:
    return dict(
        utonia="logs/utonia/utonia.pth",
        grid_size=0.01,
        freeze=True,
        # multi-scale token allocation (finer -> coarser).
        scale_tokens=[256, 128, 64],
        scale_stages=None,        # None -> deepest len(scale_tokens) stages
        latent_dim=256,
        compressor_heads=8,
        compressor_layers=1,
    )


def _default_quantizer() -> dict[str, Any]:
    # (scale_tokens / dim are taken from the encoder; budget is the *index*
    # count sum_s N_s*n_q, e.g. (256+128+64)*8 = 3584 <= 4096.)
    return dict(
        n_q=8,
        codebook_size=8192,
        kmeans_init=True,
        kmeans_iters=10,
        threshold_ema_dead_code=2,
        decay=0.99,
        commitment_weight=0.25,
    )


def _default_decoder() -> dict[str, Any]:
    return dict(
        latent_dim=256,
        num_edge_queries=512,
        sample_points_num=32,
        d_model=256,
        nhead=8,
        num_layers=6,
        mlp_ratio=4.0,
        dropout=0.0,
        points_hidden=256,
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
# WireframeAE module (edge-centric VQVAE pipeline)
# ----------------------------------------------------------------------
class WireframeAEModule(_BaseModule):
    """Point cloud -> multi-scale RVQ indices -> edge-set decoder -> wireframe."""

    def __init__(
        self,
        # ----- model config (nested, fully overridable from YAML) -----
        pc_encoder: dict[str, Any] | None = None,
        quantizer: dict[str, Any] | None = None,
        decoder: dict[str, Any] | None = None,
        # ----- loss weights (edge-set criterion) -----
        w_exist: float = 1.0,
        w_points: float = 5.0,
        w_endpoint: float = 5.0,
        w_smooth: float = 0.5,
        w_seglen: float = 0.1,
        w_commit: float = 0.25,
        # existence focal-loss params.
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        # Hungarian matching cost = match_w_geo * endpoint_L1 - match_w_exist * p.
        match_w_geo: float = 1.0,
        match_w_exist: float = 0.5,
        # ----- quantization schedule -----
        quant_warmup_steps: int = 2000,   # continuous-z warmup (no commit)
        commit_ramp_steps: int = 2000,    # commit weight 0 -> w_commit ramp
        # ----- decode (validation / predict): endpoint aggregation -----
        edge_thresh: float = 0.5,
        tau_merge: float = 0.015,
        topk_edges: int = 0,
        num_per_edge: int = 32,
        # ----- eval metric -----
        eval_w_ccd: float = 0.3,
        eval_w_ta: float = 0.4,
        eval_w_vpe: float = 0.3,
        eval_ccd_tau: float = 0.1,
        eval_vpe_tau: float = 0.1,
        eval_match_thresh: float = 0.1,
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
        dec_cfg = dict(decoder or _default_decoder())
        dec_cfg.setdefault("num_scales", len(self.encoder.scale_tokens))
        dec_cfg.setdefault("latent_dim", self.encoder.latent_dim)
        self.decoder = EdgeSetDecoder(**dec_cfg)
        self.num_edge_queries = int(self.decoder.num_edge_queries)

        self.criterion = EdgeSetCriterion(
            w_exist=w_exist, w_points=w_points, w_endpoint=w_endpoint,
            w_smooth=w_smooth, w_seglen=w_seglen,
            focal_gamma=focal_gamma, focal_alpha=focal_alpha,
            match_w_geo=match_w_geo, match_w_exist=match_w_exist,
        )

        self.val_metrics = WireframeScore(
            w_ccd=eval_w_ccd, w_ta=eval_w_ta, w_vpe=eval_w_vpe,
            ccd_tau=eval_ccd_tau, vpe_tau=eval_vpe_tau,
            match_thresh=eval_match_thresh, num_per_edge=num_per_edge,
        )

    # ------------------------------------------------------------------
    def _commit_weight(self) -> float:
        """0 during the continuous-z warmup, then linearly ramp to ``w_commit``."""
        step = int(self.global_step)
        warm = int(self.hparams.quant_warmup_steps)
        ramp = max(1, int(self.hparams.commit_ramp_steps))
        if step < warm:
            return 0.0
        return float(self.hparams.w_commit) * min(1.0, (step - warm) / ramp)

    def encode(self, batch: dict[str, Any]) -> list[torch.Tensor]:
        """Per-scale continuous latent tokens ``[z_s (B, N_s, D)]``."""
        return self.encoder(batch["point_cloud"], batch["pc_offset"])

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        z_list = self.encode(batch)
        qo = self.quantizer(z_list)
        # During the warmup the decoder sees continuous z (the codebooks still
        # update via EMA); afterwards (and always at eval) it sees the
        # straight-through z_q, matching the discrete submission.
        use_q = (not self.training) or (
            int(self.global_step) >= int(self.hparams.quant_warmup_steps))
        dec_in = qo["z_q"] if use_q else z_list
        out = self.decoder(dec_in)
        out["indices"] = qo["indices"]
        out["commit"] = qo["commit"]
        out["idx_list"] = qo["idx_list"]
        return out

    # ------------------------------------------------------------------
    def _loss(
        self, out: dict[str, torch.Tensor], gt_wireframes: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        return self.criterion(
            out["edge_exist_logit"], out["edge_points"], gt_wireframes)

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
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items() if k != "loss_geom"},
            batch_size=bs, sync_dist=True,
        )
        self.log_dict(self._perplexity_logs(out), batch_size=bs, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_to_wireframes(
        self, out: dict[str, torch.Tensor]
    ) -> list[dict[str, np.ndarray]]:
        """Endpoint-aggregate each shape's edge set into a wireframe (numpy)."""
        exist_prob = torch.sigmoid(out["edge_exist_logit"])   # (B, Q)
        edge_points = out["edge_points"]                       # (B, Q, P, 3)
        b = exist_prob.shape[0]
        et = float(self.hparams.edge_thresh)
        tau = float(self.hparams.tau_merge)
        topk = int(self.hparams.topk_edges)
        npe = int(self.hparams.num_per_edge)

        wfs: list[dict[str, np.ndarray]] = []
        for i in range(b):
            wfs.append(aggregate_wireframe(
                edge_points[i].detach().cpu().numpy(),
                exist_prob[i].detach().cpu().numpy(),
                edge_threshold=et, tau_merge=tau, topk_edges=topk,
                num_per_edge=npe,
            ))
        return wfs

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
        """Observability: non-empty ratio, aggregated vertex/edge counts, recall."""
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
        preds = self.decode_to_wireframes(out)
        gts = self._gt_to_numpy(batch["gt_wireframes"])
        self.log_dict(
            self._recon_stats(preds, gts), batch_size=bs,
            on_step=False, on_epoch=True, sync_dist=True,
        )
        self.val_metrics.update(preds, gts)

    def on_validation_epoch_end(self) -> None:
        res = self.val_metrics.compute()
        self.log_dict(
            {f"val/{k}": v for k, v in res.items() if k != "score"},
            prog_bar=False, sync_dist=False,
        )
        self.log("val/score", res["score"], prog_bar=True, sync_dist=False)
        self.val_metrics.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """Per-shape flat indices (submission) + decoded wireframe.

        The flat ``indices (B, T<=4096)`` are the competition submission; the
        wireframe is decoded from the quantized latent for visualisation /
        scoring.
        """
        out = self.forward(batch)
        preds = self.decode_to_wireframes(out)
        return {
            "shape_id": batch.get("shape_id"),
            "indices": out["indices"],
            "latent": out["indices"],
            "wireframes": preds,
        }


__all__ = ["WireframeAEModule"]
