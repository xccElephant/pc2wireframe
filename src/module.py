"""LightningModule for the Rectified-Flow PC2Wireframe branch.

A single trainable model is driven through ``LightningCLI`` (see
``src/main.py``):

    packed point cloud (coord (P_sum, 3), offset (B,))   (native, variable size)
        -> PCEncoder (PTv3 + latent compressor, deterministic z = mu)
        -> latent z (B, 16, 256)                       (4096-float budget)
    noise x0 ~ N(0, I) (B, N, 4)
        -> RFPointSetVelocity (point-set DiT), conditioned on z
        -> ODE integrate t: 0 -> 1 (torchdiffeq)
        -> wireframe point set x1_hat (B, N, 4) = (xyz, type)
        -> traditional reconstruction -> wireframe {vertices, edge_index,
           edge_points} -> CCD / TA / VPE score

Training is 1-rectified flow: TorchCFM ``ConditionalFlowMatcher(sigma=0)`` gives
``(t, xt, ut)`` and the network regresses the velocity ``ut`` with an MSE loss.
Validation samples deterministically from a fixed noise seed so the metric is
comparable across epochs.
"""
from __future__ import annotations

import math
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .metrics import WireframeScore
from .models.pc_encoder import PCEncoder
from .models.rf_pointset import RFPointSetVelocity
from .models.wireframe_grouper import WireframeGrouper, grouper_loss
from .recon import group_wireframe, reconstruct_wireframe


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
        compressor_layers=1,
    )


def _default_rf_net() -> dict[str, Any]:
    return dict(
        point_dim=4,
        cond_dim=256,
        d_model=384,
        depth=8,
        nhead=6,
        mlp_ratio=4.0,
        dropout=0.0,
        grad_checkpoint=False,
    )


def _default_grouper() -> dict[str, Any]:
    return dict(
        point_dim=4,
        d_model=256,
        depth=6,
        nhead=8,
        mlp_ratio=4.0,
        dropout=0.0,
        embed_dim=8,
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
        """Clip grads by global norm using the module's ``grad_clip`` hparam."""
        clip = float(getattr(self.hparams, "grad_clip", 0.0) or 0.0)
        if clip > 0.0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip,
                gradient_clip_algorithm="norm",
            )


# ----------------------------------------------------------------------
# Rectified-Flow wireframe module
# ----------------------------------------------------------------------
class RFWireframeModule(_BaseModule):
    """Point cloud -> latent -> Rectified-Flow point set -> wireframe."""

    def __init__(
        self,
        # ----- model config (nested, fully overridable from YAML) -----
        pc_encoder: dict[str, Any] | None = None,
        rf_net: dict[str, Any] | None = None,
        # ----- RF target / flow -----
        wf_num_points: int = 8192,
        flow_sigma: float = 0.0,
        # ----- ODE sampling (validation / prediction) -----
        ode_steps: int = 50,
        ode_method: str = "euler",
        sample_seed: int = 0,
        # ----- traditional reconstruction -----
        type_threshold: float = 0.5,
        merge_radius: float = 0.03,
        min_votes: int = 3,
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

        self.encoder = PCEncoder(**(pc_encoder or _default_pc_encoder()))
        self.net = RFPointSetVelocity(**(rf_net or _default_rf_net()))
        self.point_dim = int(self.net.point_dim)

        from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

        self.flow_matcher = ConditionalFlowMatcher(sigma=float(flow_sigma))

        self.val_metrics = WireframeScore(
            w_ccd=eval_w_ccd, w_ta=eval_w_ta, w_vpe=eval_w_vpe,
            ccd_tau=eval_ccd_tau, vpe_tau=eval_vpe_tau,
            match_thresh=eval_match_thresh, num_per_edge=eval_num_per_edge,
        )

    # ------------------------------------------------------------------
    def encode(self, batch: dict[str, Any]) -> torch.Tensor:
        """Latent ``z`` of shape ``(B, K, D)``.

        Consumes the packed point cloud ``batch["point_cloud"] (P_sum, 3)`` +
        ``batch["pc_offset"] (B,)``.
        """
        return self.encoder(batch["point_cloud"], batch["pc_offset"])

    def training_step(self, batch, batch_idx):
        z = self.encode(batch)
        x1 = batch["wf_points"]
        x0 = torch.randn_like(x1)
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)
        v = self.net(t, xt, z)
        loss = F.mse_loss(v, ut)
        self.log("train/loss", loss, batch_size=z.shape[0],
                 prog_bar=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, z: torch.Tensor, num_points: int | None = None) -> torch.Tensor:
        """Deterministic ODE sampling ``x0 -> x1`` conditioned on ``z``.

        Integrates ``dx/dt = net(t, x, z)`` from ``t=0`` to ``t=1`` with a fixed
        noise seed (so the reconstruction is reproducible across epochs).
        Returns ``x1_hat (B, N, point_dim)``.
        """
        from torchdiffeq import odeint

        b = z.shape[0]
        n = int(num_points or self.hparams.wf_num_points)
        gen = torch.Generator(device=z.device)
        gen.manual_seed(int(self.hparams.sample_seed))
        x0 = torch.randn(
            b, n, self.point_dim, device=z.device, dtype=z.dtype, generator=gen)

        def func(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return self.net(t.reshape(()).expand(b), x, z)

        t_span = torch.linspace(
            0.0, 1.0, int(self.hparams.ode_steps) + 1, device=z.device, dtype=z.dtype)
        traj = odeint(func, x0, t_span, method=str(self.hparams.ode_method))
        return traj[-1]

    def _reconstruct_batch(self, x1_hat: torch.Tensor) -> list[dict[str, Any]]:
        pts = x1_hat.detach().cpu().numpy()
        out: list[dict[str, Any]] = []
        for i in range(pts.shape[0]):
            out.append(reconstruct_wireframe(
                pts[i],
                type_threshold=self.hparams.type_threshold,
                merge_radius=self.hparams.merge_radius,
                min_votes=self.hparams.min_votes,
                num_per_edge=self.hparams.eval_num_per_edge,
            ))
        return out

    @staticmethod
    def _gt_to_numpy(gt_wireframes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for g in gt_wireframes:
            out.append({
                "vertices": g["vertices"].detach().cpu().numpy(),
                "edge_index": g["edge_index"].detach().cpu().numpy(),
                "edge_points": g["edge_points"].detach().cpu().numpy(),
            })
        return out

    def validation_step(self, batch, batch_idx):
        z = self.encode(batch)
        x1_hat = self.sample(z)
        preds = self._reconstruct_batch(x1_hat)
        self.val_metrics.update(preds, self._gt_to_numpy(batch["gt_wireframes"]))

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

        Predictions live in the per-shape *normalized* frame; they are mapped
        back to raw CAD coordinates with the stored ``pc_center`` / ``pc_scale``.
        """
        z = self.encode(batch)
        x1_hat = self.sample(z)
        wireframes = self._reconstruct_batch(x1_hat)
        if "pc_center" in batch:
            center = batch["pc_center"].detach().cpu().numpy()
            scale = batch["pc_scale"].detach().cpu().numpy()
            for s, wf in enumerate(wireframes):
                self._denormalize_wireframe(wf, center[s], float(scale[s]))
        return {
            "shape_id": batch.get("shape_id"),
            "z": z,
            "wireframes": wireframes,
        }

    @staticmethod
    def _denormalize_wireframe(wf: dict[str, Any], center, scale) -> None:
        """Inverse of the dataset unit-cube transform (in place)."""
        if wf.get("vertices") is not None and wf["vertices"].size:
            wf["vertices"] = wf["vertices"] * scale + center
        if wf.get("edge_points") is not None and wf["edge_points"].size:
            wf["edge_points"] = wf["edge_points"] * scale + center


# ----------------------------------------------------------------------
# Wireframe grouper module (learned point-set -> wireframe read-out)
# ----------------------------------------------------------------------
class WireframeGrouperModule(_BaseModule):
    """Learned read-out: a wireframe point set ``(N, 4)`` -> explicit wireframe.

    Trains :class:`WireframeGrouper` on the labelled point set produced by
    ``WireframeGrouperDataModule`` (per-point vertex/edge mask, edge id,
    arc-length, endpoints, vertex target). Decoding at validation time uses
    :func:`src.recon.group_wireframe` and scores against the native GT graph
    with the same CCD / TA / VPE proxy as the RF stage, so ``val/score`` is
    directly comparable to the traditional-reconstruction baseline.

    This module is standalone: it consumes wireframe point sets, not raw point
    clouds, so it can be trained on GT point sets without the RF encoder. At
    inference it is fed the RF stage's ODE-sampled point set.
    """

    def __init__(
        self,
        # ----- model -----
        grouper: dict[str, Any] | None = None,
        # ----- loss weights -----
        w_score: float = 1.0,
        w_vertex: float = 1.0,
        w_endpoint: float = 1.0,
        w_arclen: float = 1.0,
        w_embed: float = 1.0,
        embed_delta_var: float = 0.5,
        embed_delta_dist: float = 1.5,
        # ----- decode (validation) -----
        vertex_thresh: float = 0.5,
        vertex_merge_radius: float = 0.01,
        vertex_merge_relative: bool = True,
        split_by_embedding: bool = True,
        embed_eps: float = 0.5,
        min_edge_points: int = 3,
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
        warmup_steps: int = 500,
        max_steps: int = 100_000,
        grad_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.net = WireframeGrouper(**(grouper or _default_grouper()))
        self.val_metrics = WireframeScore(
            w_ccd=eval_w_ccd, w_ta=eval_w_ta, w_vpe=eval_w_vpe,
            ccd_tau=eval_ccd_tau, vpe_tau=eval_vpe_tau,
            match_thresh=eval_match_thresh, num_per_edge=num_per_edge,
        )

    def _loss(self, out, batch) -> dict[str, torch.Tensor]:
        return grouper_loss(
            out, batch,
            w_score=self.hparams.w_score,
            w_vertex=self.hparams.w_vertex,
            w_endpoint=self.hparams.w_endpoint,
            w_arclen=self.hparams.w_arclen,
            w_embed=self.hparams.w_embed,
            embed_kwargs=dict(
                delta_var=self.hparams.embed_delta_var,
                delta_dist=self.hparams.embed_delta_dist,
            ),
        )

    def training_step(self, batch, batch_idx):
        out = self.net(batch["wf_points"])
        losses = self._loss(out, batch)
        bs = batch["wf_points"].shape[0]
        self.log("train/loss", losses["loss"], batch_size=bs,
                 prog_bar=True, sync_dist=True)
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items() if k != "loss"},
            batch_size=bs, sync_dist=True,
        )
        return losses["loss"]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, out: dict[str, torch.Tensor], pts: torch.Tensor
               ) -> list[dict[str, Any]]:
        """Decode a batch of per-point fields into wireframes (numpy)."""
        xyz = pts[..., :3].detach().cpu().numpy()
        vs = out["vertex_score"].detach().cpu().numpy()
        voff = out["vertex_offset"].detach().cpu().numpy()
        eoff = out["endpoint_offset"].detach().cpu().numpy()
        emb = out["embedding"].detach().cpu().numpy()
        arclen = out["arclen"].detach().cpu().numpy()
        wfs: list[dict[str, Any]] = []
        for i in range(xyz.shape[0]):
            wfs.append(group_wireframe(
                {
                    "xyz": xyz[i],
                    "vertex_score": vs[i],
                    "vertex_offset": voff[i],
                    "endpoint_offset": eoff[i],
                    "embedding": emb[i],
                    "arclen": arclen[i],
                },
                vertex_thresh=self.hparams.vertex_thresh,
                vertex_merge_radius=self.hparams.vertex_merge_radius,
                merge_relative=self.hparams.vertex_merge_relative,
                split_by_embedding=self.hparams.split_by_embedding,
                embed_eps=self.hparams.embed_eps,
                min_edge_points=self.hparams.min_edge_points,
                num_per_edge=self.hparams.num_per_edge,
            ))
        return wfs

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

    def validation_step(self, batch, batch_idx):
        out = self.net(batch["wf_points"])
        losses = self._loss(out, batch)
        self.log("val/loss", losses["loss"],
                 batch_size=batch["wf_points"].shape[0], sync_dist=True)
        preds = self.decode(out, batch["wf_points"])
        self.val_metrics.update(preds, self._gt_to_numpy(batch["gt_wireframes"]))

    def on_validation_epoch_end(self) -> None:
        res = self.val_metrics.compute()
        self.log_dict(
            {f"val/{k}": v for k, v in res.items() if k != "score"},
            prog_bar=False, sync_dist=False,
        )
        self.log("val/score", res["score"], prog_bar=True, sync_dist=False)
        self.val_metrics.reset()


__all__ = ["RFWireframeModule", "WireframeGrouperModule"]
