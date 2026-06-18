"""LightningModule for the Rectified-Flow PC2Wireframe branch.

A single trainable model is driven through ``LightningCLI`` (see
``src/main.py``):

    packed point cloud (coord (P_sum, 3), offset (B,))   (native, variable size)
        -> UtoniaEncoder (frozen PTv3 + trainable latent compressor)
        -> latent z (B, 64, 64)                        (4096-float budget)
    noise x0 ~ N(0, I) (B, N, 3)
        -> RFPointSetVelocity (point-set DiT), conditioned on z
        -> ODE integrate t: 0 -> 1 (torchdiffeq)
        -> wireframe anchor point set x1_hat (B, N, 3) = xyz

Stage-1 stops at the point set; turning it into an explicit wireframe graph is
stage-2's (the grouper's) job.

Training is 1-rectified flow (TorchCFM ``ConditionalFlowMatcher``) with
**per-sample optimal-transport coupling**: within each sample the ``N`` noise
points are matched to the ``N`` target points by entropic OT (``coupling="ot"``)
so every target is paired with a nearby noise sample, giving ``(t, xt, ut)``.
This is essential on a permutation-invariant point set -- random/independent
coupling leaves the per-point velocity target inconsistent and collapses the
model to the data-centroid mean (a blob). All three xyz channels regress the
conditional-flow velocity ``ut`` with an MSE loss (so training and the
velocity-integration sampler use one consistent parameterization). Stage-1
validation logs only the flow-matching ``val/loss`` on the held-out split and
selects the checkpoint by it; ``predict_step`` ODE-samples the point set for
stage-2.
"""
from __future__ import annotations

import math
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .metrics import WireframeScore
from .models.utonia_encoder import UtoniaEncoder
from .models.rf_pointset import RFPointSetVelocity
from .models.wireframe_grouper import WireframeGrouper, grouper_loss
from .recon import group_wireframe


# ----------------------------------------------------------------------
# default sub-config builders (overridable from YAML)
# ----------------------------------------------------------------------
def _default_pc_encoder() -> dict[str, Any]:
    return dict(
        utonia="logs/utonia/utonia.pth",
        grid_size=0.01,
        freeze=True,
        # 64 * 64 = 4096 floats (competition latent budget).
        latent_num=64,
        latent_dim=64,
        compressor_heads=8,
        compressor_layers=1,
    )


def _default_rf_net() -> dict[str, Any]:
    return dict(
        point_dim=3,
        cond_dim=64,
        d_model=384,
        depth=8,
        nhead=6,
        mlp_ratio=4.0,
        dropout=0.0,
        grad_checkpoint=False,
    )


def _default_grouper() -> dict[str, Any]:
    return dict(
        point_dim=3,
        d_model=256,
        depth=6,
        nhead=8,
        mlp_ratio=4.0,
        dropout=0.0,
        embed_dim=8,
    )


# ----------------------------------------------------------------------
# Per-sample optimal-transport coupling for the rectified-flow target
# ----------------------------------------------------------------------
@torch.no_grad()
def ot_couple_noise(
    x0: torch.Tensor, x1: torch.Tensor, *, reg: float, iters: int
) -> torch.Tensor:
    """Reorder the noise ``x0`` to align with the target ``x1``, per sample.

    For each sample in the batch *independently*, solve an entropic optimal
    transport between the ``N`` noise points ``x0`` and the ``N`` target points
    ``x1`` (squared-Euclidean cost over all channels, uniform marginals) with a
    numerically-stable log-domain Sinkhorn, then draw one noise index per target
    row from the transport plan and gather the noise in that order.

    Why: with independent (random) coupling each target point is paired with an
    arbitrary noise point, so on a permutation-invariant point set the velocity
    target ``ut = x1 - x0`` has no consistent per-point assignment and its
    Bayes-optimal regressor is a mean-seeking field that contracts noise to the
    data centroid (the model collapses to a blob). OT coupling pairs every
    target with a *nearby* noise sample, turning ``ut`` into a coherent,
    low-variance target that the network can actually fit. No gradients flow
    through the coupling -- it only selects (x0, x1) pairs.
    """
    b, n, _ = x1.shape
    out = torch.empty_like(x0)
    log_n = math.log(max(n, 1))
    for i in range(b):
        # cost[r, c] between target row r and noise col c
        cost = torch.cdist(x1[i], x0[i]) ** 2            # (N, N)
        eps = float(reg) * cost.mean().clamp_min(1e-8)
        neg_M = -cost / eps                              # (N, N) = log-kernel
        f = x1.new_zeros(n)
        g = x1.new_zeros(n)
        for _ in range(int(iters)):
            f = -log_n - torch.logsumexp(neg_M + g[None, :], dim=1)
            g = -log_n - torch.logsumexp(neg_M + f[:, None], dim=0)
        # log transport plan (row r sums to 1/N); pick one noise col per row.
        log_pi = f[:, None] + neg_M + g[None, :]
        probs = torch.softmax(log_pi, dim=1)             # row-normalized
        idx = torch.multinomial(probs, 1).squeeze(1)     # (N,)
        out[i] = x0[i][idx]
    return out


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
        w_xyz: float = 1.0,
        # ----- noise<->target coupling -----
        # "ot": per-sample entropic-OT coupling (aligns each target point with a
        # nearby noise sample so the velocity target is consistent on the
        # permutation-invariant set). "independent": standard random pairing.
        coupling: str = "ot",
        ot_reg: float = 0.05,
        ot_iters: int = 50,
        # ----- ODE sampling (prediction: point set for stage-2) -----
        ode_steps: int = 50,
        ode_method: str = "euler",
        sample_seed: int = 0,
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

        self.encoder = UtoniaEncoder(**(pc_encoder or _default_pc_encoder()))
        self.net = RFPointSetVelocity(**(rf_net or _default_rf_net()))
        self.point_dim = int(self.net.point_dim)

        from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

        # The flow matcher only turns a (x0, x1) pair into (t, xt, ut); the
        # *coupling* (how x0 is paired with x1) is chosen in ``_flow_loss``.
        # Per-sample OT coupling (``coupling="ot"``) is the important knob here:
        # on a permutation-invariant point set, random/independent coupling
        # leaves the per-point velocity target inconsistent and the model
        # collapses to the data-centroid mean (a Gaussian blob). OT over the N
        # points *within each sample* pairs every target with a nearby noise
        # point, which is exactly what makes the velocity learnable.
        self.flow_matcher = ConditionalFlowMatcher(sigma=float(flow_sigma))

        # Stage-1 validation selects the checkpoint purely by the flow-matching
        # loss on the val split (no ODE sampling / traditional reconstruction /
        # CCD-TA-VPE proxy here -- that scoring belongs to stage-2). The eval_*
        # hparams are kept only for ``predict_step`` reconstruction at export.

    # ------------------------------------------------------------------
    def encode(self, batch: dict[str, Any]) -> torch.Tensor:
        """Latent ``z`` of shape ``(B, K, D)``.

        Consumes the packed point cloud ``batch["point_cloud"] (P_sum, 3)`` +
        ``batch["pc_offset"] (B,)``.
        """
        return self.encoder(batch["point_cloud"], batch["pc_offset"])

    def _flow_loss(
        self, z: torch.Tensor, x1: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Flow-matching loss (shared by train + val).

        The xyz velocity target ``ut`` is regressed so training and the
        velocity-integration sampler share ONE parameterization. The anchor
        point set is pure xyz (3 channels), so the loss is a single MSE over
        all three channels.
        """
        x0 = torch.randn_like(x1)
        if str(self.hparams.coupling) == "ot":
            x0 = ot_couple_noise(
                x0, x1,
                reg=float(self.hparams.ot_reg),
                iters=int(self.hparams.ot_iters),
            )
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)
        v = self.net(t, xt, z)

        loss_xyz = F.mse_loss(v, ut)
        loss = self.hparams.w_xyz * loss_xyz
        return {"loss": loss, "loss_xyz": loss_xyz}

    def training_step(self, batch, batch_idx):
        z = self.encode(batch)
        losses = self._flow_loss(z, batch["wf_points"])
        bs = z.shape[0]
        self.log("train/loss", losses["loss"], batch_size=bs,
                 prog_bar=True, sync_dist=True)
        self.log("train/loss_xyz", losses["loss_xyz"], batch_size=bs, sync_dist=True)
        return losses["loss"]

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
        # Pure-xyz anchor point set: the integrated endpoint already lives in
        # the data frame, so just return it.
        return traj[-1]

    def validation_step(self, batch, batch_idx):
        """Stage-1 validation = flow-matching loss only (drives checkpointing).

        No ODE sampling / reconstruction / CCD-TA-VPE here. ``val/loss`` is the
        epoch-mean velocity-matching loss on the held-out split; the loss is
        stochastic per call (random t / noise) but its mean over the whole val
        set is stable enough to rank epochs.
        """
        z = self.encode(batch)
        losses = self._flow_loss(z, batch["wf_points"])
        bs = z.shape[0]
        self.log("val/loss", losses["loss"], batch_size=bs, prog_bar=True,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss_xyz", losses["loss_xyz"], batch_size=bs,
                 on_step=False, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """Per-shape latent + ODE-sampled wireframe point set.

        Stage-1 stops at the anchor point set ``wf_points (B, N, 3) = xyz``;
        turning it into an explicit wireframe is stage-2's (the grouper's) job.
        The point set is emitted in the per-shape *normalized* frame together
        with ``pc_center`` / ``pc_scale`` so downstream can map xyz back to raw
        CAD coordinates.
        """
        z = self.encode(batch)
        wf_points = self.sample(z)
        return {
            "shape_id": batch.get("shape_id"),
            "z": z,
            "wf_points": wf_points,
            "pc_center": batch.get("pc_center"),
            "pc_scale": batch.get("pc_scale"),
        }


# ----------------------------------------------------------------------
# Wireframe grouper module (learned point-set -> wireframe read-out)
# ----------------------------------------------------------------------
class WireframeGrouperModule(_BaseModule):
    """Learned read-out: a wireframe anchor set ``(N, 3)`` -> explicit wireframe.

    Trains :class:`WireframeGrouper` on the labelled anchor set produced by
    ``WireframeGrouperDataModule`` (per-point edge id, arc-length, endpoints,
    curve type, anchors). All supervision is teacher-forced; decoding at
    validation time uses :func:`src.recon.group_wireframe` and scores against
    the native GT graph with the same CCD / TA / VPE proxy as the RF stage, so
    ``val/score`` is directly comparable to the traditional-reconstruction
    baseline.

    This module is standalone: it consumes wireframe point sets, not raw point
    clouds, so it can be trained on GT point sets without the RF encoder. At
    inference it is fed the RF stage's ODE-sampled point set.
    """

    def __init__(
        self,
        # ----- model -----
        grouper: dict[str, Any] | None = None,
        # ----- loss weights -----
        w_endpoint: float = 1.0,
        w_anchor: float = 1.0,
        w_curve_type: float = 1.0,
        w_arclen: float = 1.0,
        w_embed: float = 1.0,
        w_topo: float = 1.0,
        w_curve_geom: float = 0.1,
        embed_delta_var: float = 0.5,
        embed_delta_dist: float = 1.5,
        # type CE class weights (line / arc / bezier); line tends to dominate.
        curve_type_class_weights: list[float] | None = None,
        topo_tau: float = 0.1,
        geom_num_per_edge: int = 32,
        # ----- decode (validation) -----
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
            gt_wireframes=batch.get("gt_wireframes"),
            w_endpoint=self.hparams.w_endpoint,
            w_anchor=self.hparams.w_anchor,
            w_curve_type=self.hparams.w_curve_type,
            w_arclen=self.hparams.w_arclen,
            w_embed=self.hparams.w_embed,
            w_topo=self.hparams.w_topo,
            w_curve_geom=self.hparams.w_curve_geom,
            curve_type_class_weights=self.hparams.curve_type_class_weights,
            topo_tau=self.hparams.topo_tau,
            geom_num_per_edge=self.hparams.geom_num_per_edge,
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
        eoff = out["endpoint_offset"].detach().cpu().numpy()
        emb = out["embedding"].detach().cpu().numpy()
        ctype = out["curve_type"].detach().cpu().numpy()
        anchor = out["anchor"].detach().cpu().numpy()
        wfs: list[dict[str, Any]] = []
        for i in range(xyz.shape[0]):
            wfs.append(group_wireframe(
                {
                    "xyz": xyz[i],
                    "endpoint_offset": eoff[i],
                    "embedding": emb[i],
                    "curve_type": ctype[i],
                    "anchor": anchor[i],
                },
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
