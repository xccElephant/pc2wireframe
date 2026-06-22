"""LightningModules for the Rectified-Flow PC2Wireframe branch (two stages).

Both stages are trained independently and driven through ``LightningCLI`` (see
``src/main.py``).

**Stage 1 -- corner Rectified Flow** (:class:`RFWireframeModule`)::

    packed point cloud (coord (P_sum, 3), offset (B,))   (native, variable size)
        -> UtoniaEncoder (frozen PTv3 + trainable latent compressor)
        -> latent z (B, 64, 64)                        (4096-float budget)
    noise x0 ~ N(0, I) (B, N, 3)
        -> RFPointSetVelocity (point-set DiT), conditioned on z
        -> ODE integrate t: 0 -> 1 (torchdiffeq)
        -> corner point set x1_hat (B, N, 3) -> DBSCAN dedup -> vertices V

Training is 1-rectified flow (TorchCFM ``ConditionalFlowMatcher``) with
**per-sample optimal-transport coupling**: within each sample the ``N`` noise
points are matched to the ``N`` corner-target points by entropic OT
(``coupling="ot"``) so every target is paired with a nearby noise sample,
giving ``(t, xt, ut)``. This is essential on a permutation-invariant point set
-- random/independent coupling leaves the per-point velocity target
inconsistent and collapses the model to the data-centroid mean (a blob). All
three xyz channels regress the conditional-flow velocity ``ut`` with an MSE
loss. Validation logs the flow-matching ``val/loss`` (the checkpoint monitor)
and, optionally, a ``val/vpe`` (vertex chamfer between the dedup'd ODE sample
and the GT corners) so checkpoints can also be ranked by corner quality.

**Stage 2 -- edge predictor** (:class:`EdgePredictorModule`)::

    point cloud -> UtoniaEncoder (own compressor) -> latent z
    GT vertices (+ train-time augmentation) -> EdgePredictor(z)
        -> pairwise edge logits (V x V) + per-edge U x 3 curve points
        -> threshold + assemble -> wireframe -> CCD / TA / VPE

The two encoders are fully decoupled (only the PTv3 backbone is shared and
frozen); stage 2 trains its own compressor from scratch.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .metrics import WireframeScore
from .models.utonia_encoder import UtoniaEncoder
from .models.rf_pointset import RFPointSetVelocity
from .models.edge_predictor import (
    EdgePredictor,
    assemble_wireframe,
    dedup_vertices,
    edges_from_logits,
)


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


def _default_edge_predictor() -> dict[str, Any]:
    return dict(
        point_dim=3,
        z_dim=64,
        d_model=256,
        depth=6,
        nhead=8,
        mlp_ratio=4.0,
        dropout=0.0,
        pair_dim=64,
        num_edge_points=32,
        use_edge_evidence=True,
        edge_evidence_points=4,
        evidence_dim=64,
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
        wf_num_points: int = 1024,
        flow_sigma: float = 0.0,
        w_xyz: float = 1.0,
        # ----- noise<->target coupling -----
        # "ot": per-sample entropic-OT coupling (aligns each target point with a
        # nearby noise sample so the velocity target is consistent on the
        # permutation-invariant set). "independent": standard random pairing.
        coupling: str = "ot",
        ot_reg: float = 0.05,
        ot_iters: int = 50,
        # ----- ODE sampling (prediction: corner set for stage-2) -----
        ode_steps: int = 50,
        ode_method: str = "euler",
        sample_seed: int = 0,
        # ----- corner dedup (sampled point cloud -> vertices) -----
        dedup_eps: float = 0.02,
        dedup_relative: bool = True,
        dedup_min_samples: int = 1,
        # When True, validation also ODE-samples, dedups, and logs a vertex
        # chamfer ``val/vpe`` against the GT corners (slower, but lets the
        # stage-1 checkpoint be ranked by corner quality, not just flow loss).
        val_vpe: bool = False,
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
        # loss on the val split; the CCD-TA-VPE proxy belongs to stage 2. When
        # ``val_vpe`` is set the step additionally logs a dedup'd-corner vertex
        # chamfer so checkpoints can be ranked by corner quality too.

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

    # ------------------------------------------------------------------
    @torch.no_grad()
    def dedup(self, points: torch.Tensor) -> np.ndarray:
        """DBSCAN-dedup a sampled corner cloud ``(N, 3)`` into vertices."""
        return dedup_vertices(
            points.detach().cpu().numpy(),
            eps=float(self.hparams.dedup_eps),
            relative=bool(self.hparams.dedup_relative),
            min_samples=int(self.hparams.dedup_min_samples),
        )

    @staticmethod
    def _vertex_chamfer(pred: np.ndarray, gt: np.ndarray) -> float:
        """Symmetric vertex chamfer (mean of means); ``inf`` if either empty."""
        if pred.shape[0] == 0 or gt.shape[0] == 0:
            return float("inf")
        p = torch.from_numpy(pred.astype(np.float32))
        g = torch.from_numpy(gt.astype(np.float32))
        d = torch.cdist(p, g)                     # (P, G)
        return 0.5 * (float(d.min(dim=1)[0].mean()) + float(d.min(dim=0)[0].mean()))

    def validation_step(self, batch, batch_idx):
        """Stage-1 validation: flow-matching loss (+ optional corner ``val/vpe``).

        ``val/loss`` is the epoch-mean velocity-matching loss on the held-out
        split (the checkpoint monitor); it is stochastic per call (random t /
        noise) but its mean over the whole val set is stable enough to rank
        epochs. When ``val_vpe`` is set, the step also ODE-samples, dedups the
        corner cloud into vertices, and logs the vertex chamfer against the GT
        corners so checkpoints can be ranked by corner quality.
        """
        z = self.encode(batch)
        losses = self._flow_loss(z, batch["wf_points"])
        bs = z.shape[0]
        self.log("val/loss", losses["loss"], batch_size=bs, prog_bar=True,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss_xyz", losses["loss_xyz"], batch_size=bs,
                 on_step=False, on_epoch=True, sync_dist=True)

        if self.hparams.val_vpe and "gt_wireframes" in batch:
            sampled = self.sample(z)              # (B, N, 3)
            vpes: list[float] = []
            for i, gt in enumerate(batch["gt_wireframes"]):
                pred_v = self.dedup(sampled[i])
                gt_v = gt["vertices"].detach().cpu().numpy()
                vpe = self._vertex_chamfer(pred_v, gt_v)
                if math.isfinite(vpe):
                    vpes.append(vpe)
            if vpes:
                self.log("val/vpe", float(np.mean(vpes)), batch_size=bs,
                         on_step=False, on_epoch=True, prog_bar=True,
                         sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """Per-shape latent + ODE-sampled corner cloud + dedup'd vertices.

        Stage 1 emits the corner point set in the per-shape *normalized* frame
        together with the DBSCAN-dedup'd ``vertices`` and ``pc_center`` /
        ``pc_scale`` so downstream (stage 2 / submission) can map xyz back to
        raw CAD coordinates.
        """
        z = self.encode(batch)
        sampled = self.sample(z)
        vertices = [self.dedup(sampled[i]) for i in range(sampled.shape[0])]
        return {
            "shape_id": batch.get("shape_id"),
            "z": z,
            "wf_points": sampled,
            "vertices": vertices,
            "pc_center": batch.get("pc_center"),
            "pc_scale": batch.get("pc_scale"),
        }


# ----------------------------------------------------------------------
# Edge predictor module (vertices + latent -> wireframe connectivity)
# ----------------------------------------------------------------------
class EdgePredictorModule(_BaseModule):
    """Stage-2: GT corners + latent ``z`` -> pairwise edges + per-edge curves.

    Trains :class:`EdgePredictor` on padded GT vertex sets conditioned on the
    shape latent ``z`` (its **own** frozen-PTv3 + trainable-compressor encoder,
    decoupled from stage 1). The connectivity target is the GT adjacency matrix;
    edges are supervised with a class-balanced (``pos_weight`` / focal) BCE and
    the curves on GT positive edges with an order-invariant smooth-L1.

    To bridge the gap to the noisy stage-1 dedup output (jittered corners +
    spurious / dropped vertices), the training vertices are augmented with xyz
    jitter, a few **fake** vertices (all-zero adjacency rows the model must
    learn to leave unconnected), and optional vertex dropout.

    Validation feeds the **clean GT corners** (no augmentation), thresholds the
    edge logits, assembles the wireframe and scores it with the same CCD / TA /
    VPE proxy as stage 1, so ``val/score`` is directly comparable.
    """

    def __init__(
        self,
        # ----- model -----
        edge_predictor: dict[str, Any] | None = None,
        pc_encoder: dict[str, Any] | None = None,
        # ----- padded vertex set -----
        vmax: int = 512,
        # ----- loss -----
        w_edge: float = 1.0,
        w_curve: float = 1.0,
        # edge BCE: "focal" or "bce" (with pos_weight). Negatives dominate the
        # V^2 pairs, so either focal loss or a >1 pos_weight is needed.
        edge_loss: str = "focal",
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        edge_pos_weight: float = 0.0,
        # ----- train-time augmentation (bridge to noisy stage-1 corners) -----
        jitter_std: float = 0.01,
        fake_vertex_frac: float = 0.1,
        fake_vertex_jitter: float = 0.05,
        vertex_dropout: float = 0.0,
        # ----- decode (validation / predict) -----
        # Single fallback threshold + the candidate set swept on validation. The
        # focal / pos_weight loss shifts the operating point away from 0.5, so
        # the threshold that maximises val/score is selected each val epoch and
        # persisted (``best_edge_threshold``) for inference instead of a fixed 0.5.
        edge_threshold: float = 0.5,
        edge_thresholds: list[float] | None = None,
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

        self.encoder = UtoniaEncoder(**(pc_encoder or _default_pc_encoder()))
        self.net = EdgePredictor(
            **(edge_predictor or _default_edge_predictor()))
        self.vmax = int(vmax)

        # Threshold sweep: one metric accumulator per candidate threshold. The
        # epoch end picks the threshold with the best score and persists it.
        thrs = edge_thresholds or [0.3, 0.4, 0.5, 0.6, 0.7]
        self.edge_thresholds = sorted(float(t) for t in thrs)
        self.val_metrics = torch.nn.ModuleDict(
            {
                self._thr_key(t): WireframeScore(
                    w_ccd=eval_w_ccd, w_ta=eval_w_ta, w_vpe=eval_w_vpe,
                    ccd_tau=eval_ccd_tau, vpe_tau=eval_vpe_tau,
                    match_thresh=eval_match_thresh, num_per_edge=num_per_edge,
                )
                for t in self.edge_thresholds
            }
        )
        # Persisted across checkpoints; used as the default decode threshold.
        self.register_buffer(
            "best_edge_threshold", torch.tensor(float(edge_threshold)))

    @staticmethod
    def _thr_key(t: float) -> str:
        return f"t{int(round(float(t) * 1000)):04d}"

    # ------------------------------------------------------------------
    def encode(self, batch: dict[str, Any]) -> torch.Tensor:
        """Latent ``z (B, K, D)`` from the packed point cloud."""
        return self.encoder(batch["point_cloud"], batch["pc_offset"])

    def _prepare_sample(
        self, gt: dict[str, torch.Tensor], training: bool, device: torch.device
    ) -> dict[str, torch.Tensor] | None:
        """Build the padded vertices + adjacency target for one shape.

        Applies the training-time augmentation (vertex dropout -> jitter -> fake
        vertices) and returns ``None`` when the shape has too few vertices to
        form an edge. The returned dict has ``verts (Vmax,3)``, ``mask (Vmax,)``,
        ``adj (Vmax,Vmax)``, ``pos_edges (E,2)`` and ``pos_points (E,U,3)``.
        """
        verts = gt["vertices"].to(device=device, dtype=torch.float32)
        edges = gt["edge_index"].to(device=device, dtype=torch.long)
        epts = gt["edge_points"].to(device=device, dtype=torch.float32)
        v = verts.shape[0]
        if v < 2 or edges.shape[0] == 0:
            return None

        if training and self.hparams.vertex_dropout > 0.0 and v > 2:
            keep = torch.rand(v, device=device) >= self.hparams.vertex_dropout
            if int(keep.sum()) >= 2:
                old2new = torch.full((v,), -1, device=device, dtype=torch.long)
                old2new[keep] = torch.arange(int(keep.sum()), device=device)
                verts = verts[keep]
                emask = keep[edges[:, 0]] & keep[edges[:, 1]]
                edges = old2new[edges[emask]]
                epts = epts[emask]
                v = verts.shape[0]
                if v < 2 or edges.shape[0] == 0:
                    return None

        if training and self.hparams.jitter_std > 0.0:
            verts = verts + torch.randn_like(verts) * self.hparams.jitter_std

        n_fake = 0
        if training and self.hparams.fake_vertex_frac > 0.0:
            n_fake = int(round(v * float(self.hparams.fake_vertex_frac)))
            n_fake = max(0, min(n_fake, self.vmax - v))
        if v + n_fake > self.vmax:
            # Too many real vertices to pad; truncating would corrupt the
            # adjacency, so drop the shape (the datamodule's max_vertices cap
            # should normally prevent this).
            return None

        if n_fake > 0:
            src = verts[torch.randint(0, v, (n_fake,), device=device)]
            fake = src + torch.randn_like(src) * self.hparams.fake_vertex_jitter
            all_verts = torch.cat([verts, fake], dim=0)
        else:
            all_verts = verts
        vtot = all_verts.shape[0]

        padded = verts.new_zeros(self.vmax, 3)
        padded[:vtot] = all_verts
        mask = torch.zeros(self.vmax, dtype=torch.bool, device=device)
        mask[:vtot] = True

        adj = verts.new_zeros(self.vmax, self.vmax)
        i, j = edges[:, 0], edges[:, 1]
        adj[i, j] = 1.0
        adj[j, i] = 1.0
        return {
            "verts": padded,
            "mask": mask,
            "adj": adj,
            "pos_edges": edges,
            "pos_points": epts,
        }

    def _edge_bce(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Class-balanced BCE over a flat set of candidate-pair logits."""
        if logits.numel() == 0:
            return logits.new_zeros(())
        if str(self.hparams.edge_loss) == "focal":
            p = torch.sigmoid(logits)
            ce = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none")
            p_t = p * target + (1.0 - p) * (1.0 - target)
            alpha = self.hparams.focal_alpha
            a_t = alpha * target + (1.0 - alpha) * (1.0 - target)
            loss = a_t * (1.0 - p_t).clamp_min(1e-6) ** self.hparams.focal_gamma * ce
            return loss.mean()
        pw = float(self.hparams.edge_pos_weight)
        weight = logits.new_tensor(pw) if pw > 0.0 else None
        return F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=weight)

    @staticmethod
    def _curve_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Order-invariant smooth-L1 between predicted and GT curve points."""
        if pred.shape[0] == 0:
            return pred.new_zeros(())
        if pred.shape[1] != gt.shape[1]:
            raise ValueError(
                f"curve point count mismatch: predictor U={pred.shape[1]} vs "
                f"data num_edge_points={gt.shape[1]} (keep them equal)")
        fwd = F.smooth_l1_loss(pred, gt, reduction="none").mean(dim=(1, 2))
        rev = F.smooth_l1_loss(
            pred, torch.flip(gt, dims=[1]), reduction="none").mean(dim=(1, 2))
        return torch.minimum(fwd, rev).mean()

    def _zero_loss(self, z: torch.Tensor) -> torch.Tensor:
        """A 0-valued loss that still touches every net parameter (DDP-safe)."""
        device = z.device
        verts = z.new_zeros(1, 2, 3)
        mask = torch.ones(1, 2, dtype=torch.bool, device=device)
        zb = z[:1]
        out = self.net(verts, mask, zb)
        bi = torch.zeros(1, dtype=torch.long, device=device)
        ii = torch.zeros(1, dtype=torch.long, device=device)
        jj = torch.ones(1, dtype=torch.long, device=device)
        curves = self.net.curves_for_pairs(
            out["h"], verts, out["z_global"], bi, ii, jj)
        touch = out["edge_logits"].sum() + curves.sum() + z.sum()
        return touch * 0.0

    def _step(self, batch, training: bool) -> dict[str, torch.Tensor]:
        z = self.encode(batch)
        device = z.device
        gts = batch["gt_wireframes"]

        verts_l, mask_l, adj_l, z_l, used = [], [], [], [], []
        pos_b, pos_i, pos_j, pos_pts = [], [], [], []
        for b, gt in enumerate(gts):
            s = self._prepare_sample(gt, training, device)
            if s is None:
                continue
            idx = len(verts_l)
            used.append(b)
            verts_l.append(s["verts"])
            mask_l.append(s["mask"])
            adj_l.append(s["adj"])
            z_l.append(z[b])
            e = s["pos_edges"]
            pos_b.append(torch.full((e.shape[0],), idx, device=device, dtype=torch.long))
            pos_i.append(e[:, 0])
            pos_j.append(e[:, 1])
            pos_pts.append(s["pos_points"])

        if not verts_l:
            # No usable shape in this batch (all skipped). Route a zero-scaled
            # loss through the WHOLE net so every trainable parameter still gets
            # a (zero) gradient -- otherwise DDP (find_unused_parameters=False)
            # raises on the unused edge-predictor params.
            zero = self._zero_loss(z)
            return {"loss": zero, "loss_edge": zero.detach(),
                    "loss_curve": zero.detach(), "out": None, "used": []}

        verts = torch.stack(verts_l, dim=0)
        mask = torch.stack(mask_l, dim=0)
        adj = torch.stack(adj_l, dim=0)
        zb = torch.stack(z_l, dim=0)
        out = self.net(verts, mask, zb)
        logits = out["edge_logits"]

        # edge BCE over the valid upper-triangular candidate pairs.
        bsz, vv, _ = logits.shape
        valid = mask[:, :, None] & mask[:, None, :]
        triu = torch.triu(torch.ones(vv, vv, dtype=torch.bool, device=device), 1)
        pair_mask = valid & triu[None]
        loss_edge = self._edge_bce(logits[pair_mask], adj[pair_mask])

        # curves on GT positive edges (prune-then-refine).
        bi = torch.cat(pos_b)
        ii = torch.cat(pos_i)
        jj = torch.cat(pos_j)
        gt_pts = torch.cat(pos_pts, dim=0)
        curves = self.net.curves_for_pairs(
            out["h"], verts, out["z_global"], bi, ii, jj)
        loss_curve = self._curve_loss(curves, gt_pts)

        loss = self.hparams.w_edge * loss_edge + self.hparams.w_curve * loss_curve
        return {
            "loss": loss,
            "loss_edge": loss_edge.detach(),
            "loss_curve": loss_curve.detach(),
            "out": out,
            "verts": verts,
            "mask": mask,
            "used": used,
        }

    def training_step(self, batch, batch_idx):
        res = self._step(batch, training=True)
        bs = batch["num_graphs"]
        self.log("train/loss", res["loss"], batch_size=bs,
                 prog_bar=True, sync_dist=True)
        self.log_dict(
            {"train/loss_edge": res["loss_edge"],
             "train/loss_curve": res["loss_curve"]},
            batch_size=bs, sync_dist=True,
        )
        return res["loss"]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode(
        self,
        out: dict[str, torch.Tensor],
        verts: torch.Tensor,
        mask: torch.Tensor,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Threshold edge logits + decode curves into wireframes (numpy)."""
        logits = out["edge_logits"].detach().cpu().numpy()
        thr = float(self.best_edge_threshold) if threshold is None else float(threshold)
        wfs: list[dict[str, Any]] = []
        for i in range(verts.shape[0]):
            nv = int(mask[i].sum().item())
            edge_index = edges_from_logits(logits[i], nv, threshold=thr)
            v_np = verts[i, :nv].detach().cpu().numpy()
            if edge_index.shape[0] == 0:
                wfs.append(assemble_wireframe(v_np, edge_index, None))
                continue
            ei = torch.from_numpy(edge_index).to(verts.device)
            bi = torch.full((ei.shape[0],), i, device=verts.device, dtype=torch.long)
            curves = self.net.curves_for_pairs(
                out["h"], verts, out["z_global"], bi, ei[:, 0], ei[:, 1])
            wfs.append(assemble_wireframe(
                v_np, edge_index, curves.detach().cpu().numpy()))
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

    @torch.no_grad()
    def _decode_clean(self, batch) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run the predictor on the CLEAN GT corners and decode wireframes."""
        z = self.encode(batch)
        device = z.device
        gts = batch["gt_wireframes"]
        verts_l, mask_l, z_l, used = [], [], [], []
        for b, gt in enumerate(gts):
            s = self._prepare_sample(gt, training=False, device=device)
            if s is None:
                continue
            verts_l.append(s["verts"])
            mask_l.append(s["mask"])
            z_l.append(z[b])
            used.append(b)
        if not verts_l:
            return [], []
        verts = torch.stack(verts_l, dim=0)
        mask = torch.stack(mask_l, dim=0)
        zb = torch.stack(z_l, dim=0)
        out = self.net(verts, mask, zb)
        preds = self.decode(out, verts, mask)
        gt_np = self._gt_to_numpy([gts[b] for b in used])
        return preds, gt_np

    def validation_step(self, batch, batch_idx):
        res = self._step(batch, training=False)
        bs = batch["num_graphs"]
        self.log("val/loss", res["loss"], batch_size=bs, sync_dist=True)
        self.log_dict(
            {"val/loss_edge": res["loss_edge"],
             "val/loss_curve": res["loss_curve"]},
            batch_size=bs, sync_dist=True,
        )
        # Reuse the step's forward (no second V^2 pass): decode + score the same
        # outputs against the GT graphs of the used (non-skipped) samples, at
        # every candidate threshold so the best operating point can be chosen.
        used = res.get("used") or []
        if used:
            gt_np = self._gt_to_numpy([batch["gt_wireframes"][b] for b in used])
            for t in self.edge_thresholds:
                preds = self.decode(res["out"], res["verts"], res["mask"],
                                    threshold=t)
                self.val_metrics[self._thr_key(t)].update(preds, gt_np)

    def on_validation_epoch_end(self) -> None:
        best_t, best_score, best_res = None, -1.0, None
        for t in self.edge_thresholds:
            res = self.val_metrics[self._thr_key(t)].compute()
            score = float(res["score"])
            self.log(f"val/score@{t:.2f}", score, sync_dist=False)
            if score > best_score:
                best_t, best_score, best_res = t, score, res
            self.val_metrics[self._thr_key(t)].reset()
        if best_res is not None:
            self.best_edge_threshold = self.best_edge_threshold.new_tensor(best_t)
            self.log_dict(
                {f"val/{k}": v for k, v in best_res.items() if k != "score"},
                prog_bar=False, sync_dist=False,
            )
            self.log("val/score", best_res["score"], prog_bar=True, sync_dist=False)
            self.log("val/edge_threshold", float(best_t), prog_bar=True,
                     sync_dist=False)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """Predict wireframes from the GT corners (val-set style).

        Full two-stage inference (stage-1 dedup'd vertices -> stage-2 edges) is
        assembled by the caller; here we use the GT corners when present.
        """
        if "gt_wireframes" not in batch:
            return {"shape_id": batch.get("shape_id"), "z": self.encode(batch)}
        preds, _ = self._decode_clean(batch)
        return {"shape_id": batch.get("shape_id"), "wireframes": preds}


__all__ = ["RFWireframeModule", "EdgePredictorModule"]
