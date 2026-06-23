"""LightningModule for the single-stage WireframeAE PC2Wireframe branch.

A single trainable autoencoder is driven through ``LightningCLI`` (see
``src/main.py``):

    packed point cloud (coord (P_sum, 3), offset (B,))   (native, variable size)
        -> UtoniaEncoder (frozen PTv3 + trainable latent compressor)
        -> latent z (B, 16, 256)                         (4096-float budget)
        -> WireframeAE decoder (vertex queries + pairwise edge head)
        -> wireframe {vertices, edge_index, edge_points}

The 16x256 latent is the competition submission; the decoder must reconstruct
the wireframe from it alone. Training is a DETR-style set-prediction problem:
the ``Q`` vertex queries are Hungarian-matched to the GT vertices (cost = xyz
L1), then supervised with matched vertex L1 + alive BCE, and the edges are
supervised on query pairs (existence BCE + curve type CE + anchor L1 + an
optional curve-geometry Chamfer). There is no KL term -- the latent is
deterministic.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .metrics import WireframeScore
from .models.curves import sample_curve_by_type
from .models.utonia_encoder import UtoniaEncoder
from .models.wireframe_ae import WireframeAE
from .recon import decode_wireframe


# ----------------------------------------------------------------------
# default sub-config builders (overridable from YAML)
# ----------------------------------------------------------------------
def _default_pc_encoder() -> dict[str, Any]:
    return dict(
        utonia="logs/utonia/utonia.pth",
        grid_size=0.01,
        freeze=True,
        # 16 * 256 = 4096 floats (competition latent budget).
        latent_num=16,
        latent_dim=256,
        compressor_heads=8,
        compressor_layers=1,
    )


def _default_decoder() -> dict[str, Any]:
    return dict(
        latent_dim=256,
        num_queries=512,
        d_model=256,
        nhead=8,
        num_layers=6,
        mlp_ratio=4.0,
        dropout=0.0,
        edge_hidden=256,
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
# WireframeAE module
# ----------------------------------------------------------------------
class WireframeAEModule(_BaseModule):
    """Point cloud -> latent -> WireframeAE decoder -> wireframe."""

    def __init__(
        self,
        # ----- model config (nested, fully overridable from YAML) -----
        pc_encoder: dict[str, Any] | None = None,
        decoder: dict[str, Any] | None = None,
        # ----- loss weights -----
        w_vertex: float = 1.0,
        w_alive: float = 1.0,
        w_edge: float = 1.0,
        w_edge_type: float = 1.0,
        w_edge_param: float = 1.0,
        w_curve_geom: float = 0.1,
        # alive BCE: positives are rare (V << Q), so up-weight them.
        alive_pos_weight: float = 5.0,
        # edge negatives are sampled at this multiple of the positive count.
        edge_neg_ratio: float = 3.0,
        # Hungarian matching cost = xyz L1 - match_alive_weight * alive_prob,
        # so confident-alive queries are preferred (stabilises the matching).
        match_alive_weight: float = 0.2,
        # curve type CE class weights (line / arc / bezier).
        curve_type_class_weights: list[float] | None = None,
        geom_num_per_edge: int = 32,
        # ----- decode (validation / predict) -----
        vertex_thresh: float = 0.5,
        edge_thresh: float = 0.5,
        max_decode_vertices: int = 512,
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
        self.decoder = WireframeAE(**(decoder or _default_decoder()))
        self.num_queries = int(self.decoder.num_queries)

        self.val_metrics = WireframeScore(
            w_ccd=eval_w_ccd, w_ta=eval_w_ta, w_vpe=eval_w_vpe,
            ccd_tau=eval_ccd_tau, vpe_tau=eval_vpe_tau,
            match_thresh=eval_match_thresh, num_per_edge=num_per_edge,
        )

    # ------------------------------------------------------------------
    def encode(self, batch: dict[str, Any]) -> torch.Tensor:
        """Latent ``z`` of shape ``(B, K, D)``."""
        return self.encoder(batch["point_cloud"], batch["pc_offset"])

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        z = self.encode(batch)
        out = self.decoder(z)
        out["latent"] = z
        return out

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    @staticmethod
    @torch.no_grad()
    def _match(
        pred_xyz: torch.Tensor,
        gt_v: torch.Tensor,
        alive_prob: torch.Tensor | None = None,
        alive_weight: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hungarian match queries -> GT vertices.

        Cost is the xyz L1 distance minus an optional alive-confidence reward
        (``alive_weight * sigmoid(alive_logit)``), so that -- among queries that
        are geometrically close to a GT vertex -- the one the model already
        believes is alive is preferred (DETR's classification cost term).

        Returns ``(row, col)`` (query ids matched to gt ids, length ``V``).
        """
        from scipy.optimize import linear_sum_assignment

        cost = torch.cdist(pred_xyz, gt_v, p=1)               # (Q, V)
        if alive_prob is not None and alive_weight > 0.0:
            cost = cost - float(alive_weight) * alive_prob[:, None]
        cost = cost.detach().cpu().numpy()
        row, col = linear_sum_assignment(cost)
        device = pred_xyz.device
        return (
            torch.as_tensor(row, dtype=torch.long, device=device),
            torch.as_tensor(col, dtype=torch.long, device=device),
        )

    def _loss(
        self, out: dict[str, torch.Tensor], gt_wireframes: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        vertex_logit = out["vertex_logit"]      # (B, Q)
        vertex_xyz = out["vertex_xyz"]          # (B, Q, 3)
        hidden = out["hidden"]                  # (B, Q, d)
        global_vec = out["global"]              # (B, d)
        device = vertex_logit.device
        b, q, _ = vertex_xyz.shape

        type_weight = (
            vertex_logit.new_tensor(list(self.hparams.curve_type_class_weights))
            if self.hparams.curve_type_class_weights is not None else None
        )

        l_vertex = vertex_xyz.new_zeros(())
        l_alive = vertex_xyz.new_zeros(())
        l_edge = vertex_xyz.new_zeros(())
        l_edge_type = vertex_xyz.new_zeros(())
        l_edge_param = vertex_xyz.new_zeros(())
        l_geom = vertex_xyz.new_zeros(())
        n_alive = n_vertex = n_edge = n_etype = n_geom = 0

        for i in range(b):
            g = gt_wireframes[i]
            gt_v = g["vertices"].to(device).float().reshape(-1, 3)
            v = gt_v.shape[0]

            alive_target = vertex_logit.new_zeros(q)
            if v == 0:
                l_alive = l_alive + F.binary_cross_entropy_with_logits(
                    vertex_logit[i], alive_target)
                n_alive += 1
                continue

            row, col = self._match(
                vertex_xyz[i], gt_v,
                torch.sigmoid(vertex_logit[i]),
                float(self.hparams.match_alive_weight),
            )                                             # query ids, gt ids
            alive_target[row] = 1.0

            # alive BCE (positives up-weighted) + matched vertex L1
            pos_w = vertex_logit.new_tensor(float(self.hparams.alive_pos_weight))
            l_alive = l_alive + F.binary_cross_entropy_with_logits(
                vertex_logit[i], alive_target, pos_weight=pos_w)
            l_vertex = l_vertex + F.l1_loss(
                vertex_xyz[i][row], gt_v[col])
            n_alive += 1
            n_vertex += 1

            # gt vertex id -> matched query id
            gt2q = vertex_logit.new_full((v,), -1, dtype=torch.long)
            gt2q[col] = row

            gt_edges = g["edge_index"].to(device).long().reshape(-1, 2)
            gt_type = g["edge_type"].to(device).long().reshape(-1)
            gt_params = g["edge_params"].to(device).float().reshape(-1, 2, 3)
            e = gt_edges.shape[0]

            # ---- positive edges: gt edges -> matched query pairs ----
            pos_a, pos_b, pos_t, pos_p, pos_eidx = [], [], [], [], []
            pos_set: set[tuple[int, int]] = set()
            for ei in range(e):
                gs, ge = int(gt_edges[ei, 0]), int(gt_edges[ei, 1])
                if gs < 0 or ge < 0 or gs >= v or ge >= v:
                    continue
                qs, qe = int(gt2q[gs]), int(gt2q[ge])
                if qs < 0 or qe < 0 or qs == qe:
                    continue
                # canonical ascending query order (q1 follows endpoint a)
                if qs <= qe:
                    aa, bb = qs, qe
                    p1, p2 = gt_params[ei, 0], gt_params[ei, 1]
                else:
                    aa, bb = qe, qs
                    p1, p2 = gt_params[ei, 1], gt_params[ei, 0]
                pair = (aa, bb)
                if pair in pos_set:
                    continue
                pos_set.add(pair)
                pos_a.append(aa)
                pos_b.append(bb)
                pos_t.append(int(gt_type[ei]))
                pos_p.append(torch.stack([p1, p2], dim=0))
                pos_eidx.append(ei)

            n_pos = len(pos_a)

            # ---- negative edges: random pairs (not in pos) drawn from the
            # matched queries *plus* the queries the model currently predicts
            # alive. The latter closes the train/inference gap: at decode time
            # edges are scored over all thresholded-alive queries, so the edge
            # head must also reject pairs among alive-but-unmatched queries.
            n_neg = int(round(max(1, n_pos) * float(self.hparams.edge_neg_ratio)))
            neg_a, neg_b = [], []
            matched_q = row.tolist()
            matched_set = set(matched_q)
            alive_pred = torch.nonzero(
                torch.sigmoid(vertex_logit[i].detach())
                >= float(self.hparams.vertex_thresh),
                as_tuple=False,
            ).reshape(-1).tolist()
            cand_q = matched_q + [
                int(x) for x in alive_pred if int(x) not in matched_set]
            if len(cand_q) >= 2:
                tries = 0
                max_tries = n_neg * 8 + 16
                while len(neg_a) < n_neg and tries < max_tries:
                    tries += 1
                    ia, ib = np.random.randint(0, len(cand_q), size=2)
                    if ia == ib:
                        continue
                    qa, qb = cand_q[ia], cand_q[ib]
                    lo, hi = (qa, qb) if qa < qb else (qb, qa)
                    if (lo, hi) in pos_set:
                        continue
                    neg_a.append(lo)
                    neg_b.append(hi)

            a_ids = pos_a + neg_a
            b_ids = pos_b + neg_b
            if a_ids:
                a_idx = torch.as_tensor(a_ids, dtype=torch.long, device=device)
                b_idx = torch.as_tensor(b_ids, dtype=torch.long, device=device)
                ehead = self.decoder.edge_logits(
                    hidden[i][a_idx], hidden[i][b_idx],
                    global_vec[i][None, :].expand(a_idx.shape[0], -1))
                exist_target = vertex_logit.new_zeros(a_idx.shape[0])
                exist_target[:n_pos] = 1.0
                # negatives outnumber positives by ~edge_neg_ratio: up-weight
                # the positive existence term to keep edge recall balanced.
                edge_pos_w = vertex_logit.new_tensor(
                    float(self.hparams.edge_neg_ratio))
                l_edge = l_edge + F.binary_cross_entropy_with_logits(
                    ehead["exist"], exist_target, pos_weight=edge_pos_w)
                n_edge += 1

                if n_pos > 0:
                    type_target = torch.as_tensor(
                        pos_t, dtype=torch.long, device=device)
                    l_edge_type = l_edge_type + F.cross_entropy(
                        ehead["type"][:n_pos], type_target, weight=type_weight)
                    param_target = torch.stack(pos_p, dim=0)  # (n_pos, 2, 3)
                    l_edge_param = l_edge_param + F.l1_loss(
                        ehead["params"][:n_pos], param_target)
                    n_etype += 1

                    # ---- optional curve-geometry Chamfer (matched edges) ----
                    if float(self.hparams.w_curve_geom) != 0.0:
                        gep = g["edge_points"].to(device).float()
                        eidx = torch.as_tensor(
                            pos_eidx, dtype=torch.long, device=device)
                        if gep.shape[0] and int(eidx.max()) < gep.shape[0]:
                            a_xyz = vertex_xyz[i][a_idx[:n_pos]]
                            b_xyz = vertex_xyz[i][b_idx[:n_pos]]
                            q1 = ehead["params"][:n_pos, 0]
                            q2 = ehead["params"][:n_pos, 1]
                            ct = ehead["type"][:n_pos].argmax(dim=-1)
                            pred_curve = sample_curve_by_type(
                                a_xyz, q1, q2, b_xyz, ct,
                                int(self.hparams.geom_num_per_edge))
                            gt_curve = gep[eidx]   # (n_pos, U, 3)
                            dmat = torch.cdist(pred_curve, gt_curve)
                            cd = 0.5 * (
                                dmat.min(dim=2)[0].mean(dim=1)
                                + dmat.min(dim=1)[0].mean(dim=1)
                            )
                            l_geom = l_geom + cd.mean()
                            n_geom += 1

        # ---- normalise + combine ----
        # Every term is a mean over the samples that actually contributed to it
        # (its own counter), so the relative loss magnitudes no longer drift
        # with the batch's mix of empty / non-empty wireframes.
        if n_alive > 0:
            l_alive = l_alive / n_alive
        if n_vertex > 0:
            l_vertex = l_vertex / n_vertex
        if n_edge > 0:
            l_edge = l_edge / n_edge
        if n_etype > 0:
            l_edge_type = l_edge_type / n_etype
            l_edge_param = l_edge_param / n_etype
        if n_geom > 0:
            l_geom = l_geom / n_geom

        total = (
            self.hparams.w_vertex * l_vertex
            + self.hparams.w_alive * l_alive
            + self.hparams.w_edge * l_edge
            + self.hparams.w_edge_type * l_edge_type
            + self.hparams.w_edge_param * l_edge_param
            + self.hparams.w_curve_geom * l_geom
        )
        return {
            "loss": total,
            "loss_vertex": l_vertex.detach(),
            "loss_alive": l_alive.detach(),
            "loss_edge": l_edge.detach(),
            "loss_edge_type": l_edge_type.detach(),
            "loss_edge_param": l_edge_param.detach(),
            "loss_curve_geom": l_geom.detach(),
        }

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        losses = self._loss(out, batch["gt_wireframes"])
        bs = batch["num_graphs"]
        self.log("train/loss", losses["loss"], batch_size=bs,
                 prog_bar=True, sync_dist=True)
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items() if k != "loss"},
            batch_size=bs, sync_dist=True,
        )
        return losses["loss"]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, out: dict[str, torch.Tensor]) -> list[dict[str, np.ndarray]]:
        """Decode a batch of decoder outputs into wireframes (numpy)."""
        vertex_logit = out["vertex_logit"]
        vertex_xyz = out["vertex_xyz"]
        hidden = out["hidden"]
        global_vec = out["global"]
        device = vertex_logit.device
        b = vertex_logit.shape[0]
        vt = float(self.hparams.vertex_thresh)
        cap = int(self.hparams.max_decode_vertices)

        wfs: list[dict[str, np.ndarray]] = []
        for i in range(b):
            prob = torch.sigmoid(vertex_logit[i])
            alive = torch.nonzero(prob >= vt, as_tuple=False).reshape(-1)
            if alive.numel() > cap > 0:
                alive = alive[torch.topk(prob[alive], cap).indices]
            if alive.numel() < 2:
                wfs.append({
                    "vertices": vertex_xyz[i][alive].detach().cpu().numpy()
                    .astype(np.float32).reshape(-1, 3),
                    "pair_index": np.zeros((0, 2), dtype=np.int64),
                    "edge_prob": np.zeros((0,), dtype=np.float32),
                    "edge_type": np.zeros((0,), dtype=np.int64),
                    "q1": np.zeros((0, 3), dtype=np.float32),
                    "q2": np.zeros((0, 3), dtype=np.float32),
                })
                continue

            verts = vertex_xyz[i][alive]                 # (V, 3)
            va = alive.shape[0]
            iu, ju = torch.triu_indices(va, va, offset=1, device=device)
            h = hidden[i][alive]
            ehead = self.decoder.edge_logits(
                h[iu], h[ju],
                global_vec[i][None, :].expand(iu.shape[0], -1))
            wfs.append({
                "vertices": verts.detach().cpu().numpy().astype(np.float32),
                "pair_index": torch.stack([iu, ju], dim=1).cpu().numpy(),
                "edge_prob": torch.sigmoid(ehead["exist"]).detach().cpu().numpy(),
                "edge_type": ehead["type"].argmax(dim=-1).detach().cpu().numpy(),
                "q1": ehead["params"][:, 0].detach().cpu().numpy(),
                "q2": ehead["params"][:, 1].detach().cpu().numpy(),
            })
        return wfs

    def decode_to_wireframes(
        self, out: dict[str, torch.Tensor]
    ) -> list[dict[str, np.ndarray]]:
        fields = self.decode(out)
        return [
            decode_wireframe(
                f,
                edge_thresh=float(self.hparams.edge_thresh),
                num_per_edge=int(self.hparams.num_per_edge),
            )
            for f in fields
        ]

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
        out = self.forward(batch)
        losses = self._loss(out, batch["gt_wireframes"])
        bs = batch["num_graphs"]
        self.log("val/loss", losses["loss"], batch_size=bs, prog_bar=True,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(
            {f"val/{k}": v for k, v in losses.items() if k != "loss"},
            batch_size=bs, on_step=False, on_epoch=True, sync_dist=True,
        )
        preds = self.decode_to_wireframes(out)
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
        """Per-shape latent + decoded wireframe.

        The latent ``z (B, K, D)`` is the competition submission; the wireframe
        is decoded from it for visualisation / scoring.
        """
        out = self.forward(batch)
        preds = self.decode_to_wireframes(out)
        return {
            "shape_id": batch.get("shape_id"),
            "latent": out["latent"],
            "wireframes": preds,
        }


__all__ = ["WireframeAEModule"]
