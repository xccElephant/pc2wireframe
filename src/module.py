"""LightningModule for the VQVAE PC2Wireframe branch.

A single trainable, end-to-end discrete autoencoder driven through
``LightningCLI`` (see ``src/main.py``):

    packed point cloud (coord (P_sum, 3), offset (B,))   (native, variable size)
        -> UtoniaEncoder (frozen PTv3, multi-scale per-stage features)
        -> per-scale continuous tokens  z_s (B, N_s, 256)
        -> MultiScaleResidualVQ (per-scale ResidualVQ)
        -> per-scale quantized z_q_s + flat indices (B, T<=4096)   (submission)
        -> WireframeGraphDecoder (vertex queries + GNN + kNN pairwise edges)
        -> wireframe {vertices, edge_index, edge_points}

The flat **indices** (``T = sum_s N_s * n_q <= 4096``) are the competition
submission; the decoder must reconstruct the wireframe from them alone (indices
-> codebooks -> z_q -> decoder). Training is a DETR-style set-prediction
problem: the ``Q`` vertex queries are Hungarian-matched to the GT vertices
(cost = xyz L1), supervised with matched vertex L1 + alive BCE; edges are scored
over kNN candidate pairs (unioned with the GT positives at train time) with a
focal existence loss + curve type CE + anchor L1 + an optional curve-geometry
Chamfer. A VQ commitment loss (ramped in after a continuous-``z`` warmup) trains
the codebooks; there is no KL term.
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
from .models.quantizer import MultiScaleResidualVQ
from .models.utonia_encoder import UtoniaEncoder
from .models.wireframe_graph_decoder import (
    WireframeGraphDecoder,
    knn_candidate_pairs,
)
from .recon import decode_wireframe


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
        num_queries=512,
        d_model=256,
        nhead=8,
        num_layers=6,
        gnn_rounds=3,
        knn_k=24,
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
# loss helpers
# ----------------------------------------------------------------------
def _focal_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    alpha: float,
) -> torch.Tensor:
    """Mean binary focal loss (Lin et al.) for the heavily-imbalanced edges."""
    if logits.numel() == 0:
        return logits.new_zeros(())
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).clamp_min(0.0) ** gamma * ce).mean()


# ----------------------------------------------------------------------
# WireframeAE module (VQVAE pipeline)
# ----------------------------------------------------------------------
class WireframeAEModule(_BaseModule):
    """Point cloud -> multi-scale RVQ indices -> graph decoder -> wireframe."""

    def __init__(
        self,
        # ----- model config (nested, fully overridable from YAML) -----
        pc_encoder: dict[str, Any] | None = None,
        quantizer: dict[str, Any] | None = None,
        decoder: dict[str, Any] | None = None,
        # ----- loss weights -----
        w_vertex: float = 1.0,
        w_alive: float = 1.0,
        w_edge: float = 1.0,
        w_edge_type: float = 1.0,
        w_edge_param: float = 1.0,
        w_curve_geom: float = 0.2,
        w_commit: float = 0.25,
        # alive BCE: positives are rare (V << Q), so up-weight them.
        alive_pos_weight: float = 10.0,
        # edge existence focal-loss params.
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        # Hungarian matching cost = xyz L1 - match_alive_weight * alive_prob.
        match_alive_weight: float = 0.2,
        # curve type CE class weights (line / arc / bezier).
        curve_type_class_weights: list[float] | None = None,
        geom_num_per_edge: int = 32,
        # ----- quantization schedule -----
        quant_warmup_steps: int = 2000,   # continuous-z warmup (no commit)
        commit_ramp_steps: int = 2000,    # commit weight 0 -> w_commit ramp
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
        self.decoder = WireframeGraphDecoder(**dec_cfg)
        self.num_queries = int(self.decoder.num_queries)
        self.knn_k = int(self.decoder.knn_k)

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
        """Hungarian match queries -> GT vertices (cost = xyz L1 - alive reward).

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
        gamma = float(self.hparams.focal_gamma)
        alpha = float(self.hparams.focal_alpha)

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

            pos_w = vertex_logit.new_tensor(float(self.hparams.alive_pos_weight))
            l_alive = l_alive + F.binary_cross_entropy_with_logits(
                vertex_logit[i], alive_target, pos_weight=pos_w)
            l_vertex = l_vertex + F.l1_loss(vertex_xyz[i][row], gt_v[col])
            n_alive += 1
            n_vertex += 1

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

            # ---- candidate negatives: kNN of the predicted xyz over the
            # matched-plus-predicted-alive query set (mirrors inference, where
            # edges are scored over the kNN of the thresholded-alive vertices).
            # Training candidates = GT positives UNION kNN pairs (minus pos).
            matched_q = row.tolist()
            matched_set = set(matched_q)
            alive_pred = torch.nonzero(
                torch.sigmoid(vertex_logit[i].detach())
                >= float(self.hparams.vertex_thresh),
                as_tuple=False,
            ).reshape(-1).tolist()
            cand_q = matched_q + [
                int(x) for x in alive_pred if int(x) not in matched_set]
            neg_a, neg_b = [], []
            if len(cand_q) >= 2:
                cand_t = torch.as_tensor(cand_q, dtype=torch.long, device=device)
                knn_local = knn_candidate_pairs(
                    vertex_xyz[i][cand_t].detach(), self.knn_k)
                if knn_local.numel():
                    gi = cand_t[knn_local[:, 0]]
                    gj = cand_t[knn_local[:, 1]]
                    lo = torch.minimum(gi, gj).tolist()
                    hi = torch.maximum(gi, gj).tolist()
                    for la, lb in zip(lo, hi):
                        if la == lb or (la, lb) in pos_set:
                            continue
                        neg_a.append(la)
                        neg_b.append(lb)

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
                l_edge = l_edge + _focal_bce(
                    ehead["exist"], exist_target, gamma, alpha)
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
            "loss_geom": total,
            "loss_vertex": l_vertex.detach(),
            "loss_alive": l_alive.detach(),
            "loss_edge": l_edge.detach(),
            "loss_edge_type": l_edge_type.detach(),
            "loss_edge_param": l_edge_param.detach(),
            "loss_curve_geom": l_geom.detach(),
        }

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
    def decode(self, out: dict[str, torch.Tensor]) -> list[dict[str, np.ndarray]]:
        """Decode a batch of decoder outputs into wireframes (numpy).

        Edges are scored only over the kNN candidate pairs of the alive vertices
        (``O(V*k)``), matching the training candidates and the export path.
        """
        vertex_logit = out["vertex_logit"]
        vertex_xyz = out["vertex_xyz"]
        hidden = out["hidden"]
        global_vec = out["global"]
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
            pairs = knn_candidate_pairs(verts.detach(), self.knn_k)  # (P, 2)
            h = hidden[i][alive]
            iu, ju = pairs[:, 0], pairs[:, 1]
            ehead = self.decoder.edge_logits(
                h[iu], h[ju],
                global_vec[i][None, :].expand(iu.shape[0], -1))
            wfs.append({
                "vertices": verts.detach().cpu().numpy().astype(np.float32),
                "pair_index": pairs.cpu().numpy(),
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
