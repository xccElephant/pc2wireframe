"""Joint vertex + edge set criterion (Hungarian-matched, per shape).

Supervises the :class:`~src.models.joint_set_decoder.JointSetDecoder` outputs and
the **jointly trained** curve VAE. For each shape it:

1. **vertex matching** -- Hungarian assignment of vertex queries to GT vertices
   on ``coord_L1 - w_exist * sigmoid(vexist)``. Yields a GT-vertex -> vertex-query
   map ``qv_of_gt``. Losses: calibrated existence BCE (matched = positive) +
   matched-vertex coordinate L1.

2. **curve VAE autoencoding anchor** -- the GT curves are oriented +
   canonicalised (endpoints -> ``[-1,0,0]`` / ``[1,0,0]``), encoded, sampled and
   decoded; a length-weighted reconstruction L1 + endpoint L1 + ``kl_weight * KL``
   keeps the (trainable) latent space meaningful and prevents collapse. This
   path is the clean supervision that anchors the shared decoder.

3. **A-based edge matching** -- Hungarian assignment of edge queries to GT edges
   on ``w_inc*(-logA[e,qa]-logA[e,qb]) + w_lat*||lat[e]-sg(mu_k)||_1 -
   w_exist*sigmoid(eexist)`` where ``qa, qb`` are the GT edge's endpoint vertex
   queries and ``mu_k`` is the GT canonical curve's (stop-grad) posterior mean.
   ``w_inc`` is ramped from 0 (early ``A`` is random) so matching is first driven
   by latent + existence.

4. **edge losses** -- calibrated existence BCE + a curve loss in **curve space**
   (``L1(decode(lat_pred[e]), GT canonical curve)``; the target is the fixed GT
   curve, never a drifting latent) through the shared trainable decoder, plus a
   small optional ``L2(lat_pred, sg(mu_k))`` on-manifold regulariser.

5. **association loss** -- BCE on ``A_logit`` restricted to the matched edge rows
   and matched vertex columns, with a ``pos_weight`` for the 2-of-V sparsity.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .vae.curve_packing import encode_curve_mu, normalized_curves_from_batch


def _focal_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    alpha: float,
) -> torch.Tensor:
    """Mean binary focal loss (Lin et al.) for the imbalanced existence head."""
    if logits.numel() == 0:
        return logits.new_zeros(())
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).clamp_min(0.0) ** gamma * ce).mean()


def _balanced_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float,
    label_smoothing: float,
) -> torch.Tensor:
    """Calibrated existence BCE: per-shape ``pos_weight`` + label smoothing.

    ``pos_weight`` (= #neg / #pos, capped by the caller) rebalances the heavy
    negative majority without squashing the probabilities the way a focal loss
    does, so ``sigmoid(logit)`` stays near ``0.5`` at the decision boundary. A
    small ``label_smoothing`` keeps the head from saturating into over-confident
    logits.
    """
    if logits.numel() == 0:
        return logits.new_zeros(())
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    pw = logits.new_tensor(float(pos_weight))
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pw, reduction="mean")


class JointSetCriterion(nn.Module):
    """Joint vertex/edge Hungarian matching + curve-VAE co-training losses."""

    def __init__(
        self,
        *,
        # vertex losses
        w_vexist: float = 2.0,
        w_vcoord: float = 5.0,
        # edge losses
        w_eexist: float = 2.0,
        w_curve: float = 5.0,
        w_curve_endpoint: float = 2.0,
        w_lat_reg: float = 0.1,
        # curve VAE autoencoding anchor
        w_anchor: float = 5.0,
        w_anchor_endpoint: float = 2.0,
        kl_weight: float = 1e-6,
        # association
        w_assoc: float = 2.0,
        assoc_pos_weight_max: float = 64.0,
        # calibrated existence BCE
        focal_gamma: float = 0.0,
        focal_alpha: float = 0.5,
        exist_label_smoothing: float = 0.02,
        exist_pos_weight_max: float = 20.0,
        # matching costs
        match_w_vcoord: float = 1.0,
        match_w_exist: float = 0.5,
        match_w_inc: float = 1.0,
        match_w_lat: float = 1.0,
    ) -> None:
        super().__init__()
        self.w_vexist = float(w_vexist)
        self.w_vcoord = float(w_vcoord)
        self.w_eexist = float(w_eexist)
        self.w_curve = float(w_curve)
        self.w_curve_endpoint = float(w_curve_endpoint)
        self.w_lat_reg = float(w_lat_reg)
        self.w_anchor = float(w_anchor)
        self.w_anchor_endpoint = float(w_anchor_endpoint)
        self.kl_weight = float(kl_weight)
        self.w_assoc = float(w_assoc)
        self.assoc_pos_weight_max = float(assoc_pos_weight_max)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.exist_label_smoothing = float(exist_label_smoothing)
        self.exist_pos_weight_max = float(exist_pos_weight_max)
        self.match_w_vcoord = float(match_w_vcoord)
        self.match_w_exist = float(match_w_exist)
        self.match_w_inc = float(match_w_inc)
        self.match_w_lat = float(match_w_lat)

    # ------------------------------------------------------------------
    def _exist_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        n_pos: int,
    ) -> torch.Tensor:
        """Existence loss: focal (``focal_gamma > 0``) else calibrated BCE."""
        if self.focal_gamma > 0.0:
            return _focal_bce(
                logits, targets, self.focal_gamma, self.focal_alpha)
        q = int(logits.numel())
        n_neg = max(0, q - int(n_pos))
        pos_weight = n_neg / max(1, int(n_pos))
        pos_weight = float(min(self.exist_pos_weight_max, max(1.0, pos_weight)))
        return _balanced_bce(
            logits, targets, pos_weight, self.exist_label_smoothing)

    @staticmethod
    @torch.no_grad()
    def _hungarian(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from scipy.optimize import linear_sum_assignment

        # Defensive sanitise: ``linear_sum_assignment`` raises on any NaN/Inf in
        # the cost matrix. A non-finite entry should never reach here once the
        # curve canonicalisation is guarded, but a single anomalous value must
        # not crash a whole (multi-hour, multi-rank) run -- map them to a large
        # finite cost so the assignment simply avoids them.
        cost_np = cost.detach().float().cpu().numpy()
        cost_np = np.nan_to_num(cost_np, nan=1e6, posinf=1e6, neginf=-1e6)
        row, col = linear_sum_assignment(cost_np)
        device = cost.device
        return (torch.as_tensor(row, dtype=torch.long, device=device),
                torch.as_tensor(col, dtype=torch.long, device=device))

    # ------------------------------------------------------------------
    def forward(
        self,
        out: dict[str, torch.Tensor],
        gt_wireframes: list[dict[str, Any]],
        curve_vae: nn.Module,
        *,
        w_inc: float = 1.0,
        kl_weight: float | None = None,
    ) -> dict[str, torch.Tensor]:
        vexist = out["vertex_exist_logit"]            # (B, Nv)
        vcoord = out["vertex_coord"]                  # (B, Nv, 3)
        eexist = out["edge_exist_logit"]              # (B, Ne)
        latent = out["curve_latent"]                  # (B, Ne, D)
        assoc = out["assoc_logit"]                    # (B, Ne, Nv)
        device = vexist.device
        b, nv = vexist.shape
        ne = eexist.shape[1]
        kl_w = self.kl_weight if kl_weight is None else float(kl_weight)
        w_inc = float(w_inc)

        canon_list = normalized_curves_from_batch(gt_wireframes, device)
        p = next((c.shape[1] for c in canon_list if c.shape[0] > 0), 32)

        l_vexist = vexist.new_zeros(())
        l_vcoord = vexist.new_zeros(())
        l_eexist = vexist.new_zeros(())
        l_lat_reg = vexist.new_zeros(())
        l_assoc = vexist.new_zeros(())
        n_v = n_e = n_assoc = 0
        n_matched_v = n_matched_e = 0

        # Accumulators for the batched curve-VAE paths.
        anchor_canon: list[torch.Tensor] = []          # encode w/ grad
        pred_lat_matched: list[torch.Tensor] = []       # decode w/ grad
        target_canon_matched: list[torch.Tensor] = []   # fixed GT target

        # Per-shape posterior means for matching (encoded once, stop-grad).
        mu_all = self._encode_mu(curve_vae, canon_list)   # list[(E_i, D)]

        for i in range(b):
            g = gt_wireframes[i]
            gverts = g["vertices"].to(device).float().reshape(-1, 3)
            gei = g["edge_index"].to(device).long().reshape(-1, 2)
            canon = canon_list[i]
            v = gverts.shape[0]
            e = gei.shape[0]

            # ---- vertex matching + losses ----
            vexist_target = vexist.new_zeros(nv)
            qv_of_gt = torch.full((max(v, 1),), -1, dtype=torch.long,
                                  device=device)
            if v > 0:
                cost_v = (self.match_w_vcoord
                          * torch.cdist(vcoord[i], gverts, p=1)
                          - self.match_w_exist
                          * torch.sigmoid(vexist[i])[:, None])
                vrow, vcol = self._hungarian(cost_v)
                vexist_target[vrow] = 1.0
                qv_of_gt[vcol] = vrow
                l_vcoord = l_vcoord + (
                    vcoord[i][vrow] - gverts[vcol]).abs().mean()
                n_matched_v += int(vrow.shape[0])
            else:
                vrow = torch.zeros(0, dtype=torch.long, device=device)
            l_vexist = l_vexist + self._exist_loss(
                vexist[i], vexist_target, n_pos=int(vexist_target.sum().item()))
            n_v += 1

            # ---- curve VAE anchor target ----
            if canon.shape[0] > 0:
                anchor_canon.append(canon)

            # ---- edge matching + losses ----
            eexist_target = eexist.new_zeros(ne)
            if e > 0 and v > 0 and canon.shape[0] > 0:
                mu_i = mu_all[i]                          # (E, D) stop-grad
                qa = qv_of_gt[gei[:, 0].clamp(min=0)]     # (E,)
                qb = qv_of_gt[gei[:, 1].clamp(min=0)]
                logA = F.logsigmoid(assoc[i])             # (Ne, Nv)
                inc = -(logA[:, qa] + logA[:, qb])        # (Ne, E)
                lat_cost = torch.cdist(latent[i], mu_i, p=1)   # (Ne, E)
                cost_e = (w_inc * self.match_w_inc * inc
                          + self.match_w_lat * lat_cost
                          - self.match_w_exist
                          * torch.sigmoid(eexist[i])[:, None])
                erow, ecol = self._hungarian(cost_e)
                eexist_target[erow] = 1.0
                n_matched_e += int(erow.shape[0])

                pred_lat_matched.append(latent[i][erow])
                target_canon_matched.append(canon[ecol])
                l_lat_reg = l_lat_reg + (
                    latent[i][erow] - mu_i[ecol]).pow(2).mean()

                # ---- association BCE (matched rows, matched vertex cols) ----
                if vrow.shape[0] > 0:
                    sub = assoc[i][erow][:, vrow]          # (M, V)
                    qa_m = qa[ecol]                        # (M,)
                    qb_m = qb[ecol]
                    tgt = ((vrow[None, :] == qa_m[:, None])
                           | (vrow[None, :] == qb_m[:, None])).float()
                    n_col = max(1, sub.shape[1])
                    pw = float(min(self.assoc_pos_weight_max,
                                   max(1.0, (n_col - 2) / 2.0)))
                    l_assoc = l_assoc + F.binary_cross_entropy_with_logits(
                        sub, tgt, pos_weight=sub.new_tensor(pw))
                    n_assoc += 1
            l_eexist = l_eexist + self._exist_loss(
                eexist[i], eexist_target, n_pos=int(eexist_target.sum().item()))
            n_e += 1

        # ---- batched curve VAE anchor (encode w/ grad -> sample -> decode) ----
        l_anchor = vexist.new_zeros(())
        l_anchor_ep = vexist.new_zeros(())
        l_kl = vexist.new_zeros(())
        if anchor_canon:
            l_anchor, l_anchor_ep, l_kl = self._anchor_loss(
                curve_vae, torch.cat(anchor_canon, dim=0))

        # ---- batched edge curve loss (decode predicted latents) ----
        l_curve = vexist.new_zeros(())
        l_curve_ep = vexist.new_zeros(())
        if pred_lat_matched:
            l_curve, l_curve_ep = self._edge_curve_loss(
                curve_vae,
                torch.cat(pred_lat_matched, dim=0),
                torch.cat(target_canon_matched, dim=0),
                p,
            )

        if n_v > 0:
            l_vexist = l_vexist / n_v
            l_vcoord = l_vcoord / max(1, n_v)
        if n_e > 0:
            l_eexist = l_eexist / n_e
        n_matched_shapes = max(1, len(pred_lat_matched))
        l_lat_reg = l_lat_reg / n_matched_shapes
        if n_assoc > 0:
            l_assoc = l_assoc / n_assoc

        total = (
            self.w_vexist * l_vexist
            + self.w_vcoord * l_vcoord
            + self.w_eexist * l_eexist
            + self.w_curve * l_curve
            + self.w_curve_endpoint * l_curve_ep
            + self.w_lat_reg * l_lat_reg
            + self.w_anchor * l_anchor
            + self.w_anchor_endpoint * l_anchor_ep
            + kl_w * l_kl
            + self.w_assoc * l_assoc
        )
        return {
            "loss_geom": total,
            "loss_vexist": l_vexist.detach(),
            "loss_vcoord": l_vcoord.detach(),
            "loss_eexist": l_eexist.detach(),
            "loss_curve": l_curve.detach(),
            "loss_curve_endpoint": l_curve_ep.detach(),
            "loss_lat_reg": l_lat_reg.detach(),
            "loss_anchor": l_anchor.detach(),
            "loss_anchor_endpoint": l_anchor_ep.detach(),
            "loss_kl": l_kl.detach(),
            "loss_assoc": l_assoc.detach(),
            "matched_vertices": vexist.new_tensor(n_matched_v / max(1, b)),
            "matched_edges": vexist.new_tensor(n_matched_e / max(1, b)),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_mu(
        self, curve_vae: nn.Module, canon_list: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Stop-grad posterior means per shape (for the matching cost only).

        Encodes all shapes' curves in one batched pass (under ``no_grad`` so the
        means never feed gradients into the matching), then splits back per shape.
        """
        d = int(curve_vae.config.latent_channels * curve_vae.latent_len)
        sizes = [c.shape[0] for c in canon_list]
        nonempty = [c for c in canon_list if c.shape[0] > 0]
        if not nonempty:
            return [c.new_zeros((0, d)) for c in canon_list]
        mu = encode_curve_mu(curve_vae, torch.cat(nonempty, dim=0))   # (N, D)
        out: list[torch.Tensor] = []
        k = 0
        for n in sizes:
            if n == 0:
                out.append(mu.new_zeros((0, d)))
            else:
                out.append(mu[k:k + n])
                k += n
        return out

    def _anchor_loss(
        self, curve_vae: nn.Module, canon: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Length-weighted recon L1 + endpoint L1 + KL on GT canonical curves."""
        x = rearrange(canon, "e p c -> e c p")            # (E, 3, P)
        posterior = curve_vae.encode(x)
        z = posterior.sample()
        p = canon.shape[1]
        t = torch.linspace(0.0, 1.0, p, device=canon.device, dtype=canon.dtype)
        t = t.unsqueeze(0).expand(z.shape[0], -1)
        dec = rearrange(curve_vae.decode(z, t), "e c p -> e p c")   # (E, P, 3)
        seg = (canon[:, 1:] - canon[:, :-1]).norm(dim=-1).sum(dim=-1)
        weights = torch.log(seg.clamp(min=2.0, max=31.4) + 0.2)     # (E,)
        per_curve = (dec - canon).abs().mean(dim=[1, 2])            # (E,)
        recon = (per_curve * weights).sum() / weights.sum().clamp_min(1e-6)
        endpoint = (dec[:, [0, -1]] - canon[:, [0, -1]]).abs().mean()
        kl = posterior.kl().mean()
        return recon, endpoint, kl

    def _edge_curve_loss(
        self,
        curve_vae: nn.Module,
        pred_latent: torch.Tensor,
        target_canon: torch.Tensor,
        num_points: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Curve-space L1 of decoded predicted latents vs fixed GT curves."""
        ch = int(curve_vae.config.latent_channels)
        z = rearrange(pred_latent, "m (c l) -> m c l", c=ch)
        p = target_canon.shape[1]
        t = torch.linspace(0.0, 1.0, p, device=pred_latent.device,
                           dtype=pred_latent.dtype)
        t = t.unsqueeze(0).expand(z.shape[0], -1)
        dec = rearrange(curve_vae.decode(z, t), "m c p -> m p c")   # (M, P, 3)
        curve = (dec - target_canon).abs().mean()
        endpoint = (dec[:, [0, -1]] - target_canon[:, [0, -1]]).abs().mean()
        return curve, endpoint


__all__ = ["JointSetCriterion"]
