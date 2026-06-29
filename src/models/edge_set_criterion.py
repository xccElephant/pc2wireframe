"""Edge-set prediction criterion (Hungarian-matched, per shape).

This is the WireframeDETR / PBWR-style set criterion for the
:class:`~src.models.edge_set_decoder.EdgeSetDecoder`. For every shape the
``Ne`` (= 512) edge queries are matched to the GT edges with the Hungarian
algorithm and supervised as a set:

1. **matching cost** -- for each (query, GT-edge) pair:
     * **endpoint L1** -- ``min`` over the two endpoint orderings (a -> b vs
       b -> a), so matching is orientation-free;
     * **existence** -- ``- w * sigmoid(exist)`` (prefer confident queries);
     * **curve latent** -- L1 between the predicted curve latent and the GT
       canonical curve's (stop-grad) posterior mean ``mu`` (a cheap proxy for
       the decoded-curve distance).

2. **existence loss** -- focal / calibrated BCE over **all** queries (matched =
   positive, the rest = "no-object"); this is what lets the model leave the
   surplus queries empty instead of hallucinating edges.

3. **endpoint L1** -- on matched pairs, the better of the two endpoint orderings.

4. **curve loss** -- on matched pairs, the predicted curve latent is decoded
   through the **frozen** curve VAE and compared (L1 + endpoint L1) to the fixed
   GT canonical curve, in the canonical frame.

5. **latent regulariser** -- a small ``L2(curve_latent, sg(mu))`` keeping the
   predicted latents on the curve-VAE manifold.

Deep supervision reuses the final-layer matching on every intermediate
``aux`` layer (cheaper + stable targets), scaled by ``aux_weight``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from .packing import build_targets, decode_curve_latent
from .vae.curve_packing import encode_curve_mu


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary focal loss (RetinaNet) over logits / {0,1} targets."""
    if logits.numel() == 0:
        return logits.new_zeros(())
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce * (1.0 - p_t).clamp_min(0.0).pow(gamma)
    if alpha >= 0.0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()


class EdgeSetCriterion(nn.Module):
    """Hungarian edge-set matching + endpoint / curve / existence losses."""

    def __init__(
        self,
        *,
        # ----- loss weights -----
        edge_exist_weight: float = 2.0,
        endpoint_weight: float = 5.0,
        curve_weight: float = 5.0,
        curve_endpoint_weight: float = 2.0,
        lat_reg_weight: float = 0.1,
        aux_weight: float = 1.0,
        # ----- existence loss: focal (gamma>0) else calibrated BCE -----
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        exist_pos_weight_max: float = 40.0,
        # ----- matching costs -----
        match_endpoint: float = 5.0,
        match_exist: float = 1.0,
        match_lat: float = 1.0,
        # ----- curve decode -----
        num_curve_points: int = 32,
    ) -> None:
        super().__init__()
        self.edge_exist_w = float(edge_exist_weight)
        self.endpoint_w = float(endpoint_weight)
        self.curve_w = float(curve_weight)
        self.curve_endpoint_w = float(curve_endpoint_weight)
        self.lat_reg_w = float(lat_reg_weight)
        self.aux_w = float(aux_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.exist_pos_weight_max = float(exist_pos_weight_max)
        self.match_endpoint = float(match_endpoint)
        self.match_exist = float(match_exist)
        self.match_lat = float(match_lat)
        self.num_curve_points = int(num_curve_points)

    # ------------------------------------------------------------------
    def _exist_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Existence loss: focal (``focal_gamma > 0``) else calibrated BCE."""
        if self.focal_gamma > 0.0:
            return sigmoid_focal_loss(
                logits, target, alpha=self.focal_alpha,
                gamma=self.focal_gamma, reduction="mean")
        n_pos = float(target.sum().item())
        n_neg = float(target.numel()) - n_pos
        pw = min(self.exist_pos_weight_max, max(1.0, n_neg / max(1.0, n_pos)))
        return F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=logits.new_tensor(pw))

    @staticmethod
    @torch.no_grad()
    def _hungarian(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cost_np = cost.detach().float().cpu().numpy()
        cost_np = np.nan_to_num(cost_np, nan=1e6, posinf=1e6, neginf=-1e6)
        row, col = linear_sum_assignment(cost_np)
        dev = cost.device
        return (torch.as_tensor(row, dtype=torch.long, device=dev),
                torch.as_tensor(col, dtype=torch.long, device=dev))

    @staticmethod
    def _endpoint_cost(
        pred_ep: torch.Tensor, gt_ep: torch.Tensor
    ) -> torch.Tensor:
        """Pairwise orientation-free endpoint L1 cost ``(Ne, ng)``.

        ``pred_ep (Ne, 2, 3)``, ``gt_ep (ng, 2, 3)``; for every pair take the
        smaller of the two endpoint orderings (a->b / b->a).
        """
        pa, pb = pred_ep[:, 0], pred_ep[:, 1]      # (Ne, 3)
        ga, gb = gt_ep[:, 0], gt_ep[:, 1]          # (ng, 3)
        d_aa = torch.cdist(pa, ga, p=1)
        d_bb = torch.cdist(pb, gb, p=1)
        d_ab = torch.cdist(pa, gb, p=1)
        d_ba = torch.cdist(pb, ga, p=1)
        return torch.minimum(d_aa + d_bb, d_ab + d_ba)

    @staticmethod
    def _endpoint_l1(
        pred_ep: torch.Tensor, gt_ep: torch.Tensor
    ) -> torch.Tensor:
        """Matched orientation-free endpoint L1 (mean), ``(M,2,3)`` each."""
        if pred_ep.shape[0] == 0:
            return pred_ep.new_zeros(())
        c1 = (pred_ep - gt_ep).abs().sum(dim=[1, 2])
        flip = torch.flip(gt_ep, dims=[1])
        c2 = (pred_ep - flip).abs().sum(dim=[1, 2])
        return torch.minimum(c1, c2).mean()

    # ------------------------------------------------------------------
    def _layer_loss(
        self,
        preds: dict[str, torch.Tensor],
        match: list[dict[str, torch.Tensor]],
        exist_target: torch.Tensor,
        curve_vae: nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Existence + endpoint + curve losses for one (final / aux) layer."""
        exist_logit = preds["edge_exist_logit"]
        endpoints = preds["endpoints"]
        curve_latent = preds["curve_latent"]
        device = exist_logit.device

        l_exist = self._exist_loss(exist_logit, exist_target)

        l_ep = exist_logit.new_zeros(())
        n_ep = 0
        matched_pred_lat: list[torch.Tensor] = []
        matched_gt_canon: list[torch.Tensor] = []
        matched_gt_mu: list[torch.Tensor] = []
        for s, m in enumerate(match):
            qi = m["qi"]
            if qi.numel() == 0:
                continue
            l_ep = l_ep + self._endpoint_l1(
                endpoints[s, qi], m["gt_ep"]) * qi.shape[0]
            n_ep += int(qi.shape[0])
            matched_pred_lat.append(curve_latent[s, qi])
            matched_gt_canon.append(m["gt_canon"])
            matched_gt_mu.append(m["gt_mu"])
        l_ep = l_ep / max(1, n_ep)

        l_curve = exist_logit.new_zeros(())
        l_curve_ep = exist_logit.new_zeros(())
        l_lat = exist_logit.new_zeros(())
        if matched_pred_lat:
            pred_lat = torch.cat(matched_pred_lat, dim=0)
            gt_canon = torch.cat(matched_gt_canon, dim=0)
            gt_mu = torch.cat(matched_gt_mu, dim=0)
            dec = decode_curve_latent(
                curve_vae, pred_lat, num_points=gt_canon.shape[1])
            l_curve = (dec - gt_canon).abs().mean()
            l_curve_ep = (
                dec[:, [0, -1]] - gt_canon[:, [0, -1]]).abs().mean()
            l_lat = (pred_lat - gt_mu).pow(2).mean()

        total = (
            self.edge_exist_w * l_exist
            + self.endpoint_w * l_ep
            + self.curve_w * l_curve
            + self.curve_endpoint_w * l_curve_ep
            + self.lat_reg_w * l_lat
        )
        return {
            "total": total,
            "edge_exist": l_exist.detach(),
            "endpoint": l_ep.detach(),
            "curve": l_curve.detach(),
            "curve_endpoint": l_curve_ep.detach(),
            "lat_reg": l_lat.detach(),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _match(
        self,
        preds: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        gt_mu: list[torch.Tensor],
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
        """Hungarian-match each shape's queries to its GT edges.

        Returns ``(match, exist_target)`` where ``match[s]`` carries the matched
        query ids ``qi`` and aligned GT endpoint / canonical / mu tensors, and
        ``exist_target (B, Ne)`` is the existence BCE target.
        """
        exist_logit = preds["edge_exist_logit"]
        endpoints = preds["endpoints"]
        curve_latent = preds["curve_latent"]
        b, ne = exist_logit.shape
        device = exist_logit.device
        exist_target = exist_logit.new_zeros(b, ne)
        match: list[dict[str, torch.Tensor]] = []

        for s in range(b):
            tgt = targets[s]
            gt_ep = tgt["endpoints"]
            ng = int(gt_ep.shape[0])
            if ng == 0:
                match.append({
                    "qi": torch.zeros(0, dtype=torch.long, device=device),
                    "gt_ep": gt_ep.new_zeros((0, 2, 3)),
                    "gt_canon": tgt["edge_curve"][:0],
                    "gt_mu": curve_latent.new_zeros(
                        (0, curve_latent.shape[-1])),
                })
                continue
            ep_cost = self._endpoint_cost(endpoints[s], gt_ep)      # (Ne, ng)
            lat_cost = torch.cdist(curve_latent[s], gt_mu[s], p=1)  # (Ne, ng)
            exist_reward = torch.sigmoid(exist_logit[s])[:, None]   # (Ne, 1)
            cost = (self.match_endpoint * ep_cost
                    + self.match_lat * lat_cost
                    - self.match_exist * exist_reward)
            qi, gi = self._hungarian(cost)
            exist_target[s, qi] = 1.0
            match.append({
                "qi": qi,
                "gt_ep": gt_ep[gi],
                "gt_canon": tgt["edge_curve"][gi],
                "gt_mu": gt_mu[s][gi],
            })
        return match, exist_target

    # ------------------------------------------------------------------
    def forward(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, Any],
        curve_vae: nn.Module,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = preds["edge_exist_logit"].device
        targets = build_targets(batch, device=device)
        # Stop-grad posterior means of each shape's GT canonical curves (for the
        # matching cost + latent regulariser only).
        gt_mu: list[torch.Tensor] = []
        with torch.no_grad():
            for tgt in targets:
                gt_mu.append(encode_curve_mu(curve_vae, tgt["edge_curve"]))

        match, exist_target = self._match(preds, targets, gt_mu)
        for s, m in enumerate(match):
            m["gt_mu"] = gt_mu[s][:0] if m["qi"].numel() == 0 else m["gt_mu"]

        final = self._layer_loss(preds, match, exist_target, curve_vae)
        total = final["total"]
        parts = {
            "edge_exist_loss": final["edge_exist"],
            "endpoint_loss": final["endpoint"],
            "curve_loss": final["curve"],
            "curve_endpoint_loss": final["curve_endpoint"],
            "lat_reg_loss": final["lat_reg"],
            "n_match_edges": torch.tensor(
                float(sum(int(m["qi"].numel()) for m in match)) / max(1, len(match)),
                device=device),
        }

        aux = preds.get("aux") or []
        if self.aux_w > 0.0 and aux:
            aux_total = total.new_zeros(())
            for layer in aux:
                aux_total = aux_total + self._layer_loss(
                    layer, match, exist_target, curve_vae)["total"]
            total = total + self.aux_w * aux_total
            parts["aux_loss"] = aux_total.detach()

        return total, parts


__all__ = ["EdgeSetCriterion", "sigmoid_focal_loss"]
