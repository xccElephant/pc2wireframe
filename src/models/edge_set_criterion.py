"""Edge-set criterion: order-invariant Hungarian edge matching + losses.

The edge-centric counterpart of a DETR set criterion. The decoder predicts a
fixed set of ``Q`` edge queries -- each an existence logit + ``P`` ordered
world-space sample points (``pts[0]`` / ``pts[-1]`` are the endpoints). This
module matches those queries to the GT edges and supervises them:

  * **matching** -- order-invariant Hungarian assignment of queries to GT edges
    on an *endpoint-only* cost (cheap + stable):

        cost = w_geo * min( L1(v1,ga)+L1(v2,gb), L1(v1,gb)+L1(v2,ga) )
               - w_exist * sigmoid(exist)

    The cheaper of the two endpoint pairings (forward / reversed) also fixes the
    GT point order used by the per-point losses.

  * **losses** -- only matched queries get the geometric terms:

      - ``exist``    focal BCE over all ``Q`` queries (matched = positive);
      - ``points``   ordered per-point L1 of the matched query's ``P`` points to
        the GT curve, taking the min over the two point orderings (the root of
        "points do not drift" -- far stronger than a chamfer);
      - ``endpoint`` an extra, higher-weight L1 on ``pts[0]`` / ``pts[-1]`` to
        the GT endpoints. GT edges sharing a vertex have *identical* endpoint
        coordinates, so this pulls the endpoints that should coincide onto the
        same point (the basis for the union-find merge / topology accuracy);
      - ``smooth``   small second-difference penalty (anti-jitter);
      - ``seglen``   small segment-length-variance penalty (even spacing);
      - ``consistency`` groups matched endpoints by their GT vertex id and
        penalises the intra-group variance, so endpoints that should share a
        vertex coincide (makes the union-find merge work / kills floating edges).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


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


def _resample_curve(points: torch.Tensor, num: int) -> torch.Tensor:
    """Resample ``(E, U, 3)`` ordered polylines to ``(E, num, 3)`` (linear).

    A no-op when ``U == num`` (the configured GT resolution already matches the
    decoder's ``sample_points_num``); otherwise interpolates by point index so
    the per-point loss can be computed without ever rewriting the dataset.
    """
    e, u, _ = points.shape
    if u == num:
        return points
    if u <= 1:
        return points[:, :1, :].expand(e, num, 3).contiguous()
    x = points.permute(0, 2, 1)                       # (E, 3, U)
    x = F.interpolate(x, size=num, mode="linear", align_corners=True)
    return x.permute(0, 2, 1).contiguous()            # (E, num, 3)


class EdgeSetCriterion(nn.Module):
    """Hungarian edge-set matching + existence / point / smoothness losses."""

    def __init__(
        self,
        *,
        w_exist: float = 2.0,
        w_points: float = 5.0,
        w_endpoint: float = 8.0,
        w_smooth: float = 0.5,
        w_seglen: float = 0.1,
        w_consistency: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.5,
        match_w_geo: float = 1.0,
        match_w_exist: float = 0.5,
    ) -> None:
        super().__init__()
        self.w_exist = float(w_exist)
        self.w_points = float(w_points)
        self.w_endpoint = float(w_endpoint)
        self.w_smooth = float(w_smooth)
        self.w_seglen = float(w_seglen)
        self.w_consistency = float(w_consistency)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.match_w_geo = float(match_w_geo)
        self.match_w_exist = float(match_w_exist)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _match(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
        ga: torch.Tensor,
        gb: torch.Tensor,
        exist_prob: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hungarian match queries -> GT edges on the endpoint-only cost.

        Returns ``(row, col)`` -- query ids matched to GT-edge ids
        (length ``min(Q, E)``).
        """
        from scipy.optimize import linear_sum_assignment

        d_v1a = torch.cdist(v1, ga, p=1)                   # (Q, E)
        d_v2b = torch.cdist(v2, gb, p=1)
        d_v1b = torch.cdist(v1, gb, p=1)
        d_v2a = torch.cdist(v2, ga, p=1)
        geo = torch.minimum(d_v1a + d_v2b, d_v1b + d_v2a)  # (Q, E)
        cost = self.match_w_geo * geo - self.match_w_exist * exist_prob[:, None]
        cost = cost.detach().cpu().numpy()
        row, col = linear_sum_assignment(cost)
        device = v1.device
        return (
            torch.as_tensor(row, dtype=torch.long, device=device),
            torch.as_tensor(col, dtype=torch.long, device=device),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        edge_exist_logit: torch.Tensor,        # (B, Q)
        edge_points: torch.Tensor,             # (B, Q, P, 3)
        gt_wireframes: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        device = edge_exist_logit.device
        b, q, p, _ = edge_points.shape

        l_exist = edge_points.new_zeros(())
        l_points = edge_points.new_zeros(())
        l_endpoint = edge_points.new_zeros(())
        l_smooth = edge_points.new_zeros(())
        l_seglen = edge_points.new_zeros(())
        l_consistency = edge_points.new_zeros(())
        n_exist = n_geom = n_cons = 0
        n_matched_total = 0

        for i in range(b):
            g = gt_wireframes[i]
            gep_raw = g["edge_points"].to(device).float()
            if gep_raw.numel():
                gep = gep_raw.reshape(gep_raw.shape[0], -1, 3)
                if gep.shape[1] != p:
                    gep = _resample_curve(gep, p)
            else:
                gep = edge_points.new_zeros((0, p, 3))
            e = gep.shape[0]

            exist_target = edge_exist_logit.new_zeros(q)
            if e == 0:
                l_exist = l_exist + _focal_bce(
                    edge_exist_logit[i], exist_target,
                    self.focal_gamma, self.focal_alpha)
                n_exist += 1
                continue

            pts = edge_points[i]                          # (Q, P, 3)
            v1, v2 = pts[:, 0], pts[:, -1]                # (Q, 3)
            ga, gb = gep[:, 0], gep[:, -1]                # (E, 3)
            row, col = self._match(
                v1, v2, ga, gb, torch.sigmoid(edge_exist_logit[i]))
            exist_target[row] = 1.0
            l_exist = l_exist + _focal_bce(
                edge_exist_logit[i], exist_target,
                self.focal_gamma, self.focal_alpha)
            n_exist += 1

            m = row.shape[0]
            n_matched_total += m
            pred_m = pts[row]                             # (M, P, 3)
            gt_m = gep[col]                               # (M, P, 3)
            gt_rev = torch.flip(gt_m, dims=[1])           # reversed point order

            # Per-edge ordered L1 in both orderings; the cheaper one is the
            # aligned GT used for every per-point / endpoint term.
            l1_fwd = (pred_m - gt_m).abs().mean(dim=(1, 2))   # (M,)
            l1_rev = (pred_m - gt_rev).abs().mean(dim=(1, 2))
            use_rev_1d = l1_rev < l1_fwd                       # (M,)
            use_rev = use_rev_1d[:, None, None]
            gt_aligned = torch.where(use_rev, gt_rev, gt_m)   # (M, P, 3)

            l_points = l_points + (pred_m - gt_aligned).abs().mean()

            pred_ep = torch.stack([pred_m[:, 0], pred_m[:, -1]], dim=1)
            gt_ep = torch.stack([gt_aligned[:, 0], gt_aligned[:, -1]], dim=1)
            l_endpoint = l_endpoint + (pred_ep - gt_ep).abs().mean()

            if p >= 3:
                second = pred_m[:, 2:] - 2.0 * pred_m[:, 1:-1] + pred_m[:, :-2]
                l_smooth = l_smooth + (second ** 2).sum(dim=-1).mean()
            if p >= 3:
                seg = (pred_m[:, 1:] - pred_m[:, :-1]).norm(dim=-1)   # (M, P-1)
                l_seglen = l_seglen + seg.var(dim=-1).mean()
            n_geom += 1

            # ---- vertex consistency: predicted endpoints that map to the SAME
            # GT vertex should coincide. Group matched endpoints by their GT
            # vertex id and penalise the intra-group variance (the "share a
            # vertex -> connected wireframe" pressure that makes union-find work
            # and kills floating edges).
            if self.w_consistency > 0.0:
                gei = g["edge_index"].to(device).long().reshape(-1, 2)
                if m > 0 and gei.shape[0] > int(col.max().item()):
                    u = gei[col, 0]                            # (M,) GT vid of ga
                    v = gei[col, 1]                            # (M,) GT vid of gb
                    # pts[:,0]=v1 aligns with ga unless reversed; then with gb.
                    vid0 = torch.where(use_rev_1d, v, u)       # vid of pred ep0
                    vid1 = torch.where(use_rev_1d, u, v)       # vid of pred ep1
                    coords = torch.cat([pred_m[:, 0], pred_m[:, -1]], dim=0)
                    vids = torch.cat([vid0, vid1], dim=0)      # (2M,)
                    uniq, inv = torch.unique(vids, return_inverse=True)
                    g_cnt = torch.zeros(
                        uniq.shape[0], device=device).index_add_(
                        0, inv, torch.ones_like(inv, dtype=coords.dtype))
                    g_sum = torch.zeros(
                        uniq.shape[0], 3, device=device).index_add_(
                        0, inv, coords)
                    g_mean = g_sum / g_cnt[:, None].clamp_min(1.0)
                    dev = coords - g_mean[inv]                 # (2M, 3)
                    shared = g_cnt[inv] >= 2                   # only multi-edge v
                    if shared.any():
                        l_consistency = l_consistency + (
                            dev[shared] ** 2).sum(dim=-1).mean()
                        n_cons += 1

        if n_exist > 0:
            l_exist = l_exist / n_exist
        if n_geom > 0:
            l_points = l_points / n_geom
            l_endpoint = l_endpoint / n_geom
            l_smooth = l_smooth / n_geom
            l_seglen = l_seglen / n_geom
        if n_cons > 0:
            l_consistency = l_consistency / n_cons

        total = (
            self.w_exist * l_exist
            + self.w_points * l_points
            + self.w_endpoint * l_endpoint
            + self.w_smooth * l_smooth
            + self.w_seglen * l_seglen
            + self.w_consistency * l_consistency
        )
        avg_matched = float(n_matched_total) / max(1, b)
        return {
            "loss_geom": total,
            "loss_exist": l_exist.detach(),
            "loss_points": l_points.detach(),
            "loss_endpoint": l_endpoint.detach(),
            "loss_smooth": l_smooth.detach(),
            "loss_seglen": l_seglen.detach(),
            "loss_consistency": l_consistency.detach(),
            "matched_edges": edge_exist_logit.new_tensor(avg_matched),
        }


__all__ = ["EdgeSetCriterion"]
