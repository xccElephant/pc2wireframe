"""Set-prediction criterion for the point-cloud -> wireframe decoder.

The decoder emits a fixed set of node queries and edge queries; the GT graph is
a variable-size set, so supervision is done in two Hungarian-matched stages
(global optimal assignment, not greedy):

  1. **Nodes** -- match node queries to GT vertices on an L1-coordinate cost
     (minus an existence bonus). Matched queries supervise coordinate (L1) and
     are the positives for the node-existence BCE.
  2. **Edges** -- with the node match fixed, each GT edge's two endpoints map to
     specific *query* indices. Match edge queries to GT edges on a cost built
     from edge existence + the two endpoint log-probabilities. To keep the
     assignment tractable the edge match is optionally restricted to a top-k
     candidate set (each GT edge proposes its ``k`` best-scoring edge queries;
     the Hungarian runs on the union, shrinking the cost matrix from
     ``Nq_edge x ne`` to ``|C| x ne``). Matched edge queries supervise: edge
     existence (**Focal BCE**, for the heavy positive/negative imbalance), the
     two endpoint distributions (cross-entropy over node queries) and the
     per-edge curve latent (**L1** in the canonical curve frame, decoded through
     the frozen curve VAE).

On top of the matched losses we add a **topology-aware** term (no matching
needed for the second half):

  * *soft-degree consistency* -- each query's expected incident-edge count
    ``sum_e sigma(exist_e) * (P_a[e,q] + P_b[e,q])`` is regressed (smooth-L1) to
    the GT degree of its matched vertex (0 for unmatched queries), penalising
    over-/under-connected nodes.
  * *dedup / self-loop* -- the soft directed adjacency
    ``M = (P_a * exist)^T @ P_b`` should describe a simple graph: any unordered
    pair carries at most one edge (penalise ``relu(M+M^T - 1)``) and there are
    no self-loops (penalise ``diag(M)``).

Endpoints follow the dataset's directed ``(start, end)`` convention so the
endpoint-A target and the canonical curve orientation stay consistent.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from .packing import decode_curve_latent


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary focal loss (RetinaNet) over logits / {0,1} targets."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce * (1.0 - p_t).pow(gamma)
    if alpha >= 0.0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class WireframeCriterion(nn.Module):
    """Hungarian node/edge matching + the matched-space wireframe losses."""

    def __init__(
        self,
        *,
        # ----- node matching / loss -----
        node_coord_weight: float = 5.0,
        node_exist_weight: float = 1.0,
        match_node_coord: float = 5.0,
        match_node_exist: float = 1.0,
        node_exist_pos_weight: float = 10.0,
        # ----- edge matching / loss -----
        edge_exist_weight: float = 2.0,
        endpoint_weight: float = 1.0,
        curve_weight: float = 2.0,
        match_edge_exist: float = 0.5,
        match_edge_topk: int = 0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        # ----- topology-aware loss -----
        topo_degree_weight: float = 0.0,
        topo_dedup_weight: float = 0.0,
        # ----- curve decode -----
        num_curve_points: int = 32,
    ) -> None:
        super().__init__()
        self.node_coord_w = node_coord_weight
        self.node_exist_w = node_exist_weight
        self.match_node_coord = match_node_coord
        self.match_node_exist = match_node_exist
        self.node_exist_pos_weight = node_exist_pos_weight
        self.edge_exist_w = edge_exist_weight
        self.endpoint_w = endpoint_weight
        self.curve_w = curve_weight
        self.match_edge_exist = match_edge_exist
        self.match_edge_topk = match_edge_topk
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.topo_degree_w = topo_degree_weight
        self.topo_dedup_w = topo_dedup_weight
        self.num_curve_points = num_curve_points

    # ------------------------------------------------------------------
    @staticmethod
    @torch.no_grad()
    def _match_nodes(
        coord: torch.Tensor, exist_logit: torch.Tensor, gt_coords: torch.Tensor,
        coord_w: float, exist_w: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hungarian match queries -> GT vertices. Returns ``(q_idx, g_idx)``."""
        nv = gt_coords.shape[0]
        if nv == 0:
            z = coord.new_zeros(0, dtype=torch.long)
            return z, z
        cost = coord_w * torch.cdist(coord, gt_coords)        # (Nq, nv)
        cost = cost - exist_w * torch.sigmoid(exist_logit).unsqueeze(1)
        qi, gi = linear_sum_assignment(cost.detach().cpu().numpy())
        dev = coord.device
        return (torch.as_tensor(qi, dtype=torch.long, device=dev),
                torch.as_tensor(gi, dtype=torch.long, device=dev))

    @staticmethod
    @torch.no_grad()
    def _match_edges(
        edge_exist_logit: torch.Tensor,   # (Ne,)
        log_pa: torch.Tensor,             # (Ne, Nq)
        log_pb: torch.Tensor,             # (Ne, Nq)
        ta: torch.Tensor,                 # (ne,) target endpoint-A query id
        tb: torch.Tensor,                 # (ne,) target endpoint-B query id
        exist_w: float,
        topk: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hungarian match edge queries -> GT edges. Returns ``(e_idx, g_idx)``.

        With ``topk > 0`` the assignment is restricted to a candidate set: each
        GT edge contributes its ``topk`` best-scoring edge queries (by endpoint
        log-prob) and the Hungarian runs on the union ``C`` only, shrinking the
        cost matrix from ``Ne x ne`` to ``|C| x ne``. The candidate set is padded
        (by global best endpoint score) up to ``min(ne, Ne)`` so every GT edge
        can still receive a distinct query.
        """
        ne = ta.shape[0]
        if ne == 0:
            z = edge_exist_logit.new_zeros(0, dtype=torch.long)
            return z, z
        # endpoint log-prob of each (query, gt-edge) pair, directed (A=start).
        ep = log_pa[:, ta] + log_pb[:, tb]                    # (Ne, ne)
        cost = -ep - exist_w * F.logsigmoid(edge_exist_logit).unsqueeze(1)
        dev = edge_exist_logit.device
        n_q = ep.shape[0]

        if topk and 0 < topk < n_q:
            k = min(topk, n_q)
            sel = torch.zeros(n_q, dtype=torch.bool, device=dev)
            sel[torch.topk(ep, k, dim=0).indices.reshape(-1)] = True
            need = min(ne, n_q)
            if int(sel.sum()) < need:
                order = torch.argsort(ep.max(dim=1).values, descending=True)
                rest = order[~sel[order]]
                sel[rest[: need - int(sel.sum())]] = True
            cand = torch.nonzero(sel, as_tuple=False).squeeze(1)   # (|C|,)
            sub = cost[cand].detach().cpu().numpy()
            ei_sub, gj = linear_sum_assignment(sub)
            ei = cand[torch.as_tensor(ei_sub, dtype=torch.long, device=dev)]
            return ei, torch.as_tensor(gj, dtype=torch.long, device=dev)

        ei, gj = linear_sum_assignment(cost.detach().cpu().numpy())
        return (torch.as_tensor(ei, dtype=torch.long, device=dev),
                torch.as_tensor(gj, dtype=torch.long, device=dev))

    # ------------------------------------------------------------------
    @staticmethod
    def _topology_losses(
        edge_exist_logit: torch.Tensor,   # (Ne,)
        log_pa: torch.Tensor,             # (Ne, Nq)
        log_pb: torch.Tensor,             # (Ne, Nq)
        edge_a: torch.Tensor,             # (ne_gt,) GT endpoint-A node id
        edge_b: torch.Tensor,             # (ne_gt,) GT endpoint-B node id
        qi: torch.Tensor,                 # (nm,) matched query ids
        gi: torch.Tensor,                 # (nm,) matched GT vertex ids
        nv: int,
        nq: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable topology losses for one sample.

        Returns ``(degree_loss, dedup_loss)``:
          * ``degree_loss`` -- smooth-L1 between the soft per-query degree and
            the GT degree of its matched vertex (0 if unmatched), summed over the
            ``nq`` queries.
          * ``dedup_loss`` -- soft simple-graph violation (self-loops +
            multi-edges), normalised by the GT edge count.
        """
        device = edge_exist_logit.device
        exist = torch.sigmoid(edge_exist_logit)               # (Ne,)
        pa = log_pa.exp()                                     # (Ne, Nq)
        pb = log_pb.exp()                                     # (Ne, Nq)

        # ---- soft-degree consistency ----
        exp_deg = (exist.unsqueeze(1) * (pa + pb)).sum(0)     # (Nq,)
        gt_degree = torch.zeros(nv, device=device)
        if edge_a.numel() > 0:
            ones = torch.ones(edge_a.shape[0], device=device)
            gt_degree.scatter_add_(0, edge_a, ones)
            gt_degree.scatter_add_(0, edge_b, ones)
        target_deg = torch.zeros(nq, device=device)
        target_deg[qi] = gt_degree[gi]
        degree_loss = F.smooth_l1_loss(exp_deg, target_deg, reduction="sum")

        # ---- dedup / self-loop (soft directed adjacency) ----
        adj = (pa * exist.unsqueeze(1)).transpose(0, 1) @ pb  # (Nq, Nq)
        self_loop = torch.diagonal(adj).sum()
        undirected = adj + adj.transpose(0, 1)
        undirected = undirected - torch.diag_embed(torch.diagonal(undirected))
        multi_edge = 0.5 * torch.relu(undirected - 1.0).sum()
        ne_gt = max(int(edge_a.shape[0]), 1)
        dedup_loss = (self_loop + multi_edge) / ne_gt
        return degree_loss, dedup_loss

    # ------------------------------------------------------------------
    def forward(
        self,
        preds: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        curve_vae: nn.Module,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        coord = preds["coord"]
        node_exist_logit = preds["node_exist_logit"]
        edge_exist_logit = preds["edge_exist_logit"]
        ep_a_logits = preds["ep_a_logits"]
        ep_b_logits = preds["ep_b_logits"]
        curve_latent = preds["curve_latent"]
        b, nq, _ = coord.shape
        ne_q = edge_exist_logit.shape[1]
        device = coord.device

        node_exist_target = torch.zeros(b, nq, device=device)
        edge_exist_target = torch.zeros(b, ne_q, device=device)

        coord_loss = coord.new_zeros(())
        endpoint_loss = coord.new_zeros(())
        curve_loss = coord.new_zeros(())
        degree_loss = coord.new_zeros(())
        dedup_loss = coord.new_zeros(())
        n_nodes = 0
        n_edges = 0
        n_deg_q = 0
        n_topo = 0

        need_topo = self.topo_degree_w > 0.0 or self.topo_dedup_w > 0.0
        log_pa_all = F.log_softmax(ep_a_logits, dim=-1)
        log_pb_all = F.log_softmax(ep_b_logits, dim=-1)

        for s in range(b):
            tgt = targets[s]
            gt_coords = tgt["node_coords"]
            nv = gt_coords.shape[0]
            if nv == 0:
                continue

            qi, gi = self._match_nodes(
                coord[s], node_exist_logit[s], gt_coords,
                self.match_node_coord, self.match_node_exist)
            node_exist_target[s, qi] = 1.0
            coord_loss = coord_loss + F.l1_loss(
                coord[s, qi], gt_coords[gi], reduction="sum")
            n_nodes += nv

            # GT-node-id -> matched query id
            g2q = torch.full((nv,), -1, dtype=torch.long, device=device)
            g2q[gi] = qi

            # ---- topology-aware loss (needs only preds + node match) ----
            if need_topo:
                d_loss, dd_loss = self._topology_losses(
                    edge_exist_logit[s], log_pa_all[s], log_pb_all[s],
                    tgt["edge_a"], tgt["edge_b"], qi, gi, nv, nq)
                degree_loss = degree_loss + d_loss
                dedup_loss = dedup_loss + dd_loss
                n_deg_q += nq
                n_topo += 1

            ne = tgt["edge_a"].shape[0]
            if ne == 0:
                continue
            ta = g2q[tgt["edge_a"]]
            tb = g2q[tgt["edge_b"]]
            valid = (ta >= 0) & (tb >= 0) & (ta != tb)
            if not bool(valid.any()):
                continue
            ta, tb = ta[valid], tb[valid]
            gt_curves = tgt["edge_curve"][valid]

            ei, gj = self._match_edges(
                edge_exist_logit[s], log_pa_all[s], log_pb_all[s],
                ta, tb, self.match_edge_exist, self.match_edge_topk)
            edge_exist_target[s, ei] = 1.0

            # endpoint cross-entropy on matched edge queries (directed A=start).
            endpoint_loss = endpoint_loss + (
                F.cross_entropy(ep_a_logits[s, ei], ta[gj], reduction="sum")
                + F.cross_entropy(ep_b_logits[s, ei], tb[gj], reduction="sum")
            )

            # curve L1 in the canonical frame (frozen curve VAE decode).
            pred_curves = decode_curve_latent(
                curve_vae, curve_latent[s, ei], self.num_curve_points)
            tgt_curves = gt_curves[gj]
            curve_loss = curve_loss + (
                F.l1_loss(pred_curves, tgt_curves, reduction="sum")
                + 0.5 * F.l1_loss(
                    pred_curves[:, [0, -1]], tgt_curves[:, [0, -1]],
                    reduction="sum")
            )
            n_edges += ei.shape[0]

        coord_loss = coord_loss / max(n_nodes, 1)
        endpoint_loss = endpoint_loss / max(2 * n_edges, 1)
        curve_loss = curve_loss / max(
            n_edges * self.num_curve_points * 3, 1)
        degree_loss = degree_loss / max(n_deg_q, 1)
        dedup_loss = dedup_loss / max(n_topo, 1)

        # node existence BCE with (clamped) positive up-weighting.
        n_pos = node_exist_target.sum()
        n_neg = node_exist_target.numel() - n_pos
        pw = torch.clamp(
            n_neg / torch.clamp(n_pos, min=1.0), 1.0, self.node_exist_pos_weight)
        node_exist_loss = F.binary_cross_entropy_with_logits(
            node_exist_logit, node_exist_target, pos_weight=pw)

        # edge existence focal BCE (heavy negative imbalance).
        edge_exist_loss = sigmoid_focal_loss(
            edge_exist_logit, edge_exist_target,
            alpha=self.focal_alpha, gamma=self.focal_gamma)

        total = (
            self.node_coord_w * coord_loss
            + self.node_exist_w * node_exist_loss
            + self.edge_exist_w * edge_exist_loss
            + self.endpoint_w * endpoint_loss
            + self.curve_w * curve_loss
            + self.topo_degree_w * degree_loss
            + self.topo_dedup_w * dedup_loss
        )
        parts = {
            "coord_loss": coord_loss.detach(),
            "node_exist_loss": node_exist_loss.detach(),
            "edge_exist_loss": edge_exist_loss.detach(),
            "endpoint_loss": endpoint_loss.detach(),
            "curve_loss": curve_loss.detach(),
            "topo_degree_loss": degree_loss.detach(),
            "topo_dedup_loss": dedup_loss.detach(),
            "n_match_edges": torch.tensor(float(n_edges), device=device),
        }
        return total, parts


__all__ = ["WireframeCriterion", "sigmoid_focal_loss"]
