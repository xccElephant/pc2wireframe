"""Set-prediction criterion for the point-cloud -> wireframe model.

Vertices and edges are supervised differently:

  1. **Nodes (DETR set prediction)** -- match node queries to GT vertices with
     the Hungarian algorithm on an L1-coordinate cost (minus an existence
     bonus). Matched queries supervise coordinate (L1) and are the positives
     for the node-existence BCE. Deep supervision reuses this matching on every
     intermediate node layer.
  2. **Edges (Relationformer pairwise prediction)** -- there is no edge query
     set and no edge matching. With the node match fixed, each GT edge becomes a
     pair of *matched query* indices (a positive vertex pair). For each sample
     we build a small candidate set (all matched-GT positive pairs + random and
     kNN hard negatives, see ``edge_decoder.sample_pairs_train``) and score it
     with the relation edge head (``model.score_pairs``). The candidates are
     supervised with an **existence (alive) Focal BCE** (handling the heavy
     positive/negative imbalance) and, on the positive pairs only, a **curve
     L1** in the canonical curve frame (decoded through the frozen curve VAE).

Edge orientation is fixed by a deterministic coordinate rule in the dataset
(``A`` = lexicographically smaller endpoint), so the canonical GT curve is
already oriented start -> end = A -> B and no endpoint cross-entropy / curve
direction handling is needed here.

**Deep supervision** is applied to the node stack only: the node Hungarian
matching is run once on the final layer and reused for every ``aux_node`` layer
(cheaper + consistent targets), scaled by ``aux_weight``. The pairwise edge head
runs once on the final-layer node tokens / relation token.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from .edge_decoder import sample_pairs_train
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
    """Hungarian node matching + pairwise (Relationformer) edge losses."""

    def __init__(
        self,
        *,
        # ----- node matching / loss -----
        node_coord_weight: float = 5.0,
        node_exist_weight: float = 1.0,
        match_node_coord: float = 5.0,
        match_node_exist: float = 1.0,
        node_exist_pos_weight: float = 10.0,
        # ----- pairwise edge loss -----
        edge_exist_weight: float = 2.0,
        curve_weight: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        # ----- training candidate-pair sampling -----
        max_train_pairs: int = 2048,
        neg_ratio: float = 3.0,
        knn_k: int = 8,
        # ----- deep supervision -----
        aux_weight: float = 1.0,
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
        self.curve_w = curve_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.max_train_pairs = int(max_train_pairs)
        self.neg_ratio = float(neg_ratio)
        self.knn_k = int(knn_k)
        self.aux_weight = aux_weight
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

    # ------------------------------------------------------------------
    def _node_exist_bce(
        self, node_exist_logit: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Node-existence BCE with clamped positive up-weighting (shared)."""
        n_pos = target.sum()
        n_neg = target.numel() - n_pos
        pw = torch.clamp(
            n_neg / torch.clamp(n_pos, min=1.0), 1.0, self.node_exist_pos_weight)
        return F.binary_cross_entropy_with_logits(
            node_exist_logit, target, pos_weight=pw)

    def _node_layer_loss(
        self,
        coord_l: torch.Tensor,            # (B, Nq, 3)
        node_exist_logit_l: torch.Tensor,  # (B, Nq)
        match_info: list[dict | None],
        node_exist_target: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        """Weighted node loss for one (aux) layer, reusing the final matching."""
        cl = coord_l.new_zeros(())
        for s, info in enumerate(match_info):
            if info is None:
                continue
            cl = cl + F.l1_loss(
                coord_l[s, info["qi"]], info["gt_c"], reduction="sum")
        cl = cl / max(n_nodes, 1)
        nel = self._node_exist_bce(node_exist_logit_l, node_exist_target)
        return self.node_coord_w * cl + self.node_exist_w * nel

    # ------------------------------------------------------------------
    def forward(
        self,
        preds: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        model: nn.Module,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        coord = preds["coord"]
        node_exist_logit = preds["node_exist_logit"]
        node_tokens = preds["node_tokens"]
        rln_token = preds["rln_token"]
        b, nq, _ = coord.shape
        device = coord.device
        curve_vae = model.curve_vae

        node_exist_target = torch.zeros(b, nq, device=device)
        coord_loss = coord.new_zeros(())
        edge_exist_loss = coord.new_zeros(())
        curve_loss = coord.new_zeros(())
        n_nodes = 0
        n_pairs = 0
        n_pos_edges = 0

        # Per-sample node matching (computed once on the final-layer predictions)
        # is cached and reused for every deep-supervision aux node layer.
        match_info: list[dict | None] = [None] * b

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
            gt_c = gt_coords[gi]
            coord_loss = coord_loss + F.l1_loss(
                coord[s, qi], gt_c, reduction="sum")
            n_nodes += nv
            match_info[s] = {"qi": qi, "gt_c": gt_c}

            # candidate vertices for the pairwise edge head = matched queries.
            v = int(qi.shape[0])
            if v < 2:
                continue
            # query id -> local candidate index; GT vertex id -> matched query.
            q2local = torch.full((nq,), -1, dtype=torch.long, device=device)
            q2local[qi] = torch.arange(v, device=device)
            g2q = torch.full((nv,), -1, dtype=torch.long, device=device)
            g2q[gi] = qi

            # positive local pairs (one per valid GT edge), curve targets aligned.
            ne = tgt["edge_a"].shape[0]
            if ne > 0:
                ta = g2q[tgt["edge_a"]]
                tb = g2q[tgt["edge_b"]]
                valid = (ta >= 0) & (tb >= 0) & (ta != tb)
                ta, tb = ta[valid], tb[valid]
                gt_curves = tgt["edge_curve"][valid]
                pos_local = torch.stack([q2local[ta], q2local[tb]], dim=-1)
            else:
                pos_local = torch.zeros(0, 2, dtype=torch.long, device=device)
                gt_curves = tgt["edge_curve"][:0]

            cand_tokens = node_tokens[s, qi]              # (V, D) with grad
            cand_pos = coord[s, qi]                        # (V, 3) with grad
            pair_idx, alive_target = sample_pairs_train(
                pos_local, cand_pos.detach(),
                neg_ratio=self.neg_ratio,
                max_train_pairs=self.max_train_pairs,
                knn_k=self.knn_k,
            )
            if pair_idx.shape[0] == 0:
                continue

            res = model.score_pairs(
                cand_tokens, cand_pos, rln_token[s], pair_idx)
            edge_exist_loss = edge_exist_loss + sigmoid_focal_loss(
                res["alive_logit"], alive_target,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
                reduction="sum")
            n_pairs += int(pair_idx.shape[0])

            # curve L1 on the positive pairs (first rows; aligned to gt_curves).
            n_pos = min(int(pos_local.shape[0]), int(pair_idx.shape[0]))
            if n_pos > 0 and "curve_latent" in res:
                pred_curves = decode_curve_latent(
                    curve_vae, res["curve_latent"][:n_pos], self.num_curve_points)
                tgt_curves = gt_curves[:n_pos]
                curve_loss = curve_loss + (
                    F.l1_loss(pred_curves, tgt_curves, reduction="sum")
                    + 0.5 * F.l1_loss(
                        pred_curves[:, [0, -1]], tgt_curves[:, [0, -1]],
                        reduction="sum")
                )
                n_pos_edges += n_pos

        coord_loss = coord_loss / max(n_nodes, 1)
        edge_exist_loss = edge_exist_loss / max(n_pairs, 1)
        curve_loss = curve_loss / max(
            n_pos_edges * self.num_curve_points * 3, 1)

        # node existence BCE with (clamped) positive up-weighting.
        node_exist_loss = self._node_exist_bce(
            node_exist_logit, node_exist_target)

        total = (
            self.node_coord_w * coord_loss
            + self.node_exist_w * node_exist_loss
            + self.edge_exist_w * edge_exist_loss
            + self.curve_w * curve_loss
        )
        parts = {
            "coord_loss": coord_loss.detach(),
            "node_exist_loss": node_exist_loss.detach(),
            "edge_exist_loss": edge_exist_loss.detach(),
            "curve_loss": curve_loss.detach(),
            "n_match_edges": torch.tensor(float(n_pos_edges), device=device),
            "n_cand_pairs": torch.tensor(float(n_pairs), device=device),
        }

        # ---- deep supervision: node losses on each intermediate node layer,
        # reusing the final-layer node matching (cheaper + stable targets). ----
        aux_node = preds.get("aux_node") or []
        if self.aux_weight > 0.0 and aux_node:
            aux_total = coord.new_zeros(())
            for layer in aux_node:
                aux_total = aux_total + self._node_layer_loss(
                    layer["coord"], layer["node_exist_logit"],
                    match_info, node_exist_target, n_nodes)
            total = total + self.aux_weight * aux_total
            parts["aux_loss"] = aux_total.detach()

        return total, parts


__all__ = ["WireframeCriterion", "sigmoid_focal_loss"]
