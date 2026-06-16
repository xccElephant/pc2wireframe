"""Relationformer-style pairwise edge head + candidate-pair samplers.

The vertex side stays a DETR decoder (node queries + iterative refinement +
Hungarian matching, see ``wireframe_decoder.py``); edges are no longer a
separate query set. Instead, given the decoded node tokens ``o``, their
predicted coordinates ``p`` and a *global* relation token ``r`` (the learnable
``[rln]`` token carried through the node self-attention), each candidate vertex
pair ``(i, j)`` is scored **directly**:

    pair_repr = [ o_i + o_j , |o_i - o_j| , r , p_i , p_j , p_i - p_j ]
              -> MLP (LayerNorm) -> { alive_logit , curve_latent }

This is the Relationformer (ECCV 2022) relation head: the symmetric
``o_i + o_j`` / ``|o_i - o_j|`` terms make the representation order-invariant
(edges are scored undirected), the global ``r`` token injects the graph-level
context that a plain ``MLP([o_i, o_j])`` baseline lacks, and the geometric
terms ground the score in the predicted positions. The endpoints of an edge are
simply the two vertex indices of the pair -- there is no pointer / endpoint
vocabulary and no edge Hungarian matching.

Because the head is symmetric, edge **orientation** (which endpoint is the
curve's start) is fixed by a deterministic coordinate rule applied downstream
(see ``dataset.py`` / ``pc2wireframe.reconstruct``), not learned here.

Two pair samplers mirror the verifier's needs:

  * :func:`sample_pairs_train` -- training candidates = all matched GT positive
    pairs plus ``neg_ratio`` x negatives (half random, half kNN hard negatives),
    capped at ``max_train_pairs``. Positives are always kept first so curve
    targets stay aligned.
  * :func:`shortlist_pairs_infer` -- inference candidates: all pairs when
    ``V*(V-1)/2 <= max_pairs``, otherwise the union of each vertex's ``knn_k``
    nearest neighbours.
"""
from __future__ import annotations

import torch
from torch import nn


def _pair_features(
    node_tokens: torch.Tensor,   # (V, D)
    node_pos: torch.Tensor,      # (V, 3)
    rln: torch.Tensor,           # (D,)
    pair_idx: torch.Tensor,      # (P, 2) long, indices into [0, V)
) -> torch.Tensor:
    """Gather the symmetric Relationformer pair representation ``(P, 3D + 9)``."""
    ia = pair_idx[:, 0]
    ib = pair_idx[:, 1]
    oa = node_tokens[ia]                       # (P, D)
    ob = node_tokens[ib]
    pa = node_pos[ia]                          # (P, 3)
    pb = node_pos[ib]
    sym = oa + ob                              # order-invariant
    asym = (oa - ob).abs()                     # order-invariant
    r = rln.unsqueeze(0).expand(pair_idx.shape[0], -1)
    return torch.cat([sym, asym, r, pa, pb, pa - pb], dim=-1)


class RelationEdgeHead(nn.Module):
    """MLP over the symmetric pair representation -> ``alive`` + ``curve``.

    Args:
        d_model: node-token / rln-token channel size.
        curve_latent_dim: per-edge curve-VAE latent size (0 disables the curve
            head, e.g. for ablations).
        hidden: trunk hidden width.
        num_layers: number of trunk linear layers (each ``Linear -> LayerNorm
            -> GELU -> Dropout``).
        dropout: dropout in the trunk.
    """

    def __init__(
        self,
        d_model: int,
        curve_latent_dim: int = 0,
        *,
        hidden: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.curve_latent_dim = int(curve_latent_dim)
        in_dim = 3 * self.d_model + 3 * 3
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(max(1, num_layers) - 1):
            layers += [
                nn.Linear(d, hidden), nn.LayerNorm(hidden),
                nn.GELU(), nn.Dropout(dropout),
            ]
            d = hidden
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.alive_head = nn.Linear(d, 1)
        self.curve_head = (
            nn.Linear(d, self.curve_latent_dim)
            if self.curve_latent_dim > 0 else None
        )

    def forward(
        self,
        node_tokens: torch.Tensor,   # (V, D)
        node_pos: torch.Tensor,      # (V, 3)
        rln: torch.Tensor,           # (D,) or (R, D)
        pair_idx: torch.Tensor,      # (P, 2)
    ) -> dict[str, torch.Tensor]:
        """Score one sample's candidate pairs.

        Returns ``{"alive_logit": (P,), "curve_latent": (P, Dc)}`` (the curve
        entry is omitted when the curve head is disabled).
        """
        if rln.dim() == 2:                     # mean-pool multiple rln tokens
            rln = rln.mean(0)
        if pair_idx.numel() == 0:
            out = {"alive_logit": node_tokens.new_zeros(0)}
            if self.curve_head is not None:
                out["curve_latent"] = node_tokens.new_zeros(
                    0, self.curve_latent_dim)
            return out
        feats = _pair_features(node_tokens, node_pos, rln, pair_idx)
        h = self.trunk(feats)
        out = {"alive_logit": self.alive_head(h).squeeze(-1)}
        if self.curve_head is not None:
            out["curve_latent"] = self.curve_head(h)
        return out


# ----------------------------------------------------------------------
# Pair sampling (training)
# ----------------------------------------------------------------------
def sample_pairs_train(
    pos_pairs: torch.Tensor,         # (n_pos, 2) local vertex-index positives
    vertex_pos: torch.Tensor,        # (V, 3) predicted candidate-vertex coords
    *,
    neg_ratio: float,
    max_train_pairs: int,
    knn_k: int = 8,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(pair_idx, alive_target)`` for one training sample.

    All vertex indices are LOCAL ``[0, V)`` into ``vertex_pos`` (the candidate
    vertex set, i.e. the matched node queries). Strategy:

        * keep every matched GT pair as a positive (``alive = 1``), first;
        * draw ``round(neg_ratio * n_pos)`` negatives: half random pairs, half
          kNN hard negatives (closest non-edge pairs in 3D);
        * cap the total at ``max_train_pairs`` (positives are always kept).

    The returned positives keep the input ``pos_pairs`` order so curve targets
    stay aligned. Pairs are stored as ``(lo, hi)`` (undirected); orientation is
    resolved later by the deterministic coordinate rule.
    """
    device = vertex_pos.device
    v = int(vertex_pos.shape[0])
    if v < 2:
        return (torch.zeros(0, 2, dtype=torch.long, device=device),
                torch.zeros(0, device=device))

    pos = pos_pairs.long()
    n_pos = int(pos.shape[0])
    n_all = v * (v - 1) // 2
    if n_pos == 0:
        n_neg = min(max(max_train_pairs, 0), n_all)
    else:
        n_neg = min(int(round(neg_ratio * n_pos)),
                    max(max_train_pairs - n_pos, 0),
                    n_all - n_pos)
    n_neg = max(n_neg, 0)

    # set of positive undirected keys, to exclude from negatives.
    pos_set: set[int] = set()
    if n_pos > 0:
        lo = torch.minimum(pos[:, 0], pos[:, 1])
        hi = torch.maximum(pos[:, 0], pos[:, 1])
        pos_set = set((lo * v + hi).tolist())

    n_rand = (n_neg + 1) // 2
    n_hard = n_neg - n_rand

    rand_pairs: list[tuple[int, int]] = []
    attempts = 0
    while len(rand_pairs) < n_rand and attempts < 8 * max(n_rand, 1):
        ia = torch.randint(0, v, (max(n_rand, 1) * 2,),
                           generator=rng, device=device)
        ib = torch.randint(0, v, (max(n_rand, 1) * 2,),
                           generator=rng, device=device)
        for x, y in zip(ia.tolist(), ib.tolist()):
            if x == y:
                continue
            a, b = (x, y) if x < y else (y, x)
            if (a * v + b) in pos_set:
                continue
            rand_pairs.append((a, b))
            if len(rand_pairs) >= n_rand:
                break
        attempts += 1

    hard_pairs: list[tuple[int, int]] = []
    if n_hard > 0:
        d2 = torch.cdist(vertex_pos.float(), vertex_pos.float())
        d2.fill_diagonal_(float("inf"))
        k = min(max(knn_k, 1), v - 1)
        knn = torch.topk(d2, k=k, dim=-1, largest=False).indices
        for i in range(v):
            for j in knn[i].tolist():
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a * v + b) in pos_set:
                    continue
                hard_pairs.append((a, b))
        if len(hard_pairs) > n_hard:
            idx = torch.randperm(
                len(hard_pairs), generator=rng, device=device)[:n_hard].tolist()
            hard_pairs = [hard_pairs[i] for i in idx]

    pair_list: list[tuple[int, int]] = []
    if n_pos > 0:
        plo = torch.minimum(pos[:, 0], pos[:, 1]).tolist()
        phi = torch.maximum(pos[:, 0], pos[:, 1]).tolist()
        pair_list += list(zip(plo, phi))
    pair_list += rand_pairs + hard_pairs
    if len(pair_list) > max_train_pairs:
        pair_list = pair_list[:max_train_pairs]

    if not pair_list:
        return (torch.zeros(0, 2, dtype=torch.long, device=device),
                torch.zeros(0, device=device))
    pair_idx = torch.tensor(pair_list, dtype=torch.long, device=device)
    alive = torch.zeros(pair_idx.shape[0], device=device)
    alive[:n_pos] = 1.0
    return pair_idx, alive


# ----------------------------------------------------------------------
# Pair sampling (inference)
# ----------------------------------------------------------------------
def shortlist_pairs_infer(
    vertex_pos: torch.Tensor,        # (V, 3)
    *,
    max_pairs: int,
    knn_k: int = 16,
) -> torch.Tensor:
    """Return ``(P, 2)`` candidate pair indices (LOCAL, ``(lo, hi)``).

    For ``V*(V-1)/2 <= max_pairs`` every pair is enumerated; otherwise the
    union of each vertex's ``knn_k`` nearest neighbours is used -- a cheap
    geometric prior on plausible edges. The result is capped at ``max_pairs``.
    """
    device = vertex_pos.device
    v = int(vertex_pos.shape[0])
    if v < 2:
        return torch.zeros(0, 2, dtype=torch.long, device=device)
    n_all = v * (v - 1) // 2
    if n_all <= max_pairs:
        ii, jj = torch.triu_indices(v, v, offset=1, device=device)
        return torch.stack([ii, jj], dim=-1)
    d2 = torch.cdist(vertex_pos.float(), vertex_pos.float())
    d2.fill_diagonal_(float("inf"))
    k = min(max(knn_k, 1), v - 1)
    knn = torch.topk(d2, k=k, dim=-1, largest=False).indices
    ii = torch.arange(v, device=device).unsqueeze(-1).expand(-1, k)
    pairs = torch.stack(
        [torch.minimum(ii, knn).reshape(-1),
         torch.maximum(ii, knn).reshape(-1)],
        dim=-1,
    )
    pairs = torch.unique(pairs, dim=0)
    if pairs.shape[0] > max_pairs:
        idx = torch.randperm(pairs.shape[0], device=device)[:max_pairs]
        pairs = pairs[idx]
    return pairs


__all__ = [
    "RelationEdgeHead",
    "sample_pairs_train",
    "shortlist_pairs_infer",
]
