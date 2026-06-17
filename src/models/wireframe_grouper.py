"""Stage-2 wireframe grouper: a point set ``(N, 4)`` -> structured per-point fields.

The fragile hand-written reconstruction (``src/recon/traditional.py``) is the
ceiling of the RF pipeline. This module is the learned replacement for it: a
permutation-equivariant point transformer that, instead of *guessing*
connectivity / ordering downstream, **regresses** the quantities that make the
final grouping trivial and stable (VoteNet / associative-embedding style):

    per point ->
      * vertex_score   (1)  -- vertex vs edge logit (refines the input type);
      * vertex_offset  (3)  -- offset to the host vertex centre (VoteNet vote);
      * endpoint_off   (2,3)-- offsets to the edge's two endpoint vertices
                               (double Hough vote -> connectivity);
      * embedding      (D)  -- instance embedding (same edge close, different
                               edge far) to split edges that share endpoints;
      * arclen         (1)  -- normalised arc-length position along the edge
                               (-> robust ordering, even for arcs / closed loops).

Decoding (see :mod:`src.recon.grouped`) is then pure book-keeping: cluster the
voted vertex centres, snap each edge point's two voted endpoints to vertices to
recover ``edge_index``, and sort each edge's points by ``arclen`` to recover the
curve -- no ``merge_radius`` / ``min_votes`` / chord-projection heuristics.

The backbone is the stock ``torch.nn.TransformerEncoder`` (no positional
encoding, so it is permutation equivariant; the self-attention dispatches to
``scaled_dot_product_attention`` Flash / mem-efficient kernels, keeping the
``N=8192`` self-attention at ``O(N)`` memory). The instance-embedding loss is
the standard discriminative loss of De Brabandere et al., 2017
("Semantic Instance Segmentation with a Discriminative Loss Function").
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    """A small 2-layer prediction head."""
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.GELU(),
        nn.Linear(d_hidden, d_out),
    )


class WireframeGrouper(nn.Module):
    """Point-set transformer producing per-point grouping fields.

    Args:
        point_dim: input channels per point (``4`` = xyz + type).
        d_model: transformer width.
        depth: number of ``TransformerEncoderLayer`` blocks.
        nhead: attention heads.
        mlp_ratio: feed-forward expansion.
        dropout: dropout in attention / feed-forward / heads.
        embed_dim: instance-embedding dimension ``D``.
    """

    def __init__(
        self,
        point_dim: int = 4,
        d_model: int = 256,
        depth: int = 6,
        nhead: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        embed_dim: int = 8,
    ) -> None:
        super().__init__()
        self.point_dim = int(point_dim)
        self.embed_dim = int(embed_dim)

        self.in_proj = nn.Linear(point_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)

        dh = d_model
        self.vertex_score_head = _mlp(d_model, dh, 1)
        self.vertex_offset_head = _mlp(d_model, dh, 3)
        self.endpoint_offset_head = _mlp(d_model, dh, 6)
        self.embedding_head = _mlp(d_model, dh, embed_dim)
        self.arclen_head = _mlp(d_model, dh, 1)

    def forward(self, pts: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the grouper.

        Args:
            pts: ``(B, N, point_dim)`` point set ``(xyz, type)``.

        Returns:
            dict of per-point fields::

                vertex_score    (B, N)
                vertex_offset   (B, N, 3)
                endpoint_offset (B, N, 2, 3)
                embedding       (B, N, D)
                arclen          (B, N)
        """
        b, n, _ = pts.shape
        h = self.in_proj(pts)
        h = self.encoder(h)
        h = self.norm(h)
        return {
            "vertex_score": self.vertex_score_head(h).squeeze(-1),
            "vertex_offset": self.vertex_offset_head(h),
            "endpoint_offset": self.endpoint_offset_head(h).reshape(b, n, 2, 3),
            "embedding": self.embedding_head(h),
            "arclen": self.arclen_head(h).squeeze(-1),
        }


# ----------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------
def discriminative_loss(
    emb: torch.Tensor,
    inst: torch.Tensor,
    *,
    delta_var: float = 0.5,
    delta_dist: float = 1.5,
    w_var: float = 1.0,
    w_dist: float = 1.0,
    w_reg: float = 0.001,
) -> torch.Tensor:
    """Discriminative instance-embedding loss (De Brabandere et al., 2017).

    Args:
        emb:  ``(M, D)`` embeddings of the points to group.
        inst: ``(M,)`` integer instance ids in ``[0, C)`` (contiguous).
    Returns:
        scalar loss (``var + dist + reg`` pulled / pushed terms).
    """
    device = emb.device
    if emb.shape[0] == 0:
        return emb.new_zeros(())
    c = int(inst.max().item()) + 1
    if c <= 0:
        return emb.new_zeros(())
    d = emb.shape[1]

    counts = torch.bincount(inst, minlength=c).clamp_min(1).to(emb.dtype)  # (C,)
    mu = torch.zeros(c, d, device=device, dtype=emb.dtype)
    mu.index_add_(0, inst, emb)
    mu = mu / counts[:, None]

    # variance (pull) term
    dist = torch.norm(emb - mu[inst], dim=1)
    var = torch.clamp(dist - delta_var, min=0.0) ** 2
    var_per_inst = torch.zeros(c, device=device, dtype=emb.dtype)
    var_per_inst.index_add_(0, inst, var)
    var_per_inst = var_per_inst / counts
    l_var = var_per_inst.mean()

    # distance (push) term
    if c > 1:
        diff = mu[:, None, :] - mu[None, :, :]
        dmat = torch.norm(diff, dim=2)
        margin = torch.clamp(2.0 * delta_dist - dmat, min=0.0) ** 2
        eye = torch.eye(c, device=device, dtype=torch.bool)
        margin = margin[~eye]
        l_dist = margin.mean()
    else:
        l_dist = emb.new_zeros(())

    l_reg = torch.norm(mu, dim=1).mean()
    return w_var * l_var + w_dist * l_dist + w_reg * l_reg


def grouper_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    w_score: float = 1.0,
    w_vertex: float = 1.0,
    w_endpoint: float = 1.0,
    w_arclen: float = 1.0,
    w_embed: float = 1.0,
    embed_kwargs: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Total stage-2 loss + components.

    Expects ``batch`` with the keys produced by ``collate_grouper_batch``
    (``wf_points``, ``lbl_is_vertex``, ``lbl_edge_id``, ``lbl_arclen``,
    ``lbl_endpoint_a/b``, ``lbl_vertex_target``).
    """
    embed_kwargs = embed_kwargs or {}
    pts = batch["wf_points"]
    xyz = pts[..., :3]
    is_v = batch["lbl_is_vertex"].bool()
    is_e = ~is_v

    # --- vertex / edge classification (all points) ---
    l_score = F.binary_cross_entropy_with_logits(
        out["vertex_score"], is_v.to(out["vertex_score"].dtype))

    # --- vertex-centre voting (vertex points only) ---
    pred_center = xyz + out["vertex_offset"]
    if is_v.any():
        l_vertex = F.smooth_l1_loss(
            pred_center[is_v], batch["lbl_vertex_target"][is_v])
    else:
        l_vertex = pts.new_zeros(())

    # --- endpoint voting (edge points only, order-invariant) ---
    if is_e.any():
        p1 = xyz + out["endpoint_offset"][..., 0, :]
        p2 = xyz + out["endpoint_offset"][..., 1, :]
        ea = batch["lbl_endpoint_a"]
        eb = batch["lbl_endpoint_b"]
        p1e, p2e = p1[is_e], p2[is_e]
        ea_e, eb_e = ea[is_e], eb[is_e]

        def _sl1(a, b):  # per-point summed smooth-L1 over xyz -> (M,)
            return F.smooth_l1_loss(a, b, reduction="none").sum(dim=-1)

        cost_ab = _sl1(p1e, ea_e) + _sl1(p2e, eb_e)
        cost_ba = _sl1(p1e, eb_e) + _sl1(p2e, ea_e)
        l_endpoint = torch.minimum(cost_ab, cost_ba).mean()

        l_arclen = F.mse_loss(out["arclen"][is_e], batch["lbl_arclen"][is_e])
    else:
        l_endpoint = pts.new_zeros(())
        l_arclen = pts.new_zeros(())

    # --- instance embedding (edge points, per sample, grouped by edge id) ---
    emb = out["embedding"]
    edge_id = batch["lbl_edge_id"]
    l_embed = pts.new_zeros(())
    nb = pts.shape[0]
    n_emb = 0
    for b in range(nb):
        m = is_e[b]
        if not torch.any(m):
            continue
        ids = edge_id[b][m]
        uniq, remapped = torch.unique(ids, return_inverse=True)
        if uniq.numel() < 1:
            continue
        l_embed = l_embed + discriminative_loss(
            emb[b][m], remapped, **embed_kwargs)
        n_emb += 1
    if n_emb > 0:
        l_embed = l_embed / n_emb

    total = (
        w_score * l_score
        + w_vertex * l_vertex
        + w_endpoint * l_endpoint
        + w_arclen * l_arclen
        + w_embed * l_embed
    )
    return {
        "loss": total,
        "loss_score": l_score.detach(),
        "loss_vertex": l_vertex.detach(),
        "loss_endpoint": l_endpoint.detach(),
        "loss_arclen": l_arclen.detach(),
        "loss_embed": l_embed.detach() if torch.is_tensor(l_embed) else l_embed,
    }


__all__ = ["WireframeGrouper", "discriminative_loss", "grouper_loss"]
