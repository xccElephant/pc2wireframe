"""Stage-2 wireframe grouper: an anchor set ``(N, 3)`` -> structured per-point fields.

The fragile hand-written reconstruction (``src/recon/traditional.py``) is the
ceiling of the RF pipeline. This module is the learned replacement for it: a
permutation-equivariant point transformer that, instead of *guessing*
connectivity / ordering / curve type downstream, **regresses** the quantities
that make the final grouping trivial and stable (VoteNet / associative-embedding
style). There is no "vertex point" concept any more -- every point is an edge
point and vertices are recovered by endpoint voting:

    per point ->
      * endpoint_off   (2,3)-- offsets to the edge's two endpoint vertices
                               (double Hough vote -> connectivity + vertices);
      * embedding      (D)  -- instance embedding (same edge close, different
                               edge far) to split edges that share endpoints;
      * curve_type     (3)  -- per-point logits for line / arc / bezier;
      * anchor         (2,3)-- the edge's t=1/3 and t=2/3 coordinates (with the
                               two endpoints these parameterise the curve);
      * arclen         (1)  -- normalised arc-length position along the edge
                               (auxiliary ordering signal; supervised but not
                               required by the geometric decoder).

Decoding (see :mod:`src.recon.grouped`) is then pure book-keeping: cluster the
voted endpoints into vertices, snap each edge point's two voted endpoints to
those vertices to recover ``edge_index``, split same-endpoint edges by
``embedding``, then parse each edge's ``(a, q1, q2, b)`` by its curve type.

The backbone is the stock ``torch.nn.TransformerEncoder`` (no positional
encoding, so it is permutation equivariant; the self-attention dispatches to
``scaled_dot_product_attention`` Flash / mem-efficient kernels, keeping the
``N=8192`` self-attention at ``O(N)`` memory). The instance-embedding loss is
the standard discriminative loss of De Brabandere et al., 2017
("Semantic Instance Segmentation with a Discriminative Loss Function").
"""
from __future__ import annotations

import math
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


# ----------------------------------------------------------------------
# Differentiable curve samplers (shared by the geom loss; mirrored in numpy by
# the decoder). All operate on a leading batch of edges ``(M, 3)`` control
# points and return ``(M, num, 3)`` polylines.
# ----------------------------------------------------------------------
def sample_line(a: torch.Tensor, b: torch.Tensor, num: int) -> torch.Tensor:
    """Sample ``num`` points on the straight segment ``a -> b``."""
    t = torch.linspace(0.0, 1.0, num, device=a.device, dtype=a.dtype).view(1, num, 1)
    return a[:, None, :] * (1.0 - t) + b[:, None, :] * t


def sample_bezier(
    a: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor, b: torch.Tensor, num: int
) -> torch.Tensor:
    """Cubic curve interpolating the four on-curve points ``a, q1, q2, b``.

    ``q1`` / ``q2`` are the t=1/3 / t=2/3 coordinates, so the Bezier control
    points are solved so the curve passes through all four (not the usual
    control-polygon interpretation).
    """
    # Solve the two interior control points P1, P2 from the on-curve constraints
    # B(1/3)=q1, B(2/3)=q2 (with P0=a, P3=b). Closed form (see dataset notes).
    big_a = 27.0 * q1 - 8.0 * a - b
    big_b = 27.0 * q2 - a - 8.0 * b
    p1 = (2.0 * big_a - big_b) / 18.0
    p2 = (2.0 * big_b - big_a) / 18.0
    t = torch.linspace(0.0, 1.0, num, device=a.device, dtype=a.dtype).view(1, num, 1)
    mt = 1.0 - t
    return (
        mt ** 3 * a[:, None, :]
        + 3.0 * mt ** 2 * t * p1[:, None, :]
        + 3.0 * mt * t ** 2 * p2[:, None, :]
        + t ** 3 * b[:, None, :]
    )


def sample_arc(
    a: torch.Tensor, m: torch.Tensor, b: torch.Tensor, num: int
) -> torch.Tensor:
    """Sample the circular arc through three points ``a -> m -> b``.

    Falls back to the straight segment ``a -> b`` when the three points are
    (near-)collinear.
    """
    eps = 1e-8
    aa = a - m
    bb = b - m
    cr = torch.cross(aa, bb, dim=-1)                       # (M, 3)
    cr_n2 = (cr * cr).sum(-1, keepdim=True)                # (M, 1)
    alpha = (aa * aa).sum(-1, keepdim=True)
    beta = (bb * bb).sum(-1, keepdim=True)
    center = m + torch.cross(alpha * bb - beta * aa, cr, dim=-1) / (
        2.0 * cr_n2.clamp_min(eps))
    ua = a - center
    r = ua.norm(dim=-1, keepdim=True)                     # (M, 1)
    u = ua / r.clamp_min(eps)
    nrm = cr / cr.norm(dim=-1, keepdim=True).clamp_min(eps)
    v = torch.cross(nrm, u, dim=-1)

    def _ang(p: torch.Tensor) -> torch.Tensor:
        d = p - center
        return torch.atan2((d * v).sum(-1), (d * u).sum(-1))  # (M,)

    two_pi = 2.0 * math.pi
    m_ang = _ang(m) % two_pi
    b_ang = _ang(b) % two_pi
    sweep = torch.where(m_ang <= b_ang, b_ang, b_ang - two_pi)   # (M,)
    t = torch.linspace(0.0, 1.0, num, device=a.device, dtype=a.dtype)  # (num,)
    theta = sweep[:, None] * t[None, :]                  # (M, num)
    pts = center[:, None, :] + r[:, None, :] * (
        torch.cos(theta)[..., None] * u[:, None, :]
        + torch.sin(theta)[..., None] * v[:, None, :]
    )
    collinear = (cr_n2.squeeze(-1) < eps)                # (M,)
    return torch.where(collinear[:, None, None], sample_line(a, b, num), pts)


def sample_curve_by_type(
    a: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    b: torch.Tensor,
    ctype: torch.Tensor,
    num: int,
) -> torch.Tensor:
    """Sample each edge's curve by its (integer) ``ctype`` 0=line/1=arc/2=bezier.

    All three parameterisations are evaluated and selected per edge (cheap and
    fully vectorised). ``a, q1, q2, b`` are ``(M, 3)``; ``ctype`` is ``(M,)``.
    """
    line = sample_line(a, b, num)
    arc = sample_arc(a, q1, b, num)
    bez = sample_bezier(a, q1, q2, b, num)
    sel = ctype.view(-1, 1, 1)
    out = torch.where(sel == 1, arc, line)
    out = torch.where(sel == 2, bez, out)
    return out


class WireframeGrouper(nn.Module):
    """Point-set transformer producing per-point grouping fields.

    Args:
        point_dim: input channels per point (``3`` = xyz).
        d_model: transformer width.
        depth: number of ``TransformerEncoderLayer`` blocks.
        nhead: attention heads.
        mlp_ratio: feed-forward expansion.
        dropout: dropout in attention / feed-forward / heads.
        embed_dim: instance-embedding dimension ``D``.
    """

    def __init__(
        self,
        point_dim: int = 3,
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
        self.endpoint_offset_head = _mlp(d_model, dh, 6)
        self.embedding_head = _mlp(d_model, dh, embed_dim)
        self.curve_type_head = _mlp(d_model, dh, 3)
        self.anchor_head = _mlp(d_model, dh, 6)
        self.arclen_head = _mlp(d_model, dh, 1)

    def forward(self, pts: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the grouper.

        Args:
            pts: ``(B, N, point_dim)`` anchor point set ``xyz``.

        Returns:
            dict of per-point fields::

                endpoint_offset (B, N, 2, 3)
                embedding       (B, N, D)
                curve_type      (B, N, 3)
                anchor          (B, N, 2, 3)
                arclen          (B, N)
        """
        b, n, _ = pts.shape
        h = self.in_proj(pts)
        h = self.encoder(h)
        h = self.norm(h)
        return {
            "endpoint_offset": self.endpoint_offset_head(h).reshape(b, n, 2, 3),
            "embedding": self.embedding_head(h),
            "curve_type": self.curve_type_head(h),
            "anchor": self.anchor_head(h).reshape(b, n, 2, 3),
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


def _group_mean(x: torch.Tensor, inv: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Mean of rows of ``x (M, D)`` per group id ``inv (M,)`` -> ``(num_groups, D)``."""
    out = x.new_zeros(num_groups, x.shape[1])
    out.index_add_(0, inv, x)
    cnt = torch.bincount(inv, minlength=num_groups).clamp_min(1).to(x.dtype)
    return out / cnt[:, None]


def grouper_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    gt_wireframes: list[dict[str, torch.Tensor]] | None = None,
    w_endpoint: float = 1.0,
    w_anchor: float = 1.0,
    w_curve_type: float = 1.0,
    w_arclen: float = 1.0,
    w_embed: float = 1.0,
    w_topo: float = 1.0,
    w_curve_geom: float = 1.0,
    curve_type_class_weights: list[float] | None = None,
    topo_tau: float = 0.1,
    geom_num_per_edge: int = 32,
    embed_kwargs: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Total stage-2 loss + components (fully teacher-forced).

    Expects ``batch`` with the keys produced by ``collate_grouper_batch``
    (``wf_points (B,N,3)``, ``lbl_edge_id``, ``lbl_arclen``, ``lbl_endpoint_a/b``,
    ``lbl_curve_type``, ``lbl_anchor1/2``) and, for the topology + geometry
    terms, the native GT graphs in ``gt_wireframes``.
    """
    embed_kwargs = embed_kwargs or {}
    pts = batch["wf_points"]
    xyz = pts[..., :3]
    device = xyz.device
    nb = xyz.shape[0]

    eoff = out["endpoint_offset"]          # (B, N, 2, 3)
    p_a = xyz + eoff[..., 0, :]
    p_b = xyz + eoff[..., 1, :]
    anc = out["anchor"]                    # (B, N, 2, 3)
    q1p = xyz + anc[..., 0, :]
    q2p = xyz + anc[..., 1, :]

    ea = batch["lbl_endpoint_a"]
    eb = batch["lbl_endpoint_b"]
    ga1 = batch["lbl_anchor1"]
    ga2 = batch["lbl_anchor2"]
    edge_id = batch["lbl_edge_id"]

    def _sl1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(a, b, reduction="none").sum(dim=-1)  # (B, N)

    # --- endpoint + anchor voting (order-invariant 4-tuple) ---
    ep_fwd = _sl1(p_a, ea) + _sl1(p_b, eb)
    ep_flip = _sl1(p_a, eb) + _sl1(p_b, ea)
    an_fwd = _sl1(q1p, ga1) + _sl1(q2p, ga2)
    an_flip = _sl1(q1p, ga2) + _sl1(q2p, ga1)
    use_flip = (ep_flip + an_flip) < (ep_fwd + an_fwd)
    l_endpoint = torch.where(use_flip, ep_flip, ep_fwd).mean()
    l_anchor = torch.where(use_flip, an_flip, an_fwd).mean()

    # --- curve-type classification (per point, class-weighted CE) ---
    ct_logits = out["curve_type"].reshape(-1, 3)
    ct_target = batch["lbl_curve_type"].reshape(-1).long()
    weight = (
        ct_logits.new_tensor(list(curve_type_class_weights))
        if curve_type_class_weights is not None else None
    )
    l_curve_type = F.cross_entropy(ct_logits, ct_target, weight=weight)

    # --- arc-length regression ---
    l_arclen = F.mse_loss(out["arclen"], batch["lbl_arclen"])

    # --- instance embedding (per sample, grouped by edge id) ---
    emb = out["embedding"]
    l_embed = pts.new_zeros(())
    n_emb = 0
    for b in range(nb):
        ids = edge_id[b]
        m = ids >= 0
        if not torch.any(m):
            continue
        uniq, remapped = torch.unique(ids[m], return_inverse=True)
        if uniq.numel() < 1:
            continue
        l_embed = l_embed + discriminative_loss(
            emb[b][m], remapped, **embed_kwargs)
        n_emb += 1
    if n_emb > 0:
        l_embed = l_embed / n_emb

    # --- topology-assignment CE + teacher-forced curve Chamfer ---
    l_topo = pts.new_zeros(())
    l_geom = pts.new_zeros(())
    n_topo = 0
    n_geom = 0
    if gt_wireframes is not None:
        for b in range(nb):
            g = gt_wireframes[b]
            ids = edge_id[b]
            valid = ids >= 0
            if not torch.any(valid):
                continue

            # topology: vote each endpoint onto the GT vertex set (softmax over
            # -dist/tau) and CE to the edge's two real vertex ids (order-inv).
            verts = g["vertices"].to(device).float()
            e_idx = g["edge_index"].to(device).long()
            if w_topo != 0.0 and verts.shape[0] >= 1 and e_idx.shape[0] >= 1:
                vids = ids[valid].clamp(max=e_idx.shape[0] - 1)
                tgt = e_idx[vids]                       # (M, 2)
                ti, tj = tgt[:, 0], tgt[:, 1]
                la = -torch.cdist(p_a[b][valid], verts) / max(topo_tau, 1e-6)
                lb = -torch.cdist(p_b[b][valid], verts) / max(topo_tau, 1e-6)
                ce_ai = F.cross_entropy(la, ti, reduction="none")
                ce_aj = F.cross_entropy(la, tj, reduction="none")
                ce_bi = F.cross_entropy(lb, ti, reduction="none")
                ce_bj = F.cross_entropy(lb, tj, reduction="none")
                topo_cost = torch.minimum(ce_ai + ce_bj, ce_aj + ce_bi)
                l_topo = l_topo + topo_cost.mean()
                n_topo += 1

            # geometry: per GT edge, aggregate the predicted (a, q1, q2, b),
            # parse by predicted type and Chamfer to the GT curve points.
            gep = g["edge_points"].to(device).float()
            if w_curve_geom != 0.0 and gep.shape[0] >= 1:
                vids = ids[valid].clamp(max=gep.shape[0] - 1)
                uniq, inv = torch.unique(vids, return_inverse=True)
                gnum = uniq.shape[0]
                ea_v, eb_v = ea[b][valid], eb[b][valid]
                pa_v, pb_v = p_a[b][valid], p_b[b][valid]
                d_fwd = (pa_v - ea_v).pow(2).sum(-1) + (pb_v - eb_v).pow(2).sum(-1)
                d_flip = (pa_v - eb_v).pow(2).sum(-1) + (pb_v - ea_v).pow(2).sum(-1)
                flip = (d_flip < d_fwd)[:, None]
                a_or = torch.where(flip, pb_v, pa_v)
                b_or = torch.where(flip, pa_v, pb_v)
                # anchors follow the same a<->b orientation (q1 near a, q2 near b)
                q1_v, q2_v = q1p[b][valid], q2p[b][valid]
                q1_or = torch.where(flip, q2_v, q1_v)
                q2_or = torch.where(flip, q1_v, q2_v)
                a_g = _group_mean(a_or, inv, gnum)
                b_g = _group_mean(b_or, inv, gnum)
                q1_g = _group_mean(q1_or, inv, gnum)
                q2_g = _group_mean(q2_or, inv, gnum)
                ct = out["curve_type"][b][valid].argmax(dim=-1)
                ct_oh = F.one_hot(ct, num_classes=3).to(xyz.dtype)
                ct_g = _group_mean(ct_oh, inv, gnum).argmax(dim=-1)
                pred_curve = sample_curve_by_type(
                    a_g, q1_g, q2_g, b_g, ct_g, int(geom_num_per_edge))
                gt_curve = gep[uniq]
                dmat = torch.cdist(pred_curve, gt_curve)   # (G, P, U)
                cd = 0.5 * (
                    dmat.min(dim=2)[0].mean(dim=1)
                    + dmat.min(dim=1)[0].mean(dim=1)
                )
                l_geom = l_geom + cd.mean()
                n_geom += 1
    if n_topo > 0:
        l_topo = l_topo / n_topo
    if n_geom > 0:
        l_geom = l_geom / n_geom

    total = (
        w_endpoint * l_endpoint
        + w_anchor * l_anchor
        + w_curve_type * l_curve_type
        + w_arclen * l_arclen
        + w_embed * l_embed
        + w_topo * l_topo
        + w_curve_geom * l_geom
    )
    return {
        "loss": total,
        "loss_endpoint": l_endpoint.detach(),
        "loss_anchor": l_anchor.detach(),
        "loss_curve_type": l_curve_type.detach(),
        "loss_arclen": l_arclen.detach(),
        "loss_embed": l_embed.detach(),
        "loss_topo": l_topo.detach(),
        "loss_curve_geom": l_geom.detach(),
    }


__all__ = [
    "WireframeGrouper",
    "discriminative_loss",
    "grouper_loss",
    "sample_line",
    "sample_arc",
    "sample_bezier",
    "sample_curve_by_type",
]
