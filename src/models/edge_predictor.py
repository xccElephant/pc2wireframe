"""Stage-2 edge predictor: vertices + latent ``z`` -> wireframe connectivity.

Stage 1 (Rectified Flow) recovers the wireframe **corners** (a deduplicated
vertex set ``V``). Stage 2 turns that vertex set into an explicit wireframe by
predicting, for every *pair* of vertices, whether an edge connects them, and,
for the edges that exist, the curve geometry between the two corners. The shape
latent ``z (B, K, D)`` (same frozen-PTv3 + trainable-compressor encoder as
stage 1, but an **independent** compressor) is the conditioning signal so the
connectivity decision is grounded in the real geometry, not just the corner
coordinates.

Architecture (see :class:`EdgePredictor`):

  * **Vertex encoder** -- ``Linear(3 -> d)`` then ``depth`` transformer blocks,
    each = padded vertex self-attention + cross-attention to ``z`` -> per-vertex
    features ``h (B, Vmax, d)``.
  * **Edge-existence head** (symmetric / undirected) -- for each pair ``(i, j)``
    build ``[h_i + h_j, |h_i - h_j|, h_i * h_j, dist_ij, dir_ij]`` (optionally
    concatenated with along-edge ``z`` evidence) -> MLP -> logit. The whole
    ``(B, Vmax, Vmax)`` matrix is computed at once and symmetrized.
  * **Along-edge geometric evidence** (PC2WF-style) -- sample ``M`` query points
    on each candidate segment ``a -> b``, cross-attend them to ``z`` and pool,
    so edge existence depends on real shape evidence along the segment rather
    than the two endpoints alone.
  * **Curve head** (prune-then-refine) -- run only on *selected* pairs (GT
    positives in training, thresholded positives at inference) to avoid the
    ``Vmax^2 * U * 3`` blow-up. From ``[h_i, h_j, endpoints, z-context]`` it
    predicts a ``U x 3`` residual on the straight-line ``a -> b`` baseline; the
    residual is multiplied by an envelope that vanishes at both ends, so the
    endpoints stay pinned to the two vertices (same trick as the baseline
    ``CurveVAE`` decoder).

The decode / assembly helpers (:func:`dedup_vertices`, :func:`assemble_wireframe`)
turn a stage-1 point cloud + the predicted edge logits into the
``{vertices, edge_index, edge_points}`` schema used by :mod:`src.metrics`.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
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


class _EdgeBlock(nn.Module):
    """One vertex-encoder block: masked self-attn + cross-attn to ``z`` + MLP.

    Pre-norm residual block. The self-attention runs over the padded vertex set
    (padded rows masked out via ``key_padding_mask``); the cross-attention reads
    the latent tokens ``z`` so per-vertex features are conditioned on the shape.
    """

    def __init__(
        self, dim: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, nhead, dropout=dropout, batch_first=True)
        self.norm_ca = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, nhead, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        z_tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        a = self.norm_sa(h)
        h = h + self.self_attn(
            a, a, a, key_padding_mask=key_padding_mask, need_weights=False)[0]
        c = self.norm_ca(h)
        h = h + self.cross_attn(c, z_tokens, z_tokens, need_weights=False)[0]
        h = h + self.mlp(self.norm_mlp(h))
        return h


class _AlongEdgeEvidence(nn.Module):
    """Pool along-segment ``z`` evidence for every candidate vertex pair.

    For each pair ``(i, j)`` sample ``num_points`` interior points on the
    segment ``v_i -> v_j``, embed their xyz, let them cross-attend to the latent
    tokens ``z`` (scaled-dot-product attention), and mean-pool over the samples.
    Returns a ``(B, V, V, evidence_dim)`` tensor fed to the edge head.

    Cost is ``O(B * V^2 * num_points)`` queries; keep ``Vmax`` and
    ``num_points`` modest (the module materialises ``B * V^2 * num_points``
    query rows).
    """

    def __init__(
        self,
        z_dim: int,
        evidence_dim: int,
        nhead: int,
        num_points: int,
    ) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.nhead = int(nhead)
        self.evidence_dim = int(evidence_dim)
        self.q_proj = nn.Linear(3, evidence_dim)
        self.k_proj = nn.Linear(z_dim, evidence_dim)
        self.v_proj = nn.Linear(z_dim, evidence_dim)
        self.out_proj = nn.Linear(evidence_dim, evidence_dim)

    def forward(self, verts: torch.Tensor, z_tokens: torch.Tensor) -> torch.Tensor:
        b, v, _ = verts.shape
        m = self.num_points
        # interior sample positions t in (0, 1), avoiding the endpoints.
        t = torch.linspace(0.0, 1.0, m + 2, device=verts.device, dtype=verts.dtype)
        t = t[1:-1].view(1, 1, 1, m, 1)
        vi = verts[:, :, None, None, :]            # (B, V, 1, 1, 3)
        vj = verts[:, None, :, None, :]            # (B, 1, V, 1, 3)
        pts = vi + t * (vj - vi)                   # (B, V, V, M, 3)
        q = self.q_proj(pts).reshape(b, v * v * m, self.evidence_dim)
        k = self.k_proj(z_tokens)                  # (B, K, E)
        val = self.v_proj(z_tokens)

        def _split(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(x.shape[0], x.shape[1], self.nhead, -1).transpose(1, 2)

        out = F.scaled_dot_product_attention(_split(q), _split(k), _split(val))
        out = out.transpose(1, 2).reshape(b, v * v * m, self.evidence_dim)
        out = out.reshape(b, v, v, m, self.evidence_dim).mean(dim=3)
        return self.out_proj(out)                  # (B, V, V, E)


class EdgePredictor(nn.Module):
    """Vertices + latent ``z`` -> symmetric edge logits + per-edge curves.

    Args:
        point_dim: input vertex channels (``3`` = xyz).
        z_dim: latent token channels ``D``.
        d_model: transformer width.
        depth: number of vertex-encoder blocks.
        nhead: attention heads.
        mlp_ratio: feed-forward expansion.
        dropout: dropout in attention / MLPs.
        pair_dim: per-vertex feature width used to form the (memory-bounded)
            pairwise features for the edge head.
        num_edge_points: ``U`` points per reconstructed curve.
        use_edge_evidence: enable the along-edge ``z`` evidence head.
        edge_evidence_points: ``M`` samples per candidate segment.
        evidence_dim: along-edge evidence feature width.
    """

    def __init__(
        self,
        point_dim: int = 3,
        z_dim: int = 64,
        d_model: int = 256,
        depth: int = 6,
        nhead: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pair_dim: int = 64,
        num_edge_points: int = 32,
        use_edge_evidence: bool = True,
        edge_evidence_points: int = 4,
        evidence_dim: int = 64,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        if use_edge_evidence and evidence_dim % nhead != 0:
            raise ValueError(
                f"evidence_dim ({evidence_dim}) must be divisible by nhead ({nhead})")
        self.point_dim = int(point_dim)
        self.num_edge_points = int(num_edge_points)
        self.use_edge_evidence = bool(use_edge_evidence)

        self.vertex_in = nn.Linear(point_dim, d_model)
        self.z_proj = nn.Linear(z_dim, d_model)
        self.blocks = nn.ModuleList(
            [
                _EdgeBlock(d_model, nhead, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

        self.pair_proj = nn.Linear(d_model, pair_dim)
        # pairwise feature = [h_i+h_j, |h_i-h_j|, h_i*h_j] (3*pair_dim)
        #                    + [dist, dir] (1 + 3)
        #                    + along-edge evidence (evidence_dim, optional)
        edge_in = 3 * pair_dim + 4
        if self.use_edge_evidence:
            edge_in += evidence_dim
            self.evidence = _AlongEdgeEvidence(
                z_dim=d_model, evidence_dim=evidence_dim,
                nhead=nhead, num_points=edge_evidence_points)
        self.edge_head = _mlp(edge_in, d_model, 1)

        # curve head: [h_i, h_j, a(3), b(3), z_global(d_model)] -> U*3 residual
        curve_in = 2 * d_model + 6 + d_model
        self.curve_head = _mlp(curve_in, d_model, self.num_edge_points * 3)

    # ------------------------------------------------------------------
    def encode_vertices(
        self,
        verts: torch.Tensor,
        vertex_mask: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the vertex encoder.

        Args:
            verts: ``(B, V, 3)`` padded vertex coordinates.
            vertex_mask: ``(B, V)`` bool, ``True`` for real (valid) vertices.
            z: ``(B, K, z_dim)`` latent tokens.

        Returns:
            ``(h (B, V, d), z_tokens (B, K, d), z_global (B, d))``.
        """
        z_tokens = self.z_proj(z)
        key_padding_mask = ~vertex_mask  # True at padded positions (MHA conv.)
        h = self.vertex_in(verts)
        for block in self.blocks:
            h = block(h, z_tokens, key_padding_mask)
        h = self.norm(h)
        z_global = z_tokens.mean(dim=1)
        return h, z_tokens, z_global

    def edge_logits(
        self,
        h: torch.Tensor,
        verts: torch.Tensor,
        vertex_mask: torch.Tensor,
        z_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetric edge-existence logits ``(B, V, V)``.

        Padded rows / columns and the diagonal (self-loops) are forced to a
        large negative logit so they never fire.
        """
        b, v, _ = h.shape
        eps = 1e-8
        hp = self.pair_proj(h)                     # (B, V, pair_dim)
        hi = hp[:, :, None, :]
        hj = hp[:, None, :, :]
        feats = [hi + hj, (hi - hj).abs(), hi * hj]
        sum_, absdiff, prod = (
            f.expand(b, v, v, hp.shape[-1]) for f in feats)

        diff = verts[:, None, :, :] - verts[:, :, None, :]   # (B, V, V, 3)
        dist = diff.norm(dim=-1, keepdim=True)               # (B, V, V, 1)
        dir_ = diff / dist.clamp_min(eps)                    # (B, V, V, 3)

        parts = [sum_, absdiff, prod, dist, dir_]
        if self.use_edge_evidence:
            parts.append(self.evidence(verts, z_tokens))
        feat = torch.cat(parts, dim=-1)
        logit = self.edge_head(feat).squeeze(-1)             # (B, V, V)
        logit = 0.5 * (logit + logit.transpose(1, 2))        # symmetric

        valid = vertex_mask[:, :, None] & vertex_mask[:, None, :]
        eye = torch.eye(v, dtype=torch.bool, device=h.device)[None]
        mask = valid & ~eye
        return logit.masked_fill(~mask, -30.0)

    def curves_for_pairs(
        self,
        h: torch.Tensor,
        verts: torch.Tensor,
        z_global: torch.Tensor,
        bi: torch.Tensor,
        ii: torch.Tensor,
        jj: torch.Tensor,
    ) -> torch.Tensor:
        """Decode curves for a flat list of selected ``(batch, i, j)`` pairs.

        Args:
            h: ``(B, V, d)`` per-vertex features.
            verts: ``(B, V, 3)`` vertex coordinates.
            z_global: ``(B, d)`` per-sample latent summary.
            bi / ii / jj: ``(M,)`` long tensors selecting the pair endpoints.

        Returns:
            ``(M, U, 3)`` curves whose endpoints are pinned to ``v_i`` / ``v_j``.
        """
        m = bi.shape[0]
        u = self.num_edge_points
        if m == 0:
            return verts.new_zeros((0, u, 3))
        h_i = h[bi, ii]                            # (M, d)
        h_j = h[bi, jj]
        a = verts[bi, ii]                          # (M, 3)
        b = verts[bi, jj]
        zc = z_global[bi]                          # (M, d)
        feat = torch.cat([h_i, h_j, a, b, zc], dim=-1)
        res = self.curve_head(feat).reshape(m, u, 3)

        t = torch.linspace(0.0, 1.0, u, device=verts.device, dtype=verts.dtype)
        line = a[:, None, :] * (1.0 - t)[None, :, None] + \
            b[:, None, :] * t[None, :, None]       # (M, U, 3)
        # sin envelope vanishes at t=0 and t=1 -> endpoints pinned to a / b.
        env = torch.sin(math.pi * t).view(1, u, 1)
        return line + res * env

    def forward(
        self,
        verts: torch.Tensor,
        vertex_mask: torch.Tensor,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode vertices and produce edge logits.

        Returns a dict with ``edge_logits (B, V, V)`` plus the cached
        ``h`` / ``z_global`` needed to decode curves on selected pairs.
        """
        h, z_tokens, z_global = self.encode_vertices(verts, vertex_mask, z)
        logits = self.edge_logits(h, verts, vertex_mask, z_tokens)
        return {"edge_logits": logits, "h": h, "z_global": z_global}


# ----------------------------------------------------------------------
# Decode / assembly helpers (numpy; mirror the metrics schema)
# ----------------------------------------------------------------------
def dedup_vertices(
    points: np.ndarray,
    *,
    eps: float = 0.02,
    relative: bool = True,
    min_samples: int = 1,
) -> np.ndarray:
    """Collapse a stage-1 corner point cloud into vertices via DBSCAN.

    Args:
        points: ``(N, 3)`` sampled corner points (the stage-1 RF output).
        eps: DBSCAN neighbourhood radius. When ``relative`` it is a fraction of
            the point cloud's bounding-box diagonal (scale-invariant); otherwise
            an absolute distance.
        relative: interpret ``eps`` relative to the spatial extent.
        min_samples: DBSCAN ``min_samples`` (``1`` -> no points are dropped as
            noise; every cluster contributes one vertex).

    Returns:
        ``(K, 3)`` float32 vertex centroids.
    """
    from sklearn.cluster import DBSCAN

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if relative:
        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        radius = float(eps) * max(diag, 1e-9)
    else:
        radius = float(eps)
    labels = DBSCAN(eps=max(radius, 1e-9), min_samples=int(min_samples)).fit_predict(pts)
    uniq = [c for c in np.unique(labels) if c >= 0]
    if not uniq:
        return np.zeros((0, 3), dtype=np.float32)
    centers = np.stack([pts[labels == c].mean(axis=0) for c in uniq], axis=0)
    return centers.astype(np.float32)


def assemble_wireframe(
    vertices: np.ndarray,
    edge_index: np.ndarray,
    edge_points: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Pack arrays into the ``{vertices, edge_index, edge_points}`` schema."""
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    edge_index = np.asarray(edge_index, dtype=np.int64).reshape(-1, 2)
    if edge_points is None or len(edge_points) == 0:
        edge_points = np.zeros((edge_index.shape[0], 0, 3), dtype=np.float32)
    else:
        edge_points = np.asarray(edge_points, dtype=np.float32)
    return {
        "vertices": vertices,
        "edge_index": edge_index,
        "edge_points": edge_points,
    }


def edges_from_logits(
    logits: np.ndarray,
    num_valid: int,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Threshold a symmetric logit matrix into an undirected ``edge_index``.

    Args:
        logits: ``(V, V)`` edge logits (already symmetric).
        num_valid: number of real vertices (the top-left block to consider).
        threshold: probability threshold (``sigmoid(logit) >= threshold``).

    Returns:
        ``(E, 2)`` int64 array of ``i < j`` vertex-id pairs.
    """
    v = int(num_valid)
    if v < 2:
        return np.zeros((0, 2), dtype=np.int64)
    sub = np.asarray(logits, dtype=np.float64)[:v, :v]
    prob = 1.0 / (1.0 + np.exp(-sub))
    iu, ju = np.triu_indices(v, k=1)
    keep = prob[iu, ju] >= float(threshold)
    return np.stack([iu[keep], ju[keep]], axis=1).astype(np.int64)


__all__ = [
    "EdgePredictor",
    "dedup_vertices",
    "assemble_wireframe",
    "edges_from_logits",
]
