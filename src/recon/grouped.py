"""Decode the stage-2 grouper's per-point fields into an explicit wireframe.

This is the learned counterpart of :mod:`src.recon.traditional`. Where the
traditional reconstructor *guesses* vertices / connectivity / ordering with
fragile heuristics, here the network has already regressed those quantities, so
decoding is pure book-keeping (no ``merge_radius`` / ``min_votes`` /
chord-projection):

  1. **Vertices** = cluster the voted vertex centres ``xyz + vertex_offset`` of
     the points classified as vertices (``sigmoid(vertex_score) >= thr``) with
     ``sklearn.cluster.DBSCAN``; each cluster centroid is a vertex.
  2. **Connectivity** = for each edge point, snap its two voted endpoints
     ``xyz + endpoint_offset`` to the nearest vertices (``scipy`` KD-tree) to get
     an unordered vertex pair ``(va, vb)``; group edge points by that pair.
  3. **Splitting** = within a pair, optionally split by the instance
     ``embedding`` (DBSCAN) so that two edges sharing both endpoints (e.g. the
     two arcs of a split circle) become separate edges.
  4. **Curves** = order each edge's points by the predicted ``arclen`` and
     resample to ``num_per_edge`` points, pinned to the two endpoint vertices.

Output matches the GT schema used by :mod:`src.metrics`::

    vertices:    (V, 3)    float32
    edge_index:  (E, 2)    int64
    edge_points: (E, P, 3) float32
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .traditional import _empty_wireframe, _resample_polyline


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _cluster_dbscan(pts: np.ndarray, eps: float) -> np.ndarray:
    """DBSCAN cluster centroids ``(K, 3)`` (``min_samples=1`` -> no noise)."""
    from sklearn.cluster import DBSCAN

    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    labels = DBSCAN(eps=float(eps), min_samples=1).fit_predict(pts)
    k = int(labels.max()) + 1
    centers = np.stack(
        [pts[labels == c].mean(axis=0) for c in range(k)], axis=0)
    return centers


def _bbox_diag(pts: np.ndarray) -> float:
    """Bounding-box diagonal length of a point set (0 if empty)."""
    if pts.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def group_wireframe(
    fields: dict[str, np.ndarray],
    *,
    vertex_thresh: float = 0.5,
    vertex_merge_radius: float = 0.01,
    merge_relative: bool = True,
    split_by_embedding: bool = True,
    embed_eps: float = 0.5,
    min_edge_points: int = 3,
    num_per_edge: int = 32,
) -> dict[str, np.ndarray]:
    """Group per-point grouper outputs into ``{vertices, edge_index, edge_points}``.

    Args:
        fields: per-point numpy arrays for a *single* sample with keys
            ``xyz (N,3)``, ``vertex_score (N,)`` (logits), ``vertex_offset
            (N,3)``, ``endpoint_offset (N,2,3)``, ``embedding (N,D)``,
            ``arclen (N,)``.
        vertex_thresh: ``sigmoid(vertex_score) >= thr`` -> vertex point.
        vertex_merge_radius: vertex-clustering DBSCAN ``eps``. When
            ``merge_relative`` it is a **fraction of the point-set bounding-box
            diagonal** (scale-invariant); otherwise an absolute distance.
        merge_relative: interpret ``vertex_merge_radius`` relative to the
            input's spatial extent rather than as an absolute value.
        split_by_embedding: split same-endpoint edges via embedding DBSCAN.
        embed_eps: DBSCAN ``eps`` in embedding space (set near ``delta_dist``).
        min_edge_points: drop edges supported by fewer points than this.
        num_per_edge: points per reconstructed edge curve.
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(fields["xyz"], dtype=np.float64).reshape(-1, 3)
    score = _sigmoid(np.asarray(fields["vertex_score"]).reshape(-1))
    voff = np.asarray(fields["vertex_offset"], dtype=np.float64).reshape(-1, 3)
    eoff = np.asarray(fields["endpoint_offset"], dtype=np.float64).reshape(-1, 2, 3)
    emb = np.asarray(fields["embedding"], dtype=np.float64).reshape(xyz.shape[0], -1)
    arclen = np.asarray(fields["arclen"], dtype=np.float64).reshape(-1)

    # Resolve the (possibly relative) vertex-merge radius to an absolute eps.
    if merge_relative:
        merge_eps = float(vertex_merge_radius) * max(_bbox_diag(xyz), 1e-9)
    else:
        merge_eps = float(vertex_merge_radius)

    is_v = score >= float(vertex_thresh)
    vcenters = xyz[is_v] + voff[is_v]
    vertices = _cluster_dbscan(vcenters, merge_eps)

    if vertices.shape[0] < 2 or (~is_v).sum() == 0:
        return _empty_wireframe(vertices, num_per_edge)

    # Edge points: snap both voted endpoints to the nearest vertices.
    e_mask = ~is_v
    e_xyz = xyz[e_mask]
    e_emb = emb[e_mask]
    e_arclen = arclen[e_mask]
    pa = e_xyz + eoff[e_mask, 0, :]
    pb = e_xyz + eoff[e_mask, 1, :]

    tree = cKDTree(vertices)
    _, ia = tree.query(pa, k=1)
    _, ib = tree.query(pb, k=1)
    ia = np.asarray(ia, dtype=np.int64).reshape(-1)
    ib = np.asarray(ib, dtype=np.int64).reshape(-1)
    lo = np.minimum(ia, ib)
    hi = np.maximum(ia, ib)
    valid = lo != hi

    groups: dict[tuple[int, int], list[int]] = {}
    for k in np.nonzero(valid)[0]:
        groups.setdefault((int(lo[k]), int(hi[k])), []).append(int(k))

    edge_index: list[tuple[int, int]] = []
    edge_curves: list[np.ndarray] = []

    def _emit(i: int, j: int, members: np.ndarray) -> None:
        if members.shape[0] < int(min_edge_points):
            return
        order = np.argsort(e_arclen[members], kind="stable")
        ordered = e_xyz[members][order]
        a, b = vertices[i], vertices[j]
        # Orient the curve a -> b (flip if the first sample is nearer b).
        if (np.linalg.norm(ordered[0] - a) > np.linalg.norm(ordered[0] - b)):
            ordered = ordered[::-1]
        poly = np.concatenate([a[None, :], ordered, b[None, :]], axis=0)
        edge_index.append((i, j))
        edge_curves.append(_resample_polyline(poly, num_per_edge))

    for (i, j), members in sorted(groups.items()):
        members = np.asarray(members, dtype=np.int64)
        if split_by_embedding and members.shape[0] >= 2:
            from sklearn.cluster import DBSCAN

            sub = DBSCAN(eps=float(embed_eps), min_samples=1).fit_predict(
                e_emb[members])
            for c in range(int(sub.max()) + 1):
                _emit(i, j, members[sub == c])
        else:
            _emit(i, j, members)

    if not edge_index:
        return _empty_wireframe(vertices, num_per_edge)

    return {
        "vertices": vertices.astype(np.float32),
        "edge_index": np.asarray(edge_index, dtype=np.int64),
        "edge_points": np.stack(edge_curves, axis=0).astype(np.float32),
    }


__all__ = ["group_wireframe"]
