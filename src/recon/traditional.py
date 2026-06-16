"""Deterministic wireframe reconstruction from an RF-generated point set.

The Rectified-Flow sampler emits a fixed-size point set ``(N, 4)`` whose last
channel is a continuous *type* (``~1`` = vertex, ``~0`` = edge). This module
turns that point set into an explicit wireframe graph with a simple,
fully-deterministic (no learning, no randomness) pipeline:

  1. **Split** points by ``type`` at ``type_threshold`` into vertex points and
     edge points.
  2. **Vertices** = greedy radius-merge clustering of the vertex points
     (cluster centroids). Greedy seeding follows a fixed lexicographic order so
     the result is deterministic.
  3. **Edges** = "nearest-two-vertices voting": every edge point votes for the
     unordered pair of its two nearest vertices; pairs with at least
     ``min_votes`` votes become edges.
  4. **Edge curves** = each edge's supporting edge points (the ones that voted
     for it), ordered by their projection onto the endpoint-to-endpoint line
     and resampled to ``num_per_edge`` points; edges with no support fall back
     to a straight segment.

Output matches the GT schema used by :mod:`src.metrics`::

    vertices:    (V, 3)   float32
    edge_index:  (E, 2)   int64
    edge_points: (E, P, 3) float32
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy(points: Any) -> np.ndarray:
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()
    return np.asarray(points, dtype=np.float64).reshape(-1, 4)


def _cluster_radius(
    pts: np.ndarray, radius: float
) -> np.ndarray:
    """Greedy radius-merge clustering; returns cluster centroids ``(V, 3)``.

    Deterministic: seeds are visited in a fixed lexicographic order and each
    seed absorbs all not-yet-assigned points within ``radius``.
    """
    from scipy.spatial import cKDTree

    n = pts.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
    tree = cKDTree(pts)
    assigned = np.full(n, -1, dtype=np.int64)
    centers: list[np.ndarray] = []
    for i in order:
        if assigned[i] >= 0:
            continue
        neigh = tree.query_ball_point(pts[i], radius)
        free = [j for j in neigh if assigned[j] < 0]
        if not free:
            free = [int(i)]
        cid = len(centers)
        assigned[np.asarray(free, dtype=np.int64)] = cid
        centers.append(pts[free].mean(axis=0))
    return np.stack(centers, axis=0)


def _resample_polyline(points: np.ndarray, num_points: int) -> np.ndarray:
    """Resample an ordered ``(M, 3)`` polyline to exactly ``num_points``."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if points.shape[0] == 1:
        return np.repeat(points, num_points, axis=0).astype(np.float32)
    src = np.linspace(0.0, 1.0, points.shape[0])
    dst = np.linspace(0.0, 1.0, num_points)
    out = np.stack([np.interp(dst, src, points[:, c]) for c in range(3)], axis=-1)
    return out.astype(np.float32)


def _empty_wireframe(vertices: np.ndarray, num_per_edge: int) -> dict[str, np.ndarray]:
    return {
        "vertices": vertices.astype(np.float32),
        "edge_index": np.zeros((0, 2), dtype=np.int64),
        "edge_points": np.zeros((0, num_per_edge, 3), dtype=np.float32),
    }


def reconstruct_wireframe(
    points: Any,
    *,
    type_threshold: float = 0.5,
    merge_radius: float = 0.03,
    min_votes: int = 3,
    num_per_edge: int = 32,
) -> dict[str, np.ndarray]:
    """Reconstruct a wireframe from a generated ``(N, 4)`` point set.

    Args:
        points: ``(N, 4)`` array / tensor of ``(x, y, z, type)``.
        type_threshold: ``type >= threshold`` -> vertex point, else edge point.
        merge_radius: radius for vertex-point cluster merging.
        min_votes: minimum nearest-two votes for a vertex pair to become edge.
        num_per_edge: points per reconstructed edge curve.

    Returns:
        ``{vertices (V,3), edge_index (E,2), edge_points (E,P,3)}`` (numpy).
    """
    from scipy.spatial import cKDTree

    pts = _to_numpy(points)
    is_vertex = pts[:, 3] >= float(type_threshold)
    vertex_pts = pts[is_vertex, :3]
    edge_pts = pts[~is_vertex, :3]

    vertices = _cluster_radius(vertex_pts, float(merge_radius))
    if vertices.shape[0] < 2 or edge_pts.shape[0] == 0:
        return _empty_wireframe(vertices, num_per_edge)

    # Nearest-two-vertices voting.
    tree = cKDTree(vertices)
    _, nn_idx = tree.query(edge_pts, k=2)
    nn_idx = np.asarray(nn_idx, dtype=np.int64).reshape(-1, 2)
    lo = np.minimum(nn_idx[:, 0], nn_idx[:, 1])
    hi = np.maximum(nn_idx[:, 0], nn_idx[:, 1])
    valid = lo != hi

    votes: dict[tuple[int, int], list[int]] = {}
    for k in np.nonzero(valid)[0]:
        pair = (int(lo[k]), int(hi[k]))
        votes.setdefault(pair, []).append(int(k))

    edge_index: list[tuple[int, int]] = []
    edge_curves: list[np.ndarray] = []
    for pair, members in sorted(votes.items()):
        if len(members) < int(min_votes):
            continue
        i, j = pair
        a, b = vertices[i], vertices[j]
        support = edge_pts[np.asarray(members, dtype=np.int64)]
        # Order the supporting points along the a->b direction.
        ab = b - a
        denom = float(ab @ ab)
        if denom <= 1e-12:
            curve = _resample_polyline(np.stack([a, b], axis=0), num_per_edge)
        else:
            proj = (support - a) @ ab / denom
            ordered = support[np.argsort(proj, kind="stable")]
            # Pin the endpoints so the curve spans the actual vertices.
            poly = np.concatenate(
                [a[None, :], ordered, b[None, :]], axis=0)
            curve = _resample_polyline(poly, num_per_edge)
        edge_index.append((i, j))
        edge_curves.append(curve)

    if not edge_index:
        return _empty_wireframe(vertices, num_per_edge)

    return {
        "vertices": vertices.astype(np.float32),
        "edge_index": np.asarray(edge_index, dtype=np.int64),
        "edge_points": np.stack(edge_curves, axis=0).astype(np.float32),
    }


__all__ = ["reconstruct_wireframe"]
