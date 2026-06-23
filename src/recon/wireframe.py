"""Decode WireframeAE outputs into an explicit wireframe.

The decoder has already produced, for one shape:

  * a set of **alive vertices** (alive logit > threshold), each with an ``xyz``;
  * for every retained vertex *pair* ``(i, j)``: an edge ``exist`` probability, a
    curve ``type`` (0=line / 1=arc / 2=bezier) and the two interior anchors
    ``q1`` / ``q2``.

Decoding is then pure book-keeping: keep the pairs whose ``exist`` probability
clears ``edge_thresh`` and sample each retained edge's curve from its
``(a, q1, q2, b)`` control points (``a, b`` = the pair's vertex coordinates)
with the shared curve samplers in :mod:`src.models.curves`.

Output matches the GT schema used by :mod:`src.metrics`::

    vertices:    (V, 3)    float32
    edge_index:  (E, 2)    int64
    edge_points: (E, P, 3) float32
"""
from __future__ import annotations

import numpy as np
import torch

from ..models.curves import sample_curve_by_type


def _empty_wireframe(
    vertices: np.ndarray, num_per_edge: int
) -> dict[str, np.ndarray]:
    return {
        "vertices": np.asarray(vertices, dtype=np.float32).reshape(-1, 3),
        "edge_index": np.zeros((0, 2), dtype=np.int64),
        "edge_points": np.zeros((0, num_per_edge, 3), dtype=np.float32),
    }


def decode_wireframe(
    out: dict[str, np.ndarray],
    *,
    edge_thresh: float = 0.5,
    num_per_edge: int = 32,
) -> dict[str, np.ndarray]:
    """Assemble ``{vertices, edge_index, edge_points}`` from decoder fields.

    Args:
        out: per-sample numpy fields::

            vertices   (V, 3)   alive vertex coordinates
            pair_index (P, 2)   vertex-id pairs scored by the edge head (i < j)
            edge_prob  (P,)     edge existence probability
            edge_type  (P,)     argmax curve type (0/1/2)
            q1         (P, 3)   t=1/3 anchor
            q2         (P, 3)   t=2/3 anchor
        edge_thresh: keep a pair iff ``edge_prob >= edge_thresh``.
        num_per_edge: samples per reconstructed edge curve.
    """
    vertices = np.asarray(out["vertices"], dtype=np.float32).reshape(-1, 3)
    if vertices.shape[0] < 2:
        return _empty_wireframe(vertices, num_per_edge)

    pair_index = np.asarray(out["pair_index"], dtype=np.int64).reshape(-1, 2)
    edge_prob = np.asarray(out["edge_prob"], dtype=np.float64).reshape(-1)
    if pair_index.shape[0] == 0:
        return _empty_wireframe(vertices, num_per_edge)

    keep = edge_prob >= float(edge_thresh)
    if not np.any(keep):
        return _empty_wireframe(vertices, num_per_edge)

    pair_index = pair_index[keep]
    edge_type = np.asarray(out["edge_type"], dtype=np.int64).reshape(-1)[keep]
    q1 = np.asarray(out["q1"], dtype=np.float32).reshape(-1, 3)[keep]
    q2 = np.asarray(out["q2"], dtype=np.float32).reshape(-1, 3)[keep]

    a = vertices[pair_index[:, 0]]
    b = vertices[pair_index[:, 1]]
    curves = sample_curve_by_type(
        torch.from_numpy(a),
        torch.from_numpy(q1),
        torch.from_numpy(q2),
        torch.from_numpy(b),
        torch.from_numpy(edge_type),
        int(num_per_edge),
    ).numpy().astype(np.float32)

    return {
        "vertices": vertices,
        "edge_index": pair_index.astype(np.int64),
        "edge_points": curves,
    }


__all__ = ["decode_wireframe"]
