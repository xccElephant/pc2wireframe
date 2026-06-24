"""Endpoint-aggregation reconstruction of an edge-centric wireframe.

The edge-set decoder has produced, for one shape, a set of ``Q`` edge queries --
each an existence probability + ``P`` ordered world-space sample points whose
first / last points are the two endpoints. This module turns that edge set into
a connected wireframe ``{vertices, edge_index, edge_points}`` by **aggregating
endpoints into shared vertices**:

1. keep the edges whose existence probability clears ``edge_threshold`` (and,
   optionally, only the ``topk_edges`` most-confident);
2. collect the ``2E`` endpoints (``pts[0]`` / ``pts[-1]`` per kept edge) and
   **union-find merge** any two endpoints closer than ``tau_merge``; each
   cluster's (existence-weighted) mean is a shared vertex;
3. rebuild ``edge_index`` onto the merged vertices, dropping **self-loops**
   (both endpoints merged together) and **duplicate edges** (keep the most
   confident);
4. **pin** every kept curve's first / last point to its merged vertex and blend
   the shift linearly across the interior points, so the curve stays attached to
   the final shared vertices without tearing.

Output matches the GT schema used by :mod:`src.metrics`::

    vertices:    (V, 3)    float32
    edge_index:  (E, 2)    int64
    edge_points: (E, P, 3) float32
"""
from __future__ import annotations

import numpy as np


def _empty_wireframe(num_per_edge: int) -> dict[str, np.ndarray]:
    return {
        "vertices": np.zeros((0, 3), dtype=np.float32),
        "edge_index": np.zeros((0, 2), dtype=np.int64),
        "edge_points": np.zeros((0, num_per_edge, 3), dtype=np.float32),
    }


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def aggregate_wireframe(
    edge_points: np.ndarray,
    edge_prob: np.ndarray,
    *,
    edge_threshold: float = 0.5,
    tau_merge: float = 0.015,
    topk_edges: int = 0,
    num_per_edge: int = 32,
) -> dict[str, np.ndarray]:
    """Aggregate per-edge curves into a connected wireframe.

    Args:
        edge_points: ``(Q, P, 3)`` ordered curve samples per edge query.
        edge_prob: ``(Q,)`` edge existence probability.
        edge_threshold: keep an edge iff ``edge_prob >= edge_threshold``.
        tau_merge: union-find endpoint merge radius (shared-vertex tolerance).
        topk_edges: among the kept edges, retain only the ``topk_edges`` most
            confident (``0`` = no cap).
        num_per_edge: ``P`` (used only to shape an empty result).
    """
    edge_points = np.asarray(edge_points, dtype=np.float32)
    edge_prob = np.asarray(edge_prob, dtype=np.float64).reshape(-1)
    if edge_points.ndim != 3 or edge_points.shape[0] == 0:
        return _empty_wireframe(num_per_edge)
    p = edge_points.shape[1]

    keep = edge_prob >= float(edge_threshold)
    if not np.any(keep):
        return _empty_wireframe(p)
    kept = np.nonzero(keep)[0]
    if topk_edges and kept.shape[0] > int(topk_edges):
        order = np.argsort(edge_prob[kept])[::-1][: int(topk_edges)]
        kept = kept[order]

    curves = edge_points[kept]                       # (E, P, 3)
    probs = edge_prob[kept]                           # (E,)
    e = curves.shape[0]

    # 2E endpoints: [v1_0, v2_0, v1_1, v2_1, ...].
    endpoints = np.stack([curves[:, 0], curves[:, -1]], axis=1).reshape(-1, 3)
    n_ep = endpoints.shape[0]

    # Union-find merge endpoints within tau_merge (O((2E)^2) on the kept set).
    uf = _UnionFind(n_ep)
    tau = float(tau_merge)
    if tau > 0.0 and n_ep > 1:
        dist = np.linalg.norm(
            endpoints[:, None, :] - endpoints[None, :, :], axis=-1)
        iu, ju = np.nonzero(np.triu(dist <= tau, k=1))
        for a, bb in zip(iu.tolist(), ju.tolist()):
            uf.union(a, bb)

    # Cluster -> shared vertex (existence-weighted mean of its endpoints).
    ep_weight = np.repeat(probs, 2).clip(min=1e-6)    # (2E,)
    roots = np.array([uf.find(i) for i in range(n_ep)], dtype=np.int64)
    uniq_roots, inv = np.unique(roots, return_inverse=True)
    n_v = uniq_roots.shape[0]
    vsum = np.zeros((n_v, 3), dtype=np.float64)
    wsum = np.zeros((n_v,), dtype=np.float64)
    np.add.at(vsum, inv, endpoints * ep_weight[:, None])
    np.add.at(wsum, inv, ep_weight)
    vertices = (vsum / wsum[:, None].clip(min=1e-9)).astype(np.float32)

    # Per-edge endpoint vertex ids (endpoint 2k = v1, 2k+1 = v2 of edge k).
    vid = inv.reshape(e, 2)                           # (E, 2)

    # Drop self-loops, dedup undirected edges (keep highest prob), pin curves.
    best: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
    for k in range(e):
        a, b = int(vid[k, 0]), int(vid[k, 1])
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        prev = best.get(key)
        if prev is not None and prev[0] >= probs[k]:
            continue
        # Pin the curve's endpoints to the merged vertices and blend the shift
        # linearly across the interior points.
        curve = curves[k].astype(np.float64)
        d_v1 = vertices[a].astype(np.float64) - curve[0]
        d_v2 = vertices[b].astype(np.float64) - curve[-1]
        t = np.linspace(0.0, 1.0, p)[:, None]
        pinned = curve + (1.0 - t) * d_v1[None, :] + t * d_v2[None, :]
        best[key] = (float(probs[k]), pinned.astype(np.float32))

    if not best:
        return {
            "vertices": vertices,
            "edge_index": np.zeros((0, 2), dtype=np.int64),
            "edge_points": np.zeros((0, p, 3), dtype=np.float32),
        }

    edge_index = np.array(list(best.keys()), dtype=np.int64).reshape(-1, 2)
    edge_curves = np.stack([v[1] for v in best.values()], axis=0)

    # Drop now-unreferenced vertices and reindex (keeps VPE honest).
    used = np.unique(edge_index.reshape(-1))
    remap = -np.ones((vertices.shape[0],), dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    vertices = vertices[used]
    edge_index = remap[edge_index]

    return {
        "vertices": vertices.astype(np.float32),
        "edge_index": edge_index.astype(np.int64),
        "edge_points": edge_curves.astype(np.float32),
    }


__all__ = ["aggregate_wireframe"]
