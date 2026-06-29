"""Wireframe reconstruction from the edge-set decoder predictions.

The edge-set decoder predicts, for one shape: a per-edge existence confidence,
two endpoints and a curve-VAE latent (already decoded here into a canonical
curve). This module assembles those into an explicit wireframe
``{vertices, edge_index, edge_points}`` by *merging* the free endpoints into a
shared vertex set (there is no separate vertex head):

1. keep the edges whose confidence clears ``ethr`` (fall back to the top
   ``min_edges`` so the result is never degenerate);
2. orient each kept edge by a deterministic lexicographic endpoint rule;
3. **confidence-weighted iterative vertex merge** -- repeatedly union the two
   nearest endpoints within ``merge_tol`` (merged vertex = confidence-weighted
   centroid); the two endpoints of the *same* edge are never merged together;
4. drop self-loops, then run an **edge E-NMS** that suppresses near-duplicate
   curves sharing a vertex pair (keeping up to two geometrically distinct
   parallel arcs, so a split closed loop survives);
5. (optional) prune dangling degree-1 edges whose free endpoint is not a real
   junction;
6. ``denorm`` the decoded canonical curves onto the merged endpoints and
   reindex.

Output matches the GT schema used by :mod:`src.metrics`::

    vertices:    (V, 3)    float32
    edge_index:  (E, 2)    int64
    edge_points: (E, P, 3) float32
"""
from __future__ import annotations

import numpy as np

from ..models.vae.recon_utils import denorm_curves

# Up to this many parallel edges are kept per vertex pair (2 => a split closed
# loop's two arcs survive).
_MAX_PARALLEL_EDGES = 2


def _empty_wireframe(num_per_edge: int) -> dict[str, np.ndarray]:
    return {
        "vertices": np.zeros((0, 3), dtype=np.float32),
        "edge_index": np.zeros((0, 2), dtype=np.int64),
        "edge_points": np.zeros((0, num_per_edge, 3), dtype=np.float32),
    }


def _keep_by_threshold(prob: np.ndarray, thr: float, floor: int) -> np.ndarray:
    """Indices with ``prob >= thr``; fall back to the top-``floor`` most likely."""
    kept = np.nonzero(prob >= float(thr))[0]
    floor = max(0, int(floor))
    if kept.shape[0] < floor:
        kept = np.argsort(prob)[::-1][: min(floor, prob.shape[0])]
    return kept


def _lex_less(a: np.ndarray, b: np.ndarray) -> bool:
    """Lexicographic ``a < b`` for two 3-D points."""
    for k in range(3):
        if a[k] < b[k]:
            return True
        if a[k] > b[k]:
            return False
    return False


def _straight_curve(c0: np.ndarray, c1: np.ndarray, p: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, p)[:, None]
    return (c0[None, :] * (1.0 - t) + c1[None, :] * t).astype(np.float32)


# ----------------------------------------------------------------------
# confidence-weighted iterative endpoint merge (with same-edge constraint)
# ----------------------------------------------------------------------
def _merge_endpoints(
    points: np.ndarray,        # (2E, 3) endpoints, edge e -> rows [2e, 2e+1]
    weights: np.ndarray,       # (2E,) per-endpoint confidence weight
    merge_tol: float,
) -> np.ndarray:
    """Union endpoints within ``merge_tol`` -> cluster id per endpoint ``(2E,)``.

    Greedy on increasing pair distance; two endpoints of the same edge are never
    placed in the same cluster (an edge must keep two distinct vertices).
    """
    n = points.shape[0]
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Edge id of each endpoint (rows 2e / 2e+1 belong to edge e).
    edge_of = np.arange(n) // 2
    cluster_edges: dict[int, set[int]] = {i: {int(edge_of[i])} for i in range(n)}

    # Candidate pairs within merge_tol, sorted by distance (skip same-edge pairs).
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    iu, ju = np.triu_indices(n, k=1)
    same_edge = edge_of[iu] == edge_of[ju]
    d = dist[iu, ju]
    keep = (d <= float(merge_tol)) & (~same_edge)
    iu, ju, d = iu[keep], ju[keep], d[keep]
    order = np.argsort(d)

    for k in order:
        i, j = int(iu[k]), int(ju[k])
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if cluster_edges[ri] & cluster_edges[rj]:
            continue  # would merge both endpoints of some edge
        # union (attach rj under ri) and merge their edge sets
        parent[rj] = ri
        cluster_edges[ri] |= cluster_edges[rj]
        del cluster_edges[rj]

    roots = np.array([find(i) for i in range(n)])
    _, cluster_id = np.unique(roots, return_inverse=True)
    return cluster_id


def _cluster_centroids(
    points: np.ndarray, weights: np.ndarray, cluster_id: np.ndarray
) -> np.ndarray:
    """Confidence-weighted centroid per cluster ``(V, 3)``."""
    v = int(cluster_id.max()) + 1 if cluster_id.size else 0
    out = np.zeros((v, 3), dtype=np.float64)
    wsum = np.zeros((v,), dtype=np.float64)
    np.add.at(out, cluster_id, points * weights[:, None])
    np.add.at(wsum, cluster_id, weights)
    wsum = np.clip(wsum, 1e-8, None)
    return (out / wsum[:, None]).astype(np.float32)


def assemble_wireframe(
    edge_prob: np.ndarray,         # (Ne,)
    endpoints: np.ndarray,         # (Ne, 2, 3) predicted endpoints
    canonical_curves: np.ndarray,  # (Ne, P, 3) decoded canonical curves
    *,
    ethr: float = 0.5,
    merge_tol: float = 0.04,
    min_edges: int = 1,
    num_per_edge: int = 32,
    enms_tol: float = 0.05,
    prune_dangling: bool = False,
) -> dict[str, np.ndarray]:
    """Assemble edge-set predictions into a wireframe via endpoint merging.

    Args:
        edge_prob: per-edge existence probability ``sigmoid(logit)``.
        endpoints: per-edge predicted endpoint pair.
        canonical_curves: per-edge decoded canonical curve (endpoints near
            ``[-1,0,0]`` / ``[1,0,0]``).
        ethr: edge existence threshold (top-``min_edges`` fallback).
        merge_tol: endpoints within this distance are merged into one vertex.
        min_edges: floor so a miscalibrated threshold never emits nothing.
        num_per_edge: ``P`` for empty results / straight fallbacks.
        enms_tol: edge E-NMS midpoint tolerance for near-duplicate suppression.
        prune_dangling: drop degree-1 edges whose free endpoint is not a junction.
    """
    edge_prob = np.asarray(edge_prob, dtype=np.float64).reshape(-1)
    endpoints = np.asarray(endpoints, dtype=np.float64).reshape(-1, 2, 3)
    canonical_curves = np.asarray(canonical_curves, dtype=np.float32)
    p = (canonical_curves.shape[1]
         if canonical_curves.ndim == 3 and canonical_curves.shape[1] > 0
         else int(num_per_edge))

    if endpoints.shape[0] == 0:
        return _empty_wireframe(p)

    kept = _keep_by_threshold(edge_prob, ethr, min_edges)
    if kept.shape[0] == 0:
        return _empty_wireframe(p)

    ep = endpoints[kept]                              # (E, 2, 3)
    conf = edge_prob[kept]                            # (E,)
    e = ep.shape[0]

    # ---- confidence-weighted iterative endpoint merge ----
    pts = ep.reshape(2 * e, 3)
    w = np.repeat(np.clip(conf, 1e-3, None), 2)
    cluster_id = _merge_endpoints(pts, w, merge_tol)
    vertices = _cluster_centroids(pts, w, cluster_id)
    edge_ids = cluster_id.reshape(e, 2)               # (E, 2) vertex ids

    # ---- per-edge curve (denorm canonical onto merged endpoints) ----
    order = np.argsort(conf)[::-1]                    # most confident first
    groups: dict[tuple[int, int], list[dict]] = {}
    for e_local in order:
        a_id, b_id = int(edge_ids[e_local, 0]), int(edge_ids[e_local, 1])
        if a_id == b_id:
            continue                                  # self-loop
        ca, cb = vertices[a_id], vertices[b_id]
        if _lex_less(ca, cb):
            corner = np.stack([ca, cb], axis=0)
            key = (a_id, b_id)
        else:
            corner = np.stack([cb, ca], axis=0)
            key = (b_id, a_id)

        curve = None
        if (canonical_curves.ndim == 3
                and e_local < canonical_curves.shape[0]
                and np.linalg.norm(corner[0] - corner[1]) > 1e-8):
            curve = denorm_curves(
                canonical_curves[e_local][None], corner[None])
        if curve is None or curve.shape[0] == 0:
            cur = _straight_curve(corner[0], corner[1], p)
        else:
            cur = curve[0].astype(np.float32)

        groups.setdefault(key, []).append(
            {"curve": cur, "conf": float(conf[e_local])})

    if not groups:
        return _empty_wireframe(p)

    # ---- edge E-NMS: suppress near-duplicate parallel arcs per vertex pair ----
    edge_index_list: list[tuple[int, int]] = []
    edge_curve_list: list[np.ndarray] = []
    for key, lst in groups.items():
        kept_curves: list[np.ndarray] = []
        for item in lst:                               # already conf-desc
            cur = item["curve"]
            mid = cur[p // 2]
            length = float(np.linalg.norm(cur[1:] - cur[:-1], axis=1).sum())
            dup = False
            for kc in kept_curves:
                if np.linalg.norm(kc[p // 2] - mid) < enms_tol:
                    kl = float(np.linalg.norm(kc[1:] - kc[:-1], axis=1).sum())
                    if abs(kl - length) <= 0.5 * max(kl, length, 1e-6):
                        dup = True
                        break
            if not dup and len(kept_curves) < _MAX_PARALLEL_EDGES:
                kept_curves.append(cur)
                edge_index_list.append(key)
                edge_curve_list.append(cur)

    if not edge_index_list:
        return _empty_wireframe(p)

    edge_index = np.array(edge_index_list, dtype=np.int64).reshape(-1, 2)
    edge_curves = np.stack(edge_curve_list, axis=0)

    if prune_dangling:
        edge_index, edge_curves = _prune_dangling(edge_index, edge_curves)
        if edge_index.shape[0] == 0:
            return _empty_wireframe(p)

    # ---- keep only referenced vertices + reindex ----
    used = np.unique(edge_index.reshape(-1))
    remap = -np.ones((vertices.shape[0],), dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    out_vertices = vertices[used].astype(np.float32)
    edge_index = remap[edge_index]

    return {
        "vertices": out_vertices,
        "edge_index": edge_index.astype(np.int64),
        "edge_points": edge_curves.astype(np.float32),
    }


def _prune_dangling(
    edge_index: np.ndarray, edge_curves: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop degree-1 edges whose free endpoint is not shared by any other edge.

    One pass: a vertex is a junction if its degree >= 2. An edge is dangling if
    *either* endpoint has degree 1 (a free-floating stub), so it is removed.
    """
    if edge_index.shape[0] == 0:
        return edge_index, edge_curves
    deg = np.bincount(edge_index.reshape(-1))
    keep = (deg[edge_index[:, 0]] >= 2) & (deg[edge_index[:, 1]] >= 2)
    if not keep.any():
        return edge_index, edge_curves        # never prune everything
    return edge_index[keep], edge_curves[keep]


__all__ = ["assemble_wireframe"]
