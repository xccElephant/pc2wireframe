"""Reconstruction for the joint vertex + edge decoder.

The joint decoder predicts, for one shape: per-vertex existence + coordinate,
per-edge existence + a curve VAE latent (already decoded here into a canonical
curve), and a soft edge->vertex association matrix ``A``. This module assembles
those into an explicit wireframe ``{vertices, edge_index, edge_points}`` without
any union-find merge -- the topology comes straight from ``A``:

1. keep the vertices whose existence probability clears ``vthr`` (fall back to
   the top-``min_vertices`` so the result is never degenerate);
2. keep the edges whose existence probability clears ``ethr`` (fall back to the
   top-``min_edges``);
3. for each kept edge, read its two endpoints as the **top-2** kept vertices
   under ``A[e, :]`` (forced distinct); collapse near-duplicate predictions but
   keep up to two geometrically distinct **parallel** edges per vertex pair (so a
   split closed loop's two arcs both survive);
4. orient each edge's endpoints by the same deterministic lexicographic rule used
   in training, then ``denorm`` the decoded canonical curve onto them (a straight
   segment as a safe fallback);
5. drop now-unreferenced vertices and reindex.

Output matches the GT schema used by :mod:`src.metrics`::

    vertices:    (V, 3)    float32
    edge_index:  (E, 2)    int64
    edge_points: (E, P, 3) float32
"""
from __future__ import annotations

import numpy as np

from ..models.vae.recon_utils import denorm_curves


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


# Up to this many parallel edges are kept per vertex pair (2 => a split closed
# loop's two arcs survive); curves whose midpoints are within _PARALLEL_TOL are
# treated as the same edge (a duplicate prediction, not a distinct arc).
_MAX_PARALLEL_EDGES = 2
_PARALLEL_TOL = 0.05


def assemble_wireframe(
    vertex_prob: np.ndarray,        # (Nv,)
    vertex_coord: np.ndarray,       # (Nv, 3)
    edge_prob: np.ndarray,          # (Ne,)
    assoc: np.ndarray,              # (Ne, Nv) in [0, 1]
    canonical_curves: np.ndarray,   # (Ne, P, 3) decoded canonical curves
    *,
    vthr: float = 0.5,
    ethr: float = 0.5,
    num_per_edge: int = 32,
    min_vertices: int = 2,
    min_edges: int = 1,
) -> dict[str, np.ndarray]:
    """Assemble joint-decoder fields into a wireframe via the association matrix.

    Args:
        vertex_prob / vertex_coord: per-vertex existence prob + coordinate.
        edge_prob: per-edge existence prob.
        assoc: soft edge->vertex incidence ``sigmoid(A_logit)``.
        canonical_curves: per-edge decoded canonical curve (endpoints near
            ``[-1,0,0]`` / ``[1,0,0]``).
        vthr / ethr: vertex / edge existence thresholds.
        num_per_edge: ``P`` (used to shape empty results / fallbacks).
        min_vertices / min_edges: floors so a miscalibrated threshold never emits
            a degenerate wireframe.
    """
    vertex_prob = np.asarray(vertex_prob, dtype=np.float64).reshape(-1)
    vertex_coord = np.asarray(vertex_coord, dtype=np.float64).reshape(-1, 3)
    edge_prob = np.asarray(edge_prob, dtype=np.float64).reshape(-1)
    assoc = np.asarray(assoc, dtype=np.float64)
    canonical_curves = np.asarray(canonical_curves, dtype=np.float32)
    p = (canonical_curves.shape[1]
         if canonical_curves.ndim == 3 and canonical_curves.shape[1] > 0
         else int(num_per_edge))

    if vertex_coord.shape[0] == 0 or edge_prob.shape[0] == 0:
        return _empty_wireframe(p)

    kept_v = _keep_by_threshold(vertex_prob, vthr, min_vertices)
    if kept_v.shape[0] < 2:
        return _empty_wireframe(p)
    kept_e = _keep_by_threshold(edge_prob, ethr, min_edges)
    if kept_e.shape[0] == 0:
        return _empty_wireframe(p)

    kv = kept_v                                    # global vertex ids (kept)
    vcoords = vertex_coord[kv]                      # (Vk, 3)

    # Each kept edge -> its top-2 kept vertices under A (forced distinct). Up to
    # _MAX_PARALLEL_EDGES edges per vertex pair are kept when their curves are
    # geometrically distinct (the two arcs of a split closed loop); processing
    # edges in descending confidence makes the kept ones the most confident and
    # collapses near-duplicate predictions.
    order = kept_e[np.argsort(edge_prob[kept_e])[::-1]]
    groups: dict[tuple[int, int], list[np.ndarray]] = {}
    for e in order:
        a_row = assoc[e, kv]                        # (Vk,)
        if a_row.shape[0] < 2:
            continue
        top2 = np.argsort(a_row)[::-1][:2]          # local kept-vertex ids
        la, lb = int(top2[0]), int(top2[1])
        if la == lb:
            continue
        ga, gb = int(kv[la]), int(kv[lb])
        key = (ga, gb) if ga < gb else (gb, ga)

        ca, cb = vcoords[la], vcoords[lb]
        # Orient endpoints by the training-time lexicographic rule so the decoded
        # canonical curve (start near [-1,0,0]) maps onto the right endpoint.
        if _lex_less(ca, cb):
            corner = np.stack([ca, cb], axis=0)
        else:
            corner = np.stack([cb, ca], axis=0)

        curve = None
        if (canonical_curves.ndim == 3 and e < canonical_curves.shape[0]
                and np.linalg.norm(corner[0] - corner[1]) > 1e-8):
            curve = denorm_curves(
                canonical_curves[e][None], corner[None])
        if curve is None or curve.shape[0] == 0:
            curve = _straight_curve(corner[0], corner[1], p)[None]
        cur = curve[0].astype(np.float32)

        lst = groups.setdefault(key, [])
        mid = cur[p // 2]
        is_dup = any(
            np.linalg.norm(c[p // 2] - mid) < _PARALLEL_TOL for c in lst)
        if not is_dup and len(lst) < _MAX_PARALLEL_EDGES:
            lst.append(cur)

    if not groups:
        return _empty_wireframe(p)

    edge_index_list: list[tuple[int, int]] = []
    edge_curve_list: list[np.ndarray] = []
    for key, lst in groups.items():
        for cur in lst:
            edge_index_list.append(key)
            edge_curve_list.append(cur)
    edge_index = np.array(edge_index_list, dtype=np.int64).reshape(-1, 2)
    edge_curves = np.stack(edge_curve_list, axis=0)

    # Map kept-global vertex ids onto a compact, only-referenced vertex set.
    used = np.unique(edge_index.reshape(-1))
    remap = -np.ones((vertex_coord.shape[0],), dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    vertices = vertex_coord[used].astype(np.float32)
    edge_index = remap[edge_index]

    return {
        "vertices": vertices,
        "edge_index": edge_index.astype(np.int64),
        "edge_points": edge_curves.astype(np.float32),
    }


__all__ = ["assemble_wireframe"]
