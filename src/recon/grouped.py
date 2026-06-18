"""Decode the stage-2 grouper's per-point fields into an explicit wireframe.

The network has already regressed connectivity / ordering / curve type, so
decoding is parametric book-keeping (no ``min_votes`` / chord-projection):

  1. **Vertices** = cluster *all* voted endpoints ``xyz + endpoint_offset`` (both
     of each point's two votes, i.e. ``2N`` points) with
     ``sklearn.cluster.DBSCAN``; each cluster centroid is a vertex (more robust
     than a separate vertex-point head).
  2. **Connectivity** = for each edge point, snap its two voted endpoints to the
     nearest vertices (``scipy`` KD-tree) to get an unordered vertex pair
     ``(va, vb)``; group edge points by that pair.
  3. **Splitting** = within a pair, optionally split by the instance
     ``embedding`` (DBSCAN) so that two edges sharing both endpoints (e.g. the
     two arcs of a split circle) become separate edges.
  4. **Curves** = per edge: the two endpoints are the snapped vertices, the two
     interior anchors are aggregated from the members' ``xyz + anchor`` votes,
     the curve type is a member majority vote, and the ``(a, q1, q2, b)`` tuple
     is parsed (line / arc / bezier) and sampled to ``num_per_edge`` points.

Output matches the GT schema used by :mod:`src.metrics`::

    vertices:    (V, 3)    float32
    edge_index:  (E, 2)    int64
    edge_points: (E, P, 3) float32
"""
from __future__ import annotations

import numpy as np

# Curve-type codes (in sync with the dataset labeller + grouper head).
_CURVE_LINE = 0
_CURVE_ARC = 1
_CURVE_BEZIER = 2


# ----------------------------------------------------------------------
# Numpy curve samplers (mirror the torch ones in models/wireframe_grouper.py).
# Each takes single-edge control points ``(3,)`` and returns ``(num, 3)``.
# ----------------------------------------------------------------------
def _np_line(a: np.ndarray, b: np.ndarray, num: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num)[:, None]
    return a[None, :] * (1.0 - t) + b[None, :] * t


def _np_bezier(
    a: np.ndarray, q1: np.ndarray, q2: np.ndarray, b: np.ndarray, num: int
) -> np.ndarray:
    big_a = 27.0 * q1 - 8.0 * a - b
    big_b = 27.0 * q2 - a - 8.0 * b
    p1 = (2.0 * big_a - big_b) / 18.0
    p2 = (2.0 * big_b - big_a) / 18.0
    t = np.linspace(0.0, 1.0, num)[:, None]
    mt = 1.0 - t
    return (
        mt ** 3 * a[None, :]
        + 3.0 * mt ** 2 * t * p1[None, :]
        + 3.0 * mt * t ** 2 * p2[None, :]
        + t ** 3 * b[None, :]
    )


def _np_arc(a: np.ndarray, m: np.ndarray, b: np.ndarray, num: int) -> np.ndarray:
    eps = 1e-8
    aa = a - m
    bb = b - m
    cr = np.cross(aa, bb)
    cr_n2 = float(cr @ cr)
    if cr_n2 < eps:
        return _np_line(a, b, num)
    alpha = float(aa @ aa)
    beta = float(bb @ bb)
    center = m + np.cross(alpha * bb - beta * aa, cr) / (2.0 * cr_n2)
    ua = a - center
    r = float(np.linalg.norm(ua))
    if r < eps:
        return _np_line(a, b, num)
    u = ua / r
    nrm = cr / np.linalg.norm(cr)
    v = np.cross(nrm, u)

    def _ang(p: np.ndarray) -> float:
        d = p - center
        return float(np.arctan2(d @ v, d @ u))

    two_pi = 2.0 * np.pi
    m_ang = _ang(m) % two_pi
    b_ang = _ang(b) % two_pi
    sweep = b_ang if m_ang <= b_ang else b_ang - two_pi
    theta = (np.linspace(0.0, 1.0, num) * sweep)[:, None]
    return center[None, :] + r * (np.cos(theta) * u[None, :] + np.sin(theta) * v[None, :])


def _np_curve(
    a: np.ndarray, q1: np.ndarray, q2: np.ndarray, b: np.ndarray,
    ctype: int, num: int,
) -> np.ndarray:
    if ctype == _CURVE_ARC:
        return _np_arc(a, q1, b, num).astype(np.float32)
    if ctype == _CURVE_BEZIER:
        return _np_bezier(a, q1, q2, b, num).astype(np.float32)
    return _np_line(a, b, num).astype(np.float32)


def _empty_wireframe(
    vertices: np.ndarray, num_per_edge: int
) -> dict[str, np.ndarray]:
    return {
        "vertices": vertices.astype(np.float32),
        "edge_index": np.zeros((0, 2), dtype=np.int64),
        "edge_points": np.zeros((0, num_per_edge, 3), dtype=np.float32),
    }


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
            ``xyz (N,3)``, ``endpoint_offset (N,2,3)``, ``embedding (N,D)``,
            ``curve_type (N,3)`` (logits), ``anchor (N,2,3)``. Curve ordering /
            q1-q2 assignment is geometric (projection on the a->b axis), so the
            grouper's ``arclen`` head is an auxiliary training signal only.
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
    n = xyz.shape[0]
    eoff = np.asarray(fields["endpoint_offset"], dtype=np.float64).reshape(n, 2, 3)
    emb = np.asarray(fields["embedding"], dtype=np.float64).reshape(n, -1)
    ctype = np.asarray(fields["curve_type"], dtype=np.float64).reshape(n, 3)
    anchor = np.asarray(fields["anchor"], dtype=np.float64).reshape(n, 2, 3)

    if merge_relative:
        merge_eps = float(vertex_merge_radius) * max(_bbox_diag(xyz), 1e-9)
    else:
        merge_eps = float(vertex_merge_radius)

    # 1) Vertices = cluster all 2N voted endpoints.
    pa = xyz + eoff[:, 0, :]
    pb = xyz + eoff[:, 1, :]
    votes = np.concatenate([pa, pb], axis=0)
    vertices = _cluster_dbscan(votes, merge_eps)
    if vertices.shape[0] < 2:
        return _empty_wireframe(vertices, num_per_edge)

    # 2) Connectivity = snap each point's two endpoints to the nearest vertices.
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

    ctype_pt = ctype.argmax(axis=1)  # (N,) per-point class

    edge_index: list[tuple[int, int]] = []
    edge_curves: list[np.ndarray] = []

    def _emit(i: int, j: int, members: np.ndarray) -> None:
        if members.shape[0] < int(min_edge_points):
            return
        a, b = vertices[i], vertices[j]
        # Curve type = member majority vote.
        cls = ctype_pt[members]
        ctype_edge = int(np.bincount(cls, minlength=3).argmax())
        # Anchors = aggregate the members' two anchor votes, assigning each to
        # the q1 (near a) / q2 (near b) third by its projection on the a->b axis.
        av = np.concatenate(
            [xyz[members] + anchor[members, 0, :],
             xyz[members] + anchor[members, 1, :]], axis=0)   # (2m, 3)
        ab = b - a
        denom = float(ab @ ab)
        if denom < 1e-12:
            q1 = q2 = 0.5 * (a + b)
        else:
            s = (av - a[None, :]) @ ab / denom
            near_a = av[s < 0.5]
            near_b = av[s >= 0.5]
            q1 = near_a.mean(axis=0) if near_a.shape[0] else a + ab / 3.0
            q2 = near_b.mean(axis=0) if near_b.shape[0] else a + 2.0 * ab / 3.0
        curve = _np_curve(a, q1, q2, b, ctype_edge, int(num_per_edge))
        edge_index.append((i, j))
        edge_curves.append(curve)

    for (i, j), members in sorted(groups.items()):
        members = np.asarray(members, dtype=np.int64)
        if split_by_embedding and members.shape[0] >= 2:
            from sklearn.cluster import DBSCAN

            sub = DBSCAN(eps=float(embed_eps), min_samples=1).fit_predict(
                emb[members])
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
