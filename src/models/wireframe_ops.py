"""Wireframe topology ops: canonical ordering <-> CLR-Wire differential
adjacency, and the inverse reconstruction.

CLR-Wire encodes adjacency as per-curve ``(col_diff, row_diff)`` over a
*canonically ordered* edge list:

  * vertices are renumbered in a BFS order (keeps connected vertices close, so
    ``row_diff`` stays small);
  * each edge is oriented ``(min, max)`` and the edge list is sorted
    lexicographically by ``(first, second)``;
  * ``col_diff = diff(first, prepend=first[0])`` (increment of the smaller
    endpoint between consecutive edges, expected in ``[0, max_col_diff)``);
  * ``row_diff = clip(second - first - 1, 0, None)`` (gap between endpoints,
    expected in ``[0, max_row_diff)``).

NOTE (AICAD caveat): this canonicalisation does not *guarantee* the diffs stay
within range for arbitrary wireframes. Out-of-range samples should be filtered
or the diff ranges widened in the data prep. The values are clamped here as a
safety net.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def compute_diffs(lines: np.ndarray) -> np.ndarray:
    """``(E,2)`` sorted oriented adjacency -> ``(E,2)`` ``(col_diff,row_diff)``."""
    col_diff = np.diff(lines[:, 0], prepend=lines[0, 0])
    row_diff = np.clip(lines[:, 1] - lines[:, 0] - 1, 0, None)
    return np.stack([col_diff, row_diff], axis=1)


def bfs_vertex_order(num_vertices: int, edges: np.ndarray) -> np.ndarray:
    """Return ``perm`` mapping old vertex id -> new (BFS) id.

    Multiple connected components are visited in turn (lowest unvisited id as
    each component's root), so isolated vertices still get an index.
    """
    adj: list[list[int]] = [[] for _ in range(num_vertices)]
    for u, v in edges:
        u, v = int(u), int(v)
        adj[u].append(v)
        adj[v].append(u)

    new_id = np.full(num_vertices, -1, dtype=np.int64)
    counter = 0
    for root in range(num_vertices):
        if new_id[root] != -1:
            continue
        queue = deque([root])
        new_id[root] = counter
        counter += 1
        while queue:
            cur = queue.popleft()
            # Visit neighbours by ascending degree-then-id for a stable order.
            for nb in sorted(adj[cur]):
                if new_id[nb] == -1:
                    new_id[nb] = counter
                    counter += 1
                    queue.append(nb)
    return new_id


def canonicalize(
    num_vertices: int,
    edge_index: np.ndarray,
    *,
    max_col_diff: int = 6,
    max_row_diff: int = 32,
) -> dict:
    """Canonically order a wireframe and compute its differential adjacency.

    Args:
        num_vertices: number of vertices.
        edge_index: ``(E, 2)`` local vertex ids.

    Returns dict with:
        ``perm``        old->new vertex id (``(V,)``)
        ``adj``         ``(E,2)`` oriented + sorted adjacency in new ids
        ``order``       permutation of edges applied (``(E,)``) for reordering
                        the per-edge attributes (endpoints / curves) to match
        ``diffs``       ``(E,2)`` clamped ``(col_diff, row_diff)``
    """
    edge_index = np.asarray(edge_index, dtype=np.int64).reshape(-1, 2)
    perm = bfs_vertex_order(num_vertices, edge_index)

    relabelled = perm[edge_index]
    # orient (min, max)
    oriented = np.sort(relabelled, axis=1)
    # stable lexsort by (first, second)
    order = np.lexsort((oriented[:, 1], oriented[:, 0]))
    adj = oriented[order]

    diffs = compute_diffs(adj)
    diffs[:, 0] = np.clip(diffs[:, 0], 0, max_col_diff - 1)
    diffs[:, 1] = np.clip(diffs[:, 1], 0, max_row_diff - 1)

    return {"perm": perm, "adj": adj, "order": order, "diffs": diffs}


def diffs_to_adjacency(col_diff: np.ndarray, row_diff: np.ndarray) -> np.ndarray:
    """Inverse of :func:`compute_diffs`: cumsum diffs -> ``(E,2)`` adjacency.

    ``row_diff`` here is the *already +1* gap (matching CLR-Wire reconstruction,
    which uses ``argmax + 1`` for the row logits).
    """
    first_col = np.cumsum(col_diff, axis=-1)
    second_col = first_col + row_diff
    return np.stack([first_col, second_col], axis=-1)


def refine_segment_coords_by_adj(
    adj: np.ndarray, segment_coords: np.ndarray
) -> np.ndarray:
    """Average each shared vertex's incident endpoints (CLR-Wire trick).

    Args:
        adj: ``(E, 2)`` vertex ids per edge.
        segment_coords: ``(E, 6)`` predicted endpoint coords.

    Returns ``(E, 6)`` with each vertex's position replaced by the mean over
    all incident edges, enforcing exact shared endpoints.
    """
    node_coords: dict[int, list[np.ndarray]] = {}
    for i, (a, b) in enumerate(adj):
        node_coords.setdefault(int(a), []).append(segment_coords[i, :3])
        node_coords.setdefault(int(b), []).append(segment_coords[i, 3:])

    avg = {n: np.mean(np.array(c), axis=0) for n, c in node_coords.items()}

    out = np.empty_like(segment_coords)
    for i, (a, b) in enumerate(adj):
        out[i, :3] = avg[int(a)]
        out[i, 3:] = avg[int(b)]
    return out


__all__ = [
    "compute_diffs",
    "bfs_vertex_order",
    "canonicalize",
    "diffs_to_adjacency",
    "refine_segment_coords_by_adj",
]
