#!/usr/bin/env python3
"""Clean the (very noisy) PC2Wireframe *train* ground-truth wireframes.

The raw ``data/train/sample_edge/*.npz`` ground truth is extremely dirty: even
geometrically simple parts ship hundreds — sometimes thousands — of edges,
which blows past the loader's edge cap and silently drops the sample. The mess
is always the same few things, so the clean-up is deliberately simple:

1. **Weld duplicate vertices.** Endpoints sitting within a small tolerance are
   the same vertex. Two such vertices are merged *unless* a genuine edge runs
   between them (real arc length) — a tiny straight stub between them is just
   noise and collapses with them.
2. **Drop the junk the welding exposes.** After merging, duplicate edges,
   self-loops with no arc, and zero-arc-length edges are removed.
3. **Dissolve smooth degree-2 chains.** A vertex with exactly two incident
   edges that pass straight through it is a spurious subdivision point; the two
   edges are concatenated into one.
4. **Split spirals / zig-zags.** A polyline is cut at sharp interior corners
   (so a multi-segment zig-zag becomes separate edges) and on accumulated
   turning (so a spiral / closed loop becomes a few gentle, open arcs with a
   real endpoint chord).

Every surviving edge is then resampled to a fixed point count. The NPZ schema
is preserved exactly (``start_verts``, ``end_verts``, ``edge_points``,
``edge_us``, ``original_edge_indices``) so the cleaned files are a drop-in
replacement for the dataset / dataloader. Point clouds are untouched.

Usage
-----
Test a handful of shapes and dump a before/after visualization + stats::

    python scripts/clean_wireframe.py test --num 6 --pick worst
    python scripts/clean_wireframe.py test --files <a.npz> <b.npz>

Clean the whole train split into a new directory (parallelized)::

    python scripts/clean_wireframe.py all \
        --in-dir data/train/sample_edge \
        --out-dir data/train_clean/sample_edge \
        --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is expected to be present
    cKDTree = None


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class CleanConfig:
    """Knobs for the cleaning pipeline.

    Distances are a fraction of the shape's bounding-box diagonal, so the
    defaults are scale-invariant across parts. ``weld_tol`` for example is
    ``max(weld_abs, weld_rel * bbox_diag)``.
    """

    weld_rel: float = 5e-3       # weld vertices closer than 0.5% of bbox diag
    weld_abs: float = 0.0        # ... but never below this absolute distance
    dissolve: bool = True        # merge smooth degree-2 chains into one edge
    max_turn_deg: float = 15.0   # a chain is "smooth" while its turn stays below
    split: bool = True           # split spirals / zig-zags into simple arcs
    corner_deg: float = 45.0     # cut a polyline at interior kinks above this
    max_total_turn_deg: float = 180.0  # cut a curve every this much total turn
    min_arc_rel: float = 2e-3    # drop edges shorter than this (arc length)
    min_chord_rel: float = 1e-3  # drop edges whose endpoints ~coincide (chord)
    num_edge_points: int = 32    # resample every cleaned edge to this many pts


# ----------------------------------------------------------------------
# Small geometry helpers
# ----------------------------------------------------------------------
def _bbox_diag(points: np.ndarray) -> float:
    if points.size == 0:
        return 1.0
    ext = points.max(0) - points.min(0)
    return float(np.linalg.norm(ext)) or 1.0


def _polyline_length(poly: np.ndarray) -> float:
    if poly.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())


def _chord(poly: np.ndarray) -> float:
    if poly.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(poly[-1] - poly[0]))


def _resample_polyline(poly: np.ndarray, n: int) -> np.ndarray:
    """Resample an ordered polyline to ``n`` points by cumulative arc length."""
    poly = np.asarray(poly, dtype=np.float64)
    if poly.shape[0] == 0:
        return np.zeros((n, 3), dtype=np.float64)
    if poly.shape[0] == 1:
        return np.repeat(poly, n, axis=0)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-12:
        return np.repeat(poly[:1], n, axis=0)
    dst = np.linspace(0.0, total, n)
    return np.stack([np.interp(dst, cum, poly[:, c]) for c in range(3)], axis=-1)


def _tangent_away(poly: np.ndarray, min_dist: float) -> np.ndarray:
    """Unit direction leaving ``poly[0]``, robust to tiny first segments."""
    poly = np.asarray(poly, dtype=np.float64)
    if poly.shape[0] < 2:
        return np.zeros(3)
    origin = poly[0]
    far = poly[-1]
    for p in poly[1:]:
        if np.linalg.norm(p - origin) >= min_dist:
            far = p
            break
    d = far - origin
    nrm = np.linalg.norm(d)
    return d / nrm if nrm > 1e-12 else np.zeros(3)


def _turning_angles(poly: np.ndarray) -> np.ndarray:
    """Per-interior-point turning angle (rad) along an ordered polyline.

    Returns an array of length ``K-2``; entry ``i`` is the angle between the
    segment directions meeting at ``poly[i+1]``. Near-zero-length segments
    contribute a zero turn so duplicate samples do not inject spurious corners.
    """
    if poly.shape[0] < 3:
        return np.zeros(0)
    d = poly[1:] - poly[:-1]
    L = np.linalg.norm(d, axis=1, keepdims=True)
    t = d / np.maximum(L, 1e-12)
    cos = np.clip(np.einsum("ij,ij->i", t[:-1], t[1:]), -1.0, 1.0)
    ang = np.arccos(cos)
    seg_ok = (L[:-1, 0] > 1e-9) & (L[1:, 0] > 1e-9)
    return np.where(seg_ok, ang, 0.0)


# ----------------------------------------------------------------------
# 1. Vertex welding (merge duplicate vertices, protect real edges)
# ----------------------------------------------------------------------
def _weld_vertices(
    polys: list[np.ndarray], tol: float, protect_arc: float
) -> tuple[np.ndarray, np.ndarray]:
    """Union-find weld of edge endpoints within ``tol``.

    Two endpoints are merged when they are within ``tol`` *unless* they are the
    two ends of a single edge whose arc length is >= ``protect_arc`` — that is a
    genuine (small but real) edge and must not collapse to a point. A tiny
    straight stub (arc < ``protect_arc``) is allowed to collapse.

    Returns ``(ids, centroids)`` where ``ids`` has shape ``(E, 2)`` giving the
    welded vertex id of each edge's start/end and ``centroids`` are the
    per-cluster mean coordinates.
    """
    e = len(polys)
    pts = np.empty((2 * e, 3), dtype=np.float64)
    for i, p in enumerate(polys):
        pts[2 * i] = p[0]
        pts[2 * i + 1] = p[-1]

    parent = np.arange(2 * e)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    if tol > 0 and cKDTree is not None and e:
        tree = cKDTree(pts)
        for a, b in tree.query_pairs(tol, output_type="ndarray"):
            a, b = int(a), int(b)
            if a // 2 == b // 2:
                # the two ends of the same edge: only collapse a tiny stub
                if _polyline_length(polys[a // 2]) >= protect_arc:
                    continue
            union(a, b)

    remap: dict[int, int] = {}
    ids = np.empty(2 * e, dtype=np.int64)
    for i in range(2 * e):
        r = find(i)
        if r not in remap:
            remap[r] = len(remap)
        ids[i] = remap[r]

    nv = len(remap)
    centroids = np.zeros((nv, 3))
    counts = np.zeros(nv)
    np.add.at(centroids, ids, pts)
    np.add.at(counts, ids, 1.0)
    centroids /= np.maximum(counts[:, None], 1.0)
    return ids.reshape(e, 2), centroids


# ----------------------------------------------------------------------
# 2. Edge de-duplication
# ----------------------------------------------------------------------
def _orient(edge: dict[str, Any], first: int) -> np.ndarray:
    """Return the polyline oriented so it starts at vertex id ``first``."""
    poly = edge["poly"]
    return poly if edge["u"] == first else poly[::-1]


def _dedup_edges(
    edges: list[dict[str, Any]], geom_tol: float
) -> list[dict[str, Any]]:
    """Collapse edges sharing the same (undirected) endpoints + geometry."""
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for e in edges:
        key = (min(e["u"], e["v"]), max(e["u"], e["v"]))
        groups.setdefault(key, []).append(e)

    out: list[dict[str, Any]] = []
    for key, members in groups.items():
        kept: list[dict[str, Any]] = []
        for e in sorted(members, key=lambda x: -x["len"]):
            mid_e = _resample_polyline(_orient(e, key[0]), 8)
            if any(
                np.linalg.norm(mid_e - _resample_polyline(_orient(k, key[0]), 8),
                               axis=1).mean() < geom_tol
                for k in kept
            ):
                continue
            kept.append(e)
        out.extend(kept)
    return out


def _dedup_polys(
    edges: list[dict[str, Any]], tol: float
) -> list[dict[str, Any]]:
    """Drop duplicate arcs by quantized (endpoints + midpoint), keep longest.

    Splitting can re-create overlapping arcs (e.g. two source curves that share
    a boundary); this keeps the final edge set minimal.
    """
    if tol <= 0:
        return edges
    seen: set[tuple] = set()
    out: list[dict[str, Any]] = []
    for e in sorted(edges, key=lambda x: -x.get("len", 0.0)):
        p = e["poly"]
        a = tuple(np.round(p[0] / tol).astype(np.int64).tolist())
        b = tuple(np.round(p[-1] / tol).astype(np.int64).tolist())
        mid = tuple(np.round(p[len(p) // 2] / tol).astype(np.int64).tolist())
        key = (min(a, b), max(a, b), mid)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


# ----------------------------------------------------------------------
# 3. Dissolve smooth degree-2 chains
# ----------------------------------------------------------------------
def _dissolve_chains(
    edges: list[dict[str, Any]],
    num_vertices: int,
    max_turn_deg: float,
    weld_tol: float,
) -> list[dict[str, Any]]:
    """Merge edges across smooth degree-2 vertices into single polylines.

    A degree-2 vertex (exactly two incident edges) that is *smooth* — the two
    edges pass through it with a turn below ``max_turn_deg`` — is a spurious
    subdivision point and gets dissolved, concatenating the two polylines. Real
    junctions (degree != 2) and sharp corners are preserved.
    """
    cos_thresh = np.cos(np.deg2rad(180.0 - max_turn_deg))
    adj: list[list[int]] = [[] for _ in range(num_vertices)]
    for ei, e in enumerate(edges):
        adj[e["u"]].append(ei)
        if e["v"] != e["u"]:
            adj[e["v"]].append(ei)

    def is_smooth(vid: int, inc: list[int]) -> bool:
        e0, e1 = edges[inc[0]], edges[inc[1]]
        if e0 is None or e1 is None:
            return False
        d0 = _tangent_away(_orient_to(e0, vid), weld_tol)
        d1 = _tangent_away(_orient_to(e1, vid), weld_tol)
        if not d0.any() or not d1.any():
            return False
        # smooth pass-through => the two "away" directions are ~antiparallel
        return float(d0 @ d1) <= cos_thresh

    alive = [True] * len(edges)
    changed = True
    while changed:
        changed = False
        for vid in range(num_vertices):
            inc = [ei for ei in adj[vid] if alive[ei]]
            adj[vid] = inc
            if len(inc) != 2 or inc[0] == inc[1] or not is_smooth(vid, inc):
                continue
            ei0, ei1 = inc
            e0, e1 = edges[ei0], edges[ei1]
            a = e0["v"] if e0["u"] == vid else e0["u"]
            b = e1["v"] if e1["u"] == vid else e1["u"]
            if a == b:
                # merging would fold a 2-edge bigon into a self loop; leave it
                continue
            poly_a = _orient_to(e0, a)              # a ... vid
            poly_b = _orient_to(e1, vid)           # vid ... b
            merged = np.concatenate([poly_a, poly_b[1:]], axis=0)
            new_id = len(edges)
            edges.append({"u": a, "v": b, "poly": merged,
                          "len": _polyline_length(merged)})
            alive.append(True)
            alive[ei0] = alive[ei1] = False
            edges[ei0] = edges[ei1] = None  # type: ignore[assignment]
            adj[vid] = []
            adj[a] = [x for x in adj[a] if x not in (ei0, ei1)] + [new_id]
            adj[b] = [x for x in adj[b] if x not in (ei0, ei1)] + [new_id]
            changed = True

    return [e for e in edges if e is not None]


def _orient_to(edge: dict[str, Any], start_vid: int) -> np.ndarray:
    poly = edge["poly"]
    return poly if edge["u"] == start_vid else poly[::-1]


# ----------------------------------------------------------------------
# 4. Split spirals / zig-zags into simple arcs
# ----------------------------------------------------------------------
def _split_at_corners(poly: np.ndarray, corner_rad: float) -> list[np.ndarray]:
    """Split a polyline at interior points whose turn exceeds ``corner_rad``."""
    if poly.shape[0] < 3 or corner_rad <= 0:
        return [poly]
    ang = _turning_angles(poly)
    corners = (np.where(ang > corner_rad)[0] + 1).tolist()
    if not corners:
        return [poly]
    bounds = [0, *corners, poly.shape[0] - 1]
    return [poly[a:b + 1] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def _split_by_total_turn(poly: np.ndarray, turn_cap: float) -> list[np.ndarray]:
    """Cut a smooth curve every ``turn_cap`` radians of accumulated turning.

    This turns a spiral (or a closed loop, ~2*pi of turning) into the fewest
    near-equal-turn open arcs that each keep a real endpoint chord.
    """
    if poly.shape[0] < 3 or turn_cap <= 0:
        return [poly]
    cum = np.cumsum(_turning_angles(poly))   # cum[i] ~ turn up to poly[i+1]
    total = float(cum[-1])
    if total <= turn_cap:
        return [poly]
    n_pieces = int(np.ceil(total / turn_cap))
    step = total / n_pieces
    cuts: list[int] = []
    target = step
    for i, c in enumerate(cum):
        if len(cuts) >= n_pieces - 1:
            break
        if c >= target - 1e-9:
            idx = i + 1
            if not cuts or idx > cuts[-1]:
                cuts.append(idx)
            target += step
    bounds = [0, *cuts, poly.shape[0] - 1]
    return [poly[a:b + 1] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def _split_curve(
    poly: np.ndarray, corner_rad: float, turn_cap: float
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for piece in _split_at_corners(poly, corner_rad):
        out.extend(_split_by_total_turn(piece, turn_cap))
    return out


# ----------------------------------------------------------------------
# Core cleaning
# ----------------------------------------------------------------------
def _parse_edges(data: dict[str, Any]) -> list[np.ndarray]:
    """Extract per-edge ordered polylines (K,3) from a raw npz dict."""
    ep = data["edge_points"]
    polys: list[np.ndarray] = []
    for i in range(len(ep)):
        p = np.asarray(ep[i], dtype=np.float64).reshape(-1, 3)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        if p.shape[0] >= 2:
            polys.append(p)
    return polys


def clean_wireframe(
    data: dict[str, Any], cfg: CleanConfig
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Clean one wireframe npz dict. Returns ``(new_data, stats)``."""
    polys = _parse_edges(data)
    raw_edges = len(polys)
    if not polys:
        return _empty_npz(cfg.num_edge_points), _stats(0, 0, 0, 0, 0.0)

    raw_pts = np.concatenate([p[[0, -1]] for p in polys], 0)
    raw_verts = int(np.unique(np.round(raw_pts, 6), axis=0).shape[0])

    diag = _bbox_diag(np.concatenate(polys, axis=0))
    weld_tol = max(cfg.weld_abs, cfg.weld_rel * diag)
    min_arc = max(weld_tol, cfg.min_arc_rel * diag)
    min_chord = cfg.min_chord_rel * diag

    # --- 1. weld duplicate vertices (protect genuine edges) ----------------
    ids, centroids = _weld_vertices(polys, weld_tol, protect_arc=min_arc)
    edges: list[dict[str, Any]] = []
    for i, p in enumerate(polys):
        u, v = int(ids[i, 0]), int(ids[i, 1])
        p = p.copy()
        p[0] = centroids[u]
        p[-1] = centroids[v]
        arc = _polyline_length(p)
        # drop degenerate / zero-arc / collapsed-stub edges. Closed loops
        # (u == v) with real arc length survive and are split below.
        if arc < min_arc:
            continue
        edges.append({"u": u, "v": v, "poly": p, "len": arc})

    # --- 2. drop duplicate edges -------------------------------------------
    if edges:
        edges = _dedup_edges(edges, weld_tol)

    # --- 3. dissolve smooth degree-2 chains --------------------------------
    if cfg.dissolve and edges:
        edges = _dissolve_chains(
            edges, len(centroids), cfg.max_turn_deg, weld_tol)

    # --- 4. split spirals / zig-zags into simple arcs ----------------------
    if cfg.split and edges:
        corner_rad = np.deg2rad(cfg.corner_deg)
        turn_cap = np.deg2rad(cfg.max_total_turn_deg)
        split: list[dict[str, Any]] = []
        for e in edges:
            for sub in _split_curve(e["poly"], corner_rad, turn_cap):
                if sub.shape[0] < 2:
                    continue
                arc = _polyline_length(sub)
                if arc < min_arc or _chord(sub) < min_chord:
                    continue
                split.append({"poly": sub, "len": arc})
        edges = _dedup_polys(split, weld_tol)
    elif edges:
        # no splitting: still drop near-zero-chord (closed) leftovers
        edges = [e for e in edges if _chord(e["poly"]) >= min_chord]

    # --- 5. resample + repackage -------------------------------------------
    new = _pack_edges(edges, cfg.num_edge_points)
    n_edges = new["start_verts"].shape[0]
    stats = _stats(raw_verts, raw_edges,
                   _count_unique_verts(new) if n_edges else 0, n_edges, diag)
    return new, stats


# ----------------------------------------------------------------------
# Packaging
# ----------------------------------------------------------------------
def _pack_edges(edges: list[dict[str, Any]], n_pts: int) -> dict[str, np.ndarray]:
    if not edges:
        return _empty_npz(n_pts)
    polys = [_resample_polyline(e["poly"], n_pts).astype(np.float32)
             for e in edges]
    edge_points = np.empty(len(polys), dtype=object)
    for i, p in enumerate(polys):
        edge_points[i] = p
    start = np.stack([p[0] for p in polys]).astype(np.float64)
    end = np.stack([p[-1] for p in polys]).astype(np.float64)
    us_row = np.linspace(-1.0, 1.0, n_pts).astype(np.float32)
    edge_us = np.empty(len(polys), dtype=object)
    for i in range(len(polys)):
        edge_us[i] = us_row.copy()
    original = np.arange(len(polys), dtype=np.int64)
    return {
        "edge_points": edge_points,
        "edge_us": edge_us,
        "start_verts": start,
        "end_verts": end,
        "original_edge_indices": original,
    }


def _empty_npz(n_pts: int) -> dict[str, np.ndarray]:
    return {
        "edge_points": np.empty(0, dtype=object),
        "edge_us": np.empty(0, dtype=object),
        "start_verts": np.zeros((0, 3), dtype=np.float64),
        "end_verts": np.zeros((0, 3), dtype=np.float64),
        "original_edge_indices": np.zeros(0, dtype=np.int64),
    }


def _count_unique_verts(new: dict[str, np.ndarray]) -> int:
    pts = np.concatenate([new["start_verts"], new["end_verts"]], 0)
    if pts.size == 0:
        return 0
    return int(np.unique(np.round(pts, 6), axis=0).shape[0])


def _stats(raw_v: int, raw_e: int, new_v: int, new_e: int,
           diag: float) -> dict[str, Any]:
    return {
        "raw_vertices": int(raw_v),
        "raw_edges": int(raw_e),
        "clean_vertices": int(new_v),
        "clean_edges": int(new_e),
        "vert_reduction": round(1 - new_v / raw_v, 3) if raw_v else 0.0,
        "edge_reduction": round(1 - new_e / raw_e, 3) if raw_e else 0.0,
        "bbox_diag": round(float(diag), 4),
    }


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------
def _load_npz(path: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _save_npz(path: str, data: dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **data)


# ----------------------------------------------------------------------
# Visualization (before/after)
# ----------------------------------------------------------------------
def _visualize(samples: list[dict[str, Any]], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(samples)
    fig = plt.figure(figsize=(8, 3.2 * n))
    for r, s in enumerate(samples):
        for c, (tag, polys) in enumerate(
            [("raw", s["raw_polys"]), ("clean", s["clean_polys"])]
        ):
            ax = fig.add_subplot(n, 2, r * 2 + c + 1, projection="3d")
            for p in polys:
                ax.plot(p[:, 0], p[:, 1], p[:, 2], lw=0.6)
            st = s["stats"]
            cnt = st["raw_edges"] if tag == "raw" else st["clean_edges"]
            vcnt = st["raw_vertices"] if tag == "raw" else st["clean_vertices"]
            ax.set_title(
                f"{s['name'][:22]}\n{tag}: {cnt} edges / {vcnt} verts",
                fontsize=8,
            )
            ax.set_box_aspect((1, 1, 1))
            ax.tick_params(labelsize=5)
    # tight_layout misjudges 3D-axis extents, so the bottom row's ticks/labels
    # get clipped; bbox_inches="tight" expands the canvas to include them.
    fig.subplots_adjust(top=0.98, bottom=0.02, hspace=0.3, wspace=0.05)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _polys_of(data: dict[str, Any]) -> list[np.ndarray]:
    ep = data["edge_points"]
    return [np.asarray(ep[i], dtype=np.float64).reshape(-1, 3)
            for i in range(len(ep))]


# ----------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------
def _list_npz(directory: str) -> list[str]:
    import glob
    return sorted(glob.glob(os.path.join(directory, "*.npz")))


def run_test(args: argparse.Namespace, cfg: CleanConfig) -> None:
    if args.files:
        files = list(args.files)
    else:
        files = _list_npz(args.in_dir)
        if args.pick == "worst":
            # pick the files with the most raw edges (the dirtiest ones)
            scored = []
            sample_pool = files
            if len(files) > 2000:
                random.seed(args.seed)
                sample_pool = random.sample(files, 2000)
            for f in sample_pool:
                try:
                    # read only the lightweight float array for the edge count
                    # (avoids unpickling the heavy object polyline array)
                    with np.load(f, allow_pickle=False) as z:
                        scored.append((int(z["start_verts"].shape[0]), f))
                except Exception:
                    continue
            scored.sort(reverse=True)
            files = [f for _, f in scored[: args.num]]
        else:
            random.seed(args.seed)
            files = random.sample(files, min(args.num, len(files)))

    samples = []
    print(f"{'shape':<40} {'raw_v':>6} {'raw_e':>6} -> "
          f"{'cl_v':>6} {'cl_e':>6}  {'dV%':>5} {'dE%':>5}")
    for f in files:
        data = _load_npz(f)
        new, st = clean_wireframe(data, cfg)
        name = os.path.splitext(os.path.basename(f))[0]
        print(f"{name:<40} {st['raw_vertices']:>6} {st['raw_edges']:>6} -> "
              f"{st['clean_vertices']:>6} {st['clean_edges']:>6}  "
              f"{st['vert_reduction']*100:>5.0f} {st['edge_reduction']*100:>5.0f}")
        samples.append({
            "name": name,
            "stats": st,
            "raw_polys": _polys_of(data),
            "clean_polys": _polys_of(new),
        })

    if not args.no_viz:
        _visualize(samples, args.viz_out)
        print(f"\nSaved before/after visualization -> {args.viz_out}")

    if args.save:
        for f, s in zip(files, samples):
            new, _ = clean_wireframe(_load_npz(f), cfg)
            out = os.path.join(args.out_dir, os.path.basename(f))
            _save_npz(out, new)
        print(f"Saved {len(files)} cleaned npz -> {args.out_dir}")


def _clean_one_file(args_tuple: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    in_path, out_path, cfg_dict = args_tuple
    cfg = CleanConfig(**cfg_dict)
    try:
        data = _load_npz(in_path)
        new, st = clean_wireframe(data, cfg)
        _save_npz(out_path, new)
        st["file"] = os.path.basename(in_path)
        st["ok"] = True
        return st
    except Exception as exc:  # pragma: no cover
        return {"file": os.path.basename(in_path), "ok": False,
                "error": str(exc)}


def run_all(args: argparse.Namespace, cfg: CleanConfig) -> None:
    files = _list_npz(args.in_dir)
    if not files:
        raise SystemExit(f"No npz files found in {args.in_dir!r}")
    if args.limit > 0:
        files = files[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    cfg_dict = asdict(cfg)
    tasks = [
        (f, os.path.join(args.out_dir, os.path.basename(f)), cfg_dict)
        for f in files
    ]
    print(f"Cleaning {len(tasks)} files -> {args.out_dir} "
          f"(workers={args.workers})")

    results: list[dict[str, Any]] = []
    if args.workers <= 1:
        for i, t in enumerate(tasks):
            results.append(_clean_one_file(t))
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(tasks)}")
    else:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for i, st in enumerate(pool.imap_unordered(
                    _clean_one_file, tasks, chunksize=8)):
                results.append(st)
                if (i + 1) % 500 == 0:
                    print(f"  {i + 1}/{len(tasks)}")

    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    raw_e = np.array([r["raw_edges"] for r in ok], dtype=np.float64)
    cl_e = np.array([r["clean_edges"] for r in ok], dtype=np.float64)
    raw_v = np.array([r["raw_vertices"] for r in ok], dtype=np.float64)
    cl_v = np.array([r["clean_vertices"] for r in ok], dtype=np.float64)

    def pct(a: np.ndarray) -> list[float]:
        if a.size == 0:
            return []
        return [round(float(x), 1)
                for x in np.percentile(a, [50, 75, 90, 95, 99, 100])]

    summary = {
        "config": cfg_dict,
        "num_files": len(results),
        "num_ok": len(ok),
        "num_failed": len(bad),
        "edges_raw_p50_75_90_95_99_max": pct(raw_e),
        "edges_clean_p50_75_90_95_99_max": pct(cl_e),
        "verts_raw_p50_75_90_95_99_max": pct(raw_v),
        "verts_clean_p50_75_90_95_99_max": pct(cl_v),
        "mean_edge_reduction": round(float(
            np.mean(1 - cl_e / np.maximum(raw_e, 1))), 3) if ok else 0.0,
        "frac_raw_over_1024": round(float(np.mean(raw_e > 1024)), 3) if ok else 0,
        "frac_clean_over_1024": round(float(np.mean(cl_e > 1024)), 3) if ok else 0,
        "frac_raw_over_384": round(float(np.mean(raw_e > 384)), 3) if ok else 0,
        "frac_clean_over_384": round(float(np.mean(cl_e > 384)), 3) if ok else 0,
        "failures": bad[:50],
    }
    report_path = os.path.join(args.out_dir, "_clean_report.json")
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print("\n=== Cleaning summary ===")
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "failures"}, indent=2))
    print(f"\nReport -> {report_path}")
    if bad:
        print(f"WARNING: {len(bad)} files failed (see report).")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--in-dir", default="data/train/sample_edge")
        sp.add_argument("--out-dir", default="data/train_clean/sample_edge")
        sp.add_argument("--weld-rel", type=float, default=5e-3,
                        help="weld vertices within this fraction of bbox diag")
        sp.add_argument("--weld-abs", type=float, default=0.0)
        sp.add_argument("--no-dissolve", action="store_true")
        sp.add_argument("--max-turn-deg", type=float, default=15.0,
                        help="dissolve degree-2 chains turning less than this")
        sp.add_argument("--no-split", action="store_true",
                        help="do not split spirals / zig-zags")
        sp.add_argument("--corner-deg", type=float, default=45.0,
                        help="split a polyline at interior kinks above this")
        sp.add_argument("--max-total-turn-deg", type=float, default=180.0,
                        help="split a curve every this much accumulated turn")
        sp.add_argument("--min-arc-rel", type=float, default=2e-3,
                        help="drop edges shorter than this fraction of diag")
        sp.add_argument("--min-chord-rel", type=float, default=1e-3,
                        help="drop edges whose endpoints coincide within this "
                             "fraction of bbox diag")
        sp.add_argument("--num-edge-points", type=int, default=32)

    sp_test = sub.add_parser("test", help="clean a few + visualize before/after")
    add_common(sp_test)
    sp_test.add_argument("--num", type=int, default=6)
    sp_test.add_argument("--pick", choices=["worst", "random"], default="worst")
    sp_test.add_argument("--files", nargs="*", default=None)
    sp_test.add_argument("--seed", type=int, default=0)
    sp_test.add_argument("--viz-out", default="logs/clean_preview.png")
    sp_test.add_argument("--no-viz", action="store_true")
    sp_test.add_argument("--save", action="store_true",
                         help="also write the cleaned npz of the tested files")

    sp_all = sub.add_parser("all", help="clean the whole directory")
    add_common(sp_all)
    sp_all.add_argument("--workers", type=int, default=8)
    sp_all.add_argument("--limit", type=int, default=0,
                        help="only process the first N files (debug)")

    return p


def cfg_from_args(args: argparse.Namespace) -> CleanConfig:
    return CleanConfig(
        weld_rel=args.weld_rel,
        weld_abs=args.weld_abs,
        dissolve=not args.no_dissolve,
        max_turn_deg=args.max_turn_deg,
        split=not args.no_split,
        corner_deg=args.corner_deg,
        max_total_turn_deg=args.max_total_turn_deg,
        min_arc_rel=args.min_arc_rel,
        min_chord_rel=args.min_chord_rel,
        num_edge_points=args.num_edge_points,
    )


def main() -> None:
    args = build_parser().parse_args()
    cfg = cfg_from_args(args)
    if args.mode == "test":
        run_test(args, cfg)
    elif args.mode == "all":
        run_all(args, cfg)


if __name__ == "__main__":
    main()
