#!/usr/bin/env python3
"""Clean the (very noisy) PC2Wireframe *train* ground-truth wireframes.

Motivation
----------
The raw ``data/train/sample_edge/*.npz`` ground truth is extremely dirty: even
geometrically simple parts ship hundreds — sometimes thousands — of edges. A
quick audit over the training set shows::

    edges  p50=258  p75=756  p90=1805  p99=11k  max=23k
    -> 38% of samples blow past the 384-edge cap and ~19% past 1024,
       i.e. a large chunk of training data is silently *skipped* by the loader.

The root causes are exactly what you suspected:

1. **Near-duplicate vertices.** Endpoints that are visually the same point sit
   at slightly different coordinates, so they never merge. Each surplus vertex
   keeps its incident edges separate.
2. **Subdivided edges.** A single straight / smooth CAD edge is stored as a
   chain of many tiny segments passing through degree-2 vertices.
3. **Overlapping / duplicate edges.** The same edge appears more than once
   (e.g. once per adjacent face), or two slightly-offset copies coexist.
4. **Tessellated detail patches.** Some regions are a dense 2D mesh of tiny
   segments rather than a clean B-rep wireframe.

This script applies a principled, geometry-faithful clean-up:

    weld near-duplicate vertices   (KD-tree union-find, tol ~ % of bbox diag)
      -> drop degenerate edges     (both endpoints welded together)
      -> de-duplicate edges        (same endpoints + similar polyline)
      -> dissolve degree-2 chains  (merge smooth/collinear subdivided edges)
      -> resample every edge        back to a fixed point count

The NPZ schema is preserved exactly (``start_verts``, ``end_verts``,
``edge_points``, ``edge_us``, ``original_edge_indices``) so the cleaned files
are a drop-in replacement for the dataset / dataloader.

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

Point clouds are untouched: cleaning only edits the wireframe edge files, so
the cleaned split can keep pointing at the original ``sample_pointcloud`` dir.
Tune ``--weld-rel`` / ``--max-turn-deg`` / ``--dissolve`` to trade fidelity for
sparsity; the ``test`` mode is there precisely so you can eyeball the result.
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

    ``weld_tol = max(weld_abs, weld_rel * bbox_diag)``. Everything is measured
    in the shape's own coordinate frame; using a fraction of the bounding-box
    diagonal makes the defaults scale-invariant across parts.
    """

    weld_rel: float = 5e-3      # weld points closer than 0.5% of bbox diagonal
    weld_abs: float = 0.0       # ... but never below this absolute distance
    dedup: bool = True          # collapse duplicate edges (same endpoints)
    dedup_geom_rel: float = 5e-3  # two edges are "the same" if mean dist < this
    dissolve: bool = True       # merge smooth degree-2 chains into one edge
    max_turn_deg: float = 15.0  # chain stays merged while the turn stays below
    num_edge_points: int = 32   # resample every cleaned edge to this many pts
    min_edge_len_rel: float = 0.0  # optionally drop edges shorter than this


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
    out = np.stack(
        [np.interp(dst, cum, poly[:, c]) for c in range(3)], axis=-1)
    return out


def _tangent_away(poly: np.ndarray, at_start: bool, min_dist: float) -> np.ndarray:
    """Unit direction leaving the polyline endpoint, robust to tiny segments.

    Walks a few points inward until it has moved at least ``min_dist`` so that
    a single noisy first segment cannot dominate the tangent estimate.
    """
    poly = np.asarray(poly, dtype=np.float64)
    if poly.shape[0] < 2:
        return np.zeros(3)
    if at_start:
        origin = poly[0]
        seq = poly[1:]
    else:
        origin = poly[-1]
        seq = poly[-2::-1]
    far = seq[-1]
    for p in seq:
        if np.linalg.norm(p - origin) >= min_dist:
            far = p
            break
    d = far - origin
    nrm = np.linalg.norm(d)
    return d / nrm if nrm > 1e-12 else np.zeros(3)


# ----------------------------------------------------------------------
# Vertex welding
# ----------------------------------------------------------------------
def _weld(points: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Union-find weld of points within ``tol``.

    Returns ``(ids, centroids)`` where ``ids[i]`` is the welded vertex id of
    input point ``i`` and ``centroids`` are the per-cluster mean coordinates.
    """
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3))
    parent = np.arange(n)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    if tol > 0 and cKDTree is not None:
        tree = cKDTree(points)
        pairs = tree.query_pairs(tol, output_type="ndarray")
        for a, b in pairs:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb

    remap: dict[int, int] = {}
    ids = np.empty(n, dtype=np.int64)
    for i in range(n):
        r = find(i)
        if r not in remap:
            remap[r] = len(remap)
        ids[i] = remap[r]
    nv = len(remap)
    centroids = np.zeros((nv, 3))
    counts = np.zeros(nv)
    np.add.at(centroids, ids, points)
    np.add.at(counts, ids, 1.0)
    centroids /= np.maximum(counts[:, None], 1.0)
    return ids, centroids


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
    raw_verts = int(
        np.unique(
            np.round(
                np.concatenate([p[[0, -1]] for p in polys], 0)
                if polys else np.zeros((0, 3)),
                6,
            ),
            axis=0,
        ).shape[0]
    ) if polys else 0

    if not polys:
        empty = _empty_npz(cfg.num_edge_points)
        return empty, _stats(raw_verts, raw_edges, 0, 0, 0.0)

    all_pts = np.concatenate(polys, axis=0)
    diag = _bbox_diag(all_pts)
    weld_tol = max(cfg.weld_abs, cfg.weld_rel * diag)

    # --- 1. weld endpoints --------------------------------------------------
    endpoints = np.stack([np.concatenate([p[0], p[-1]]) for p in polys])
    endpoints = endpoints.reshape(-1, 3)  # [s0,e0,s1,e1,...]
    ids, centroids = _weld(endpoints, weld_tol)
    ids = ids.reshape(raw_edges, 2)

    # snap polyline endpoints onto the welded centroids
    edges: list[dict[str, Any]] = []
    min_len = cfg.min_edge_len_rel * diag
    for i, p in enumerate(polys):
        u, v = int(ids[i, 0]), int(ids[i, 1])
        p = p.copy()
        p[0] = centroids[u]
        p[-1] = centroids[v]
        length = _polyline_length(p)
        # degenerate: endpoints welded together AND no real arc length (a true
        # closed loop with u==v but real length is kept as a loop edge).
        if u == v and length < weld_tol:
            continue
        if min_len > 0 and length < min_len and u != v:
            continue
        edges.append({"u": u, "v": v, "poly": p, "len": length})

    # --- 2. de-duplicate edges ---------------------------------------------
    if cfg.dedup and edges:
        edges = _dedup_edges(edges, cfg.dedup_geom_rel * diag)

    # --- 3. dissolve smooth degree-2 chains --------------------------------
    if cfg.dissolve and edges:
        edges = _dissolve_chains(
            edges, len(centroids), cfg.max_turn_deg, weld_tol)

    # --- 4. resample + repackage -------------------------------------------
    new = _pack_edges(edges, cfg.num_edge_points)
    stats = _stats(
        raw_verts, raw_edges,
        new["start_verts"].shape[0] and _count_unique_verts(new),
        new["start_verts"].shape[0],
        diag,
    )
    return new, stats


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
        if len(members) == 1:
            out.append(members[0])
            continue
        kept: list[dict[str, Any]] = []
        for e in sorted(members, key=lambda x: -x["len"]):
            mid_e = _resample_polyline(_orient(e, key[0]), 8)
            dup = False
            for k in kept:
                mid_k = _resample_polyline(_orient(k, key[0]), 8)
                if np.linalg.norm(mid_e - mid_k, axis=1).mean() < geom_tol:
                    dup = True
                    break
            if not dup:
                kept.append(e)
        out.extend(kept)
    return out


def _orient(edge: dict[str, Any], first: int) -> np.ndarray:
    """Return the polyline oriented so it starts at vertex id ``first``."""
    poly = edge["poly"]
    return poly if edge["u"] == first else poly[::-1]


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
    junctions (degree != 2) and sharp corners are preserved. Pure smooth cycles
    are kept as two edges so they survive the ``u != v`` requirement.
    """
    cos_thresh = np.cos(np.deg2rad(180.0 - max_turn_deg))
    # adjacency: vertex -> list of (edge_id)
    adj: list[list[int]] = [[] for _ in range(num_vertices)]
    for ei, e in enumerate(edges):
        adj[e["u"]].append(ei)
        if e["v"] != e["u"]:
            adj[e["v"]].append(ei)

    def is_smooth(vid: int) -> bool:
        inc = adj[vid]
        if len(inc) != 2 or inc[0] == inc[1]:
            return False
        e0, e1 = edges[inc[0]], edges[inc[1]]
        if e0 is None or e1 is None:
            return False
        d0 = _tangent_away(_orient_to(e0, vid), True, weld_tol)
        d1 = _tangent_away(_orient_to(e1, vid), True, weld_tol)
        if not d0.any() or not d1.any():
            return False
        # smooth pass-through => the two "away" directions are ~antiparallel
        return float(d0 @ d1) <= cos_thresh

    # iteratively dissolve until a fixpoint
    alive = [True] * len(edges)
    changed = True
    while changed:
        changed = False
        for vid in range(num_vertices):
            inc = [ei for ei in adj[vid] if alive[ei]]
            adj[vid] = inc
            if len(inc) != 2 or inc[0] == inc[1]:
                continue
            if not is_smooth(vid):
                continue
            ei0, ei1 = inc
            e0, e1 = edges[ei0], edges[ei1]
            a = e0["v"] if e0["u"] == vid else e0["u"]
            b = e1["v"] if e1["u"] == vid else e1["u"]
            if a == b:
                # merging would fold a 2-edge bigon into a self loop; leave it
                continue
            poly_a = _orient_to(e0, a)              # a ... vid
            poly_b = _orient_to(e1, vid)            # vid ... b
            merged = np.concatenate([poly_a, poly_b[1:]], axis=0)
            new_id = len(edges)
            edges.append({"u": a, "v": b, "poly": merged,
                          "len": _polyline_length(merged)})
            alive.append(True)
            alive[ei0] = alive[ei1] = False
            edges[ei0] = edges[ei1] = None  # type: ignore[assignment]
            # rewire: vid loses both edges; a and b swap their old edge -> new
            adj[vid] = []
            adj[a] = [x for x in adj[a] if x not in (ei0, ei1)] + [new_id]
            adj[b] = [x for x in adj[b] if x not in (ei0, ei1)] + [new_id]
            changed = True

    return [e for e in edges if e is not None]


def _orient_to(edge: dict[str, Any], start_vid: int) -> np.ndarray:
    poly = edge["poly"]
    return poly if edge["u"] == start_vid else poly[::-1]


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
    fig = plt.figure(figsize=(8, 4 * n))
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
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120)
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
                        help="weld points within this fraction of bbox diag")
        sp.add_argument("--weld-abs", type=float, default=0.0)
        sp.add_argument("--no-dedup", action="store_true")
        sp.add_argument("--no-dissolve", action="store_true")
        sp.add_argument("--max-turn-deg", type=float, default=15.0,
                        help="dissolve degree-2 chains turning less than this")
        sp.add_argument("--num-edge-points", type=int, default=32)
        sp.add_argument("--min-edge-len-rel", type=float, default=0.0)

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
        dedup=not args.no_dedup,
        dissolve=not args.no_dissolve,
        max_turn_deg=args.max_turn_deg,
        num_edge_points=args.num_edge_points,
        min_edge_len_rel=args.min_edge_len_rel,
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
