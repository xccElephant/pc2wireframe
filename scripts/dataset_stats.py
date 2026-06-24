"""Compute distribution statistics over the train split.

For every matched (point cloud, edge) pair under ``data/train`` we collect:

  * number of surface (point-cloud) points  -- ``surface_points``;
  * number of raw edges                      -- ``start_verts`` count;
  * number of merged wireframe vertices      -- unique endpoints (tol 1e-4);
  * point-cloud coordinate range             -- per-axis min/max, bbox extent,
    centroid, and max(|coord|).

Results are summarized as percentiles + coarse histograms and dumped to JSON.
"""
from __future__ import annotations

import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PC_DIR = os.path.join(ROOT, "data", "train", "sample_pointcloud")
EDGE_DIR = os.path.join(ROOT, "data", "train", "sample_edge")

VERTEX_MERGE_TOL = 1e-4


def _load(path: str) -> dict:
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except ValueError:
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}


def _count_merged_vertices(start: np.ndarray, end: np.ndarray) -> int:
    verts = np.concatenate([start.reshape(-1, 3), end.reshape(-1, 3)], axis=0)
    verts = verts[np.isfinite(verts).all(axis=1)]
    if verts.shape[0] == 0:
        return 0
    keys = np.rint(verts / max(VERTEX_MERGE_TOL, 1e-12)).astype(np.int64)
    return int(np.unique(keys, axis=0).shape[0])


def process(stem: str) -> dict | None:
    pc_path = os.path.join(PC_DIR, f"{stem}.npz")
    edge_path = os.path.join(EDGE_DIR, f"{stem}.npz")
    if not (os.path.isfile(pc_path) and os.path.isfile(edge_path)):
        return None
    try:
        pc = _load(pc_path)
        pts = np.asarray(
            pc.get("surface_points", pc.get("points")), dtype=np.float64
        ).reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]

        ed = _load(edge_path)
        start = np.asarray(ed["start_verts"], dtype=np.float64)
        end = np.asarray(ed["end_verts"], dtype=np.float64)
        n_edges = int(min(start.reshape(-1, 3).shape[0],
                          end.reshape(-1, 3).shape[0]))
        n_verts = _count_merged_vertices(start, end)

        if pts.shape[0] == 0:
            mn = mx = np.zeros(3)
        else:
            mn = pts.min(axis=0)
            mx = pts.max(axis=0)
        extent = mx - mn
        return {
            "stem": stem,
            "n_points": int(pts.shape[0]),
            "n_edges": n_edges,
            "n_verts": n_verts,
            "min": mn.tolist(),
            "max": mx.tolist(),
            "extent": extent.tolist(),
            "max_abs": float(np.abs(pts).max()) if pts.shape[0] else 0.0,
            "centroid": pts.mean(axis=0).tolist() if pts.shape[0] else [0, 0, 0],
        }
    except Exception as exc:  # noqa: BLE001
        return {"stem": stem, "error": str(exc)}


def pct(arr: np.ndarray, ps=(0, 1, 5, 25, 50, 75, 90, 95, 99, 100)) -> dict:
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def hist(arr: np.ndarray, bins) -> list:
    mx = float(arr.max())
    bins = sorted({b for b in bins if b <= mx} | {0.0, mx + 1})
    counts, edges = np.histogram(arr, bins=bins)
    return [
        {"range": [float(edges[i]), float(edges[i + 1])], "count": int(counts[i])}
        for i in range(len(counts))
    ]


def main() -> None:
    stems = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(EDGE_DIR, "*.npz"))
    )
    print(f"Found {len(stems)} edge files; matching point clouds...")

    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(process, stems, chunksize=64)):
            if r is not None:
                results.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  processed {i + 1}/{len(stems)}")

    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    print(f"\nOK: {len(ok)}  errors: {len(errs)}  unmatched: "
          f"{len(stems) - len(results)}")

    n_points = np.array([r["n_points"] for r in ok])
    n_edges = np.array([r["n_edges"] for r in ok])
    n_verts = np.array([r["n_verts"] for r in ok])
    max_abs = np.array([r["max_abs"] for r in ok])
    mins = np.array([r["min"] for r in ok])
    maxs = np.array([r["max"] for r in ok])
    extents = np.array([r["extent"] for r in ok])
    max_extent = extents.max(axis=1)

    def show(name, arr):
        print(f"\n=== {name} ===")
        print(f"  count={len(arr)} mean={arr.mean():.4f} std={arr.std():.4f} "
              f"min={arr.min():.4f} max={arr.max():.4f}")
        for k, v in pct(arr).items():
            print(f"    {k:>5}: {v:.4f}")

    show("点云点数 n_points", n_points)
    print("  histogram:")
    for b in hist(n_points, [0, 1000, 2000, 4000, 6000, 8000, 10000, 15000,
                             20000, 30000, n_points.max() + 1]):
        print(f"    {b['range'][0]:>8.0f}-{b['range'][1]:<8.0f}: {b['count']}")

    show("边数量 n_edges", n_edges)
    print("  histogram:")
    for b in hist(n_edges, [0, 10, 25, 50, 100, 200, 300, 500, 1000,
                            n_edges.max() + 1]):
        print(f"    {b['range'][0]:>6.0f}-{b['range'][1]:<6.0f}: {b['count']}")

    show("顶点数 n_verts(merged)", n_verts)
    print("  histogram:")
    for b in hist(n_verts, [0, 10, 25, 50, 100, 200, 300, 512, 1000,
                            n_verts.max() + 1]):
        print(f"    {b['range'][0]:>6.0f}-{b['range'][1]:<6.0f}: {b['count']}")

    show("max(|coord|)", max_abs)
    print(f"  样本 max|coord|>1.2 (会被 dataset 跳过): "
          f"{int((max_abs > 1.2).sum())} "
          f"({100 * (max_abs > 1.2).mean():.2f}%)")
    print(f"  样本 max|coord|<=1.0: {int((max_abs <= 1.0).sum())} "
          f"({100 * (max_abs <= 1.0).mean():.2f}%)")

    print("\n=== 坐标范围 (per-axis) ===")
    for ax, name in enumerate("xyz"):
        print(f"  {name}: min in [{mins[:,ax].min():.4f}, {mins[:,ax].max():.4f}] "
              f"max in [{maxs[:,ax].min():.4f}, {maxs[:,ax].max():.4f}]")
    show("最大轴向跨度 max_extent", max_extent)

    out = {
        "n_total_edge_files": len(stems),
        "n_ok": len(ok),
        "n_errors": len(errs),
        "errors": errs[:50],
        "n_points": {"percentiles": pct(n_points), "mean": float(n_points.mean()),
                     "std": float(n_points.std())},
        "n_edges": {"percentiles": pct(n_edges), "mean": float(n_edges.mean()),
                    "std": float(n_edges.std())},
        "n_verts": {"percentiles": pct(n_verts), "mean": float(n_verts.mean()),
                    "std": float(n_verts.std()),
                    "frac_gt_512": float((n_verts > 512).mean())},
        "max_abs": {"percentiles": pct(max_abs),
                    "frac_gt_1.2": float((max_abs > 1.2).mean())},
        "max_extent": {"percentiles": pct(max_extent)},
        "per_axis_min": {"x": pct(mins[:, 0]), "y": pct(mins[:, 1]),
                         "z": pct(mins[:, 2])},
        "per_axis_max": {"x": pct(maxs[:, 0]), "y": pct(maxs[:, 1]),
                         "z": pct(maxs[:, 2])},
    }
    out_path = os.path.join(HERE, "dataset_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON -> {out_path}")


if __name__ == "__main__":
    main()
