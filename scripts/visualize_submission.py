#!/usr/bin/env python3
"""Visualize input point clouds against predicted wireframes.

Picks a few shapes (random by default) that exist in *both* the test
point-cloud dir and an exported submission's ``sample_edge/`` dir, and renders
a side-by-side 3D comparison per shape:

    col 1 : input surface point cloud
    col 2 : predicted wireframe (vertices + edge polylines)
    col 3 : overlay (point cloud + wireframe)

Usage (from the project root)::

    python scripts/visualize_submission.py \
        --sub-dir logs/submission/submission \
        --test-pc-dir data/test/sample_pointcloud \
        --num 6 --out-dir logs/submission/viz

    # specific shapes instead of random
    python scripts/visualize_submission.py --stems 00050008_..._step_000 ...
"""
from __future__ import annotations

import argparse
import glob
import os
import random

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401 (registers 3d proj)
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402


def _load_pc(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=True) as z:
        for k in ("surface_points", "points", "point_cloud", "pc"):
            if k in z.files:
                return np.asarray(z[k], dtype=np.float32).reshape(-1, 3)
    raise KeyError(f"no point-cloud array in {path!r}")


def _load_pred(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        verts = np.asarray(z["vertices"], dtype=np.float32).reshape(-1, 3)
        edge_index = np.asarray(z["edge_index"], dtype=np.int64).reshape(-1, 2)
        edge_points = (
            np.asarray(z["edge_points"], dtype=np.float32)
            if "edge_points" in z.files else np.zeros((0, 0, 3), np.float32)
        )
    return {"vertices": verts, "edge_index": edge_index,
            "edge_points": edge_points}


def _edge_segments(pred: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Return a list of ``(U, 3)`` polylines for each predicted edge."""
    ep = pred["edge_points"]
    if ep.size and ep.ndim == 3 and ep.shape[0] == pred["edge_index"].shape[0]:
        return [ep[i] for i in range(ep.shape[0])]
    # fall back to straight segments between endpoint vertices
    verts, ei = pred["vertices"], pred["edge_index"]
    segs = []
    for a, b in ei:
        if 0 <= a < verts.shape[0] and 0 <= b < verts.shape[0]:
            segs.append(np.stack([verts[a], verts[b]], axis=0))
    return segs


def _set_equal_3d(ax, pts: np.ndarray) -> None:
    """Equal aspect ratio cube around ``pts`` (N,3)."""
    if pts.size == 0:
        return
    lo, hi = pts.min(0), pts.max(0)
    ctr = (lo + hi) / 2.0
    r = float((hi - lo).max()) / 2.0 or 1.0
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)
    ax.set_box_aspect((1, 1, 1))


def _draw_pc(ax, pc: np.ndarray, color="#1f77b4", s=2.0, alpha=0.5) -> None:
    if pc.size:
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=s, c=color,
                   alpha=alpha, linewidths=0, depthshade=True)


def _draw_wireframe(ax, pred: dict[str, np.ndarray],
                    line_color="#d62728", vert_color="#2ca02c") -> None:
    segs = _edge_segments(pred)
    if segs:
        lc = Line3DCollection([s[:, :3] for s in segs], colors=line_color,
                              linewidths=1.4)
        ax.add_collection3d(lc)
    v = pred["vertices"]
    if v.size:
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=14, c=vert_color,
                   depthshade=False)


def visualize(stem: str, pc_path: str, pred_path: str, out_path: str,
              elev: float = 20.0, azim: float = -60.0) -> None:
    pc = _load_pc(pc_path)
    pred = _load_pred(pred_path)
    segs = _edge_segments(pred)
    all_pts = [a for a in (pc, pred["vertices"]) if a.size]
    if segs:
        all_pts.append(np.concatenate([s.reshape(-1, 3) for s in segs], 0))
    bbox = np.concatenate(all_pts, 0) if all_pts else np.zeros((1, 3), np.float32)

    fig = plt.figure(figsize=(16, 5.6))
    titles = [
        f"input point cloud\n({pc.shape[0]} pts)",
        f"prediction\n({pred['vertices'].shape[0]} verts, "
        f"{pred['edge_index'].shape[0]} edges)",
        "overlay",
    ]
    for col in range(3):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        if col in (0, 2):
            _draw_pc(ax, pc)
        if col in (1, 2):
            _draw_wireframe(ax, pred)
        ax.set_title(titles[col], fontsize=10)
        ax.view_init(elev=elev, azim=azim)
        _set_equal_3d(ax, bbox)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(False)

    fig.suptitle(stem, fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.86,
                        wspace=0.02)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[viz] {stem} -> {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sub-dir", default="logs/submission/submission",
                   help="submission dir containing sample_edge/<stem>.npz")
    p.add_argument("--test-pc-dir", default="data/test/sample_pointcloud")
    p.add_argument("--out-dir", default="logs/submission/viz")
    p.add_argument("--num", type=int, default=6,
                   help="number of random shapes (ignored if --stems given)")
    p.add_argument("--stems", nargs="*", default=None,
                   help="explicit shape stems to render")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--elev", type=float, default=20.0)
    p.add_argument("--azim", type=float, default=-60.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sample_dir = os.path.join(args.sub_dir, "sample_edge")
    pred_stems = {
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(sample_dir, "*.npz"))
    }
    if not pred_stems:
        raise SystemExit(f"No predictions under {sample_dir!r}")

    if args.stems:
        stems = list(args.stems)
    else:
        pc_stems = {
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(args.test_pc_dir, "*.npz"))
        }
        common = sorted(pred_stems & pc_stems)
        if not common:
            raise SystemExit("No overlapping stems between predictions and "
                             "point clouds.")
        random.seed(args.seed)
        stems = random.sample(common, k=min(args.num, len(common)))

    os.makedirs(args.out_dir, exist_ok=True)
    for stem in stems:
        pc_path = os.path.join(args.test_pc_dir, f"{stem}.npz")
        pred_path = os.path.join(sample_dir, f"{stem}.npz")
        if not os.path.isfile(pc_path):
            print(f"[viz] skip {stem}: missing point cloud")
            continue
        if not os.path.isfile(pred_path):
            print(f"[viz] skip {stem}: missing prediction")
            continue
        visualize(stem, pc_path, pred_path,
                  os.path.join(args.out_dir, f"{stem}.png"),
                  elev=args.elev, azim=args.azim)

    print(f"[viz] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
