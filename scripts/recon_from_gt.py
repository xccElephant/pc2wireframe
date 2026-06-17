#!/usr/bin/env python3
"""Sanity-check the *traditional reconstruction* upper bound (the pipeline's roof).

The RF branch is ``point cloud -> z -> RF point set (xyz, type) -> traditional
reconstruction -> wireframe``. The last stage (``src/recon/traditional.py``) is
hand-written and **non-learned**, so it caps the whole pipeline: even a perfect
RF sampler can only be as good as what the traditional reconstructor can read
out of the *ideal* point set.

This script measures exactly that ceiling. For each shape it:

  1. builds the **GT-derived RF target point set** ``wf_points (N, 4)`` -- the
     same arc-length/vertex target the RF net is trained to regress -- using the
     real dataset code path (so it is faithful to training);
  2. feeds that *oracle* point set straight into ``reconstruct_wireframe`` (skip
     the encoder + RF net entirely);
  3. scores the reconstruction against the GT wireframe with the same
     CCD / TA / VPE proxies the training loop uses (numpy/scipy port, no GPU);
  4. renders a 3-column comparison per shape:
     ``GT wireframe | GT point set (colored by type) | reconstruction``.

If reconstruction is unstable *here*, no amount of generative-model tuning will
fix the final score -- the read-out is the bottleneck.

Usage (from the project root)::

    # 6 random shapes -> logs/recon_from_gt.png + a per-shape metric table
    python scripts/recon_from_gt.py --num 6

    # the dirtiest (most-edge) shapes, reproducible pick
    python scripts/recon_from_gt.py --num 8 --pick worst --seed 0

    # explicit files + tweak the reconstruction knobs
    python scripts/recon_from_gt.py \
        --files data/train/sample_edge/00030001_..._step_002.npz \
        --merge-radius 0.03 --min-votes 3 --num-per-edge 32
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
from typing import Any

import numpy as np

# ----------------------------------------------------------------------
# Load the project's real code paths WITHOUT importing the ``src`` package
# (which would drag in pytorch_lightning). We exec the two dependency-light
# modules directly: dataset.py (numpy/torch) and recon/traditional.py (numpy/
# scipy). This guarantees the experiment exercises the *exact* target-building
# and reconstruction code that training/inference use.
# ----------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name: str, relpath: str):
    path = os.path.join(_PROJECT_ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {relpath}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_dataset = _load_module("_pc2w_dataset", os.path.join("src", "data", "dataset.py"))
_recon = _load_module("_pc2w_recon", os.path.join("src", "recon", "traditional.py"))

WireframeGraphDataset = _dataset.WireframeGraphDataset
resolve_split_files = _dataset.resolve_split_files
reconstruct_wireframe = _recon.reconstruct_wireframe


# ----------------------------------------------------------------------
# Lightweight numpy/scipy metrics (a CPU port of src/metrics/functional.py so
# this script needs neither a GPU nor pytorch3d). Same definitions, same
# exp(-d/tau) score mapping and default weights as the training metric.
# ----------------------------------------------------------------------
def _flatten_curves(wf: dict[str, np.ndarray], num_per_edge: int) -> np.ndarray:
    """Dense ``(M, 3)`` sampling of all curves in a wireframe."""
    ep = wf.get("edge_points")
    if ep is not None and np.asarray(ep).size:
        return np.asarray(ep, dtype=np.float64).reshape(-1, 3)
    ei = np.asarray(wf.get("edge_index"), dtype=np.int64).reshape(-1, 2)
    v = np.asarray(wf.get("vertices"), dtype=np.float64).reshape(-1, 3)
    if ei.size == 0 or v.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    t = np.linspace(0.0, 1.0, num_per_edge)[None, :, None]
    a = v[ei[:, 0]][:, None, :]
    b = v[ei[:, 1]][:, None, :]
    return (a * (1.0 - t) + b * t).reshape(-1, 3)


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric L2 chamfer (mean of means); inf if either side is empty."""
    from scipy.spatial import cKDTree

    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("inf")
    da, _ = cKDTree(b).query(a, k=1)
    db, _ = cKDTree(a).query(b, k=1)
    return 0.5 * (float(np.mean(da)) + float(np.mean(db)))


def _topology_accuracy(
    pred: dict[str, np.ndarray], gt: dict[str, np.ndarray], match_thresh: float
) -> float:
    """Edge-level F1 after nearest-neighbour vertex matching (matches functional.py)."""
    from scipy.spatial import cKDTree

    gt_e = np.asarray(gt.get("edge_index"), dtype=np.int64).reshape(-1, 2)
    pred_e = np.asarray(pred.get("edge_index"), dtype=np.int64).reshape(-1, 2)
    pred_v = np.asarray(pred.get("vertices"), dtype=np.float64).reshape(-1, 3)
    gt_v = np.asarray(gt.get("vertices"), dtype=np.float64).reshape(-1, 3)

    if len(gt_e) == 0:
        return 1.0 if len(pred_e) == 0 else 0.0
    if len(pred_e) == 0 or pred_v.shape[0] == 0 or gt_v.shape[0] == 0:
        return 0.0

    dist, idx = cKDTree(gt_v).query(pred_v, k=1)
    mapped = np.where(dist <= match_thresh, idx, -1)
    gt_set = {frozenset((int(i), int(j))) for i, j in gt_e if i != j}

    correct_pred = 0
    pred_pairs: set[frozenset] = set()
    for i, j in pred_e:
        gi, gj = int(mapped[i]), int(mapped[j])
        if gi < 0 or gj < 0 or gi == gj:
            continue
        pair = frozenset((gi, gj))
        if pair in gt_set:
            correct_pred += 1
            pred_pairs.add(pair)

    precision = correct_pred / max(1, len(pred_e))
    recall = len(pred_pairs & gt_set) / max(1, len(gt_set))
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _dist_to_score(d: float, tau: float) -> float:
    if not np.isfinite(d):
        return 0.0
    return float(np.exp(-max(0.0, d) / max(1e-9, tau)))


def score_reconstruction(
    pred: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    *,
    num_per_edge: int,
    ccd_tau: float,
    vpe_tau: float,
    match_thresh: float,
    w_ccd: float,
    w_ta: float,
    w_vpe: float,
) -> dict[str, float]:
    ccd = _chamfer(
        _flatten_curves(pred, num_per_edge), _flatten_curves(gt, num_per_edge))
    vpe = _chamfer(pred.get("vertices"), gt.get("vertices"))
    ta = _topology_accuracy(pred, gt, match_thresh)
    ccd_s = _dist_to_score(ccd, ccd_tau)
    vpe_s = _dist_to_score(vpe, vpe_tau)
    score = w_ccd * ccd_s + w_ta * ta + w_vpe * vpe_s
    return {
        "ccd": ccd, "vpe": vpe, "ta": ta,
        "ccd_score": ccd_s, "vpe_score": vpe_s, "score": score,
    }


# ----------------------------------------------------------------------
# Sample selection
# ----------------------------------------------------------------------
def _select_indices(
    dataset: Any, num: int, pick: str, seed: int
) -> list[int]:
    n = len(dataset)
    num = min(num, n)
    if pick == "first":
        return list(range(num))
    if pick == "worst":
        # rank by raw edge count (the dirtiest / hardest shapes); only the
        # lightweight float array is read, never the heavy object polylines.
        scored: list[tuple[int, int]] = []
        for i, f in enumerate(dataset.files):
            try:
                with np.load(f, allow_pickle=False) as z:
                    scored.append((int(z["start_verts"].shape[0]), i))
            except Exception:
                continue
        scored.sort(reverse=True)
        return [i for _, i in scored[:num]]
    # random (default)
    rng = random.Random(seed)
    return rng.sample(range(n), num)


def _to_np_wf(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    """Pull the native GT wireframe out of a dataset sample as numpy."""
    return {
        "vertices": sample["vertices"].numpy(),
        "edge_index": sample["edge_index"].numpy(),
        "edge_points": sample["edge_points"].numpy(),
    }


# ----------------------------------------------------------------------
# Visualization: GT wireframe | GT point set (by type) | reconstruction
# ----------------------------------------------------------------------
def _equal_box(ax, pts_list: list[np.ndarray]) -> None:
    allp = [p.reshape(-1, 3) for p in pts_list if p is not None and p.size]
    if not allp:
        ax.set_box_aspect((1, 1, 1))
        return
    cat = np.concatenate(allp, axis=0)
    lo = cat.min(0)
    hi = cat.max(0)
    c = (lo + hi) * 0.5
    r = float((hi - lo).max()) * 0.5 or 1.0
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))


def _draw_wireframe(ax, wf: dict[str, np.ndarray]) -> None:
    ep = np.asarray(wf.get("edge_points"))
    if ep.size:
        for i in range(ep.shape[0]):
            p = ep[i].reshape(-1, 3)
            ax.plot(p[:, 0], p[:, 1], p[:, 2], lw=0.7, color="#1f77b4")
    v = np.asarray(wf.get("vertices")).reshape(-1, 3)
    if v.size:
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=10, color="#d62728", depthshade=False)


def _visualize(rows: list[dict[str, Any]], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    fig = plt.figure(figsize=(11.5, 3.6 * n))
    col_titles = ["GT wireframe", "GT point set (type)", "traditional recon"]
    for r, row in enumerate(rows):
        gt = row["gt"]
        pred = row["pred"]
        pts = row["wf_points"]  # (N, 4)
        is_v = pts[:, 3] >= row["type_threshold"]
        bounds = [gt.get("edge_points"), gt.get("vertices"),
                  pred.get("edge_points"), pred.get("vertices"), pts[:, :3]]

        # col 0: GT wireframe
        ax = fig.add_subplot(n, 3, r * 3 + 1, projection="3d")
        _draw_wireframe(ax, gt)
        _equal_box(ax, bounds)
        ax.set_title(
            f"{row['name'][:24]}\n{col_titles[0]}: "
            f"{len(gt['vertices'])} V / {len(gt['edge_index'])} E",
            fontsize=8)
        ax.tick_params(labelsize=5)

        # col 1: GT-derived RF target point set, colored by type
        ax = fig.add_subplot(n, 3, r * 3 + 2, projection="3d")
        ep = pts[~is_v, :3]
        vp = pts[is_v, :3]
        if ep.size:
            ax.scatter(ep[:, 0], ep[:, 1], ep[:, 2], s=1.5,
                       color="#1f77b4", alpha=0.35, depthshade=False)
        if vp.size:
            ax.scatter(vp[:, 0], vp[:, 1], vp[:, 2], s=14,
                       color="#d62728", depthshade=False)
        _equal_box(ax, bounds)
        ax.set_title(
            f"{col_titles[1]}\n{int(is_v.sum())} vtx-pts / "
            f"{int((~is_v).sum())} edge-pts",
            fontsize=8)
        ax.tick_params(labelsize=5)

        # col 2: reconstruction + metrics
        ax = fig.add_subplot(n, 3, r * 3 + 3, projection="3d")
        _draw_wireframe(ax, pred)
        _equal_box(ax, bounds)
        m = row["metrics"]
        ax.set_title(
            f"{col_titles[2]}: {len(pred['vertices'])} V / "
            f"{len(pred['edge_index'])} E\n"
            f"score={m['score']:.3f}  TA={m['ta']:.3f}  "
            f"CCD={m['ccd']:.3f}  VPE={m['vpe']:.3f}",
            fontsize=8)
        ax.tick_params(labelsize=5)

    fig.subplots_adjust(top=0.97, bottom=0.02, left=0.02, right=0.98,
                        hspace=0.28, wspace=0.06)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # data location (defaults mirror configs/data.yaml)
    p.add_argument("--data-root", default="data")
    p.add_argument("--edge-subdir", default="train/sample_edge")
    p.add_argument("--pc-subdir", default="train/sample_pointcloud")
    p.add_argument("--split-path", default="data/split.json")
    p.add_argument("--split", default="all",
                   choices=["all", "train", "val", "trainval"])
    # selection
    p.add_argument("--num", type=int, default=6, help="number of shapes")
    p.add_argument("--pick", default="random",
                   choices=["random", "first", "worst"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--files", nargs="*", default=None,
                   help="explicit edge npz paths (overrides --pick/--num)")
    # RF target / reconstruction knobs (defaults match module.py)
    p.add_argument("--wf-num-points", type=int, default=8192)
    p.add_argument("--num-edge-points", type=int, default=32)
    p.add_argument("--type-threshold", type=float, default=0.5)
    p.add_argument("--merge-radius", type=float, default=0.03)
    p.add_argument("--min-votes", type=int, default=3)
    p.add_argument("--num-per-edge", type=int, default=32)
    # metric knobs (defaults match module.py / WireframeScore)
    p.add_argument("--ccd-tau", type=float, default=0.1)
    p.add_argument("--vpe-tau", type=float, default=0.1)
    p.add_argument("--match-thresh", type=float, default=0.1)
    p.add_argument("--w-ccd", type=float, default=0.3)
    p.add_argument("--w-ta", type=float, default=0.4)
    p.add_argument("--w-vpe", type=float, default=0.3)
    # output
    p.add_argument("--out", default="logs/recon_from_gt.png")
    p.add_argument("--no-viz", action="store_true")
    return p.parse_args()


def _build_dataset(args: argparse.Namespace) -> Any:
    return WireframeGraphDataset(
        split=args.split,
        split_path=args.split_path,
        edge_dir=os.path.join(args.data_root, args.edge_subdir),
        pointcloud_dirs=[os.path.join(args.data_root, args.pc_subdir)],
        auto_build_split=True,
        num_edge_points=args.num_edge_points,
        wf_num_points=args.wf_num_points,
        min_edges=1,
    )


def _resolve_file_indices(dataset: Any, files: list[str]) -> list[int]:
    """Map explicit edge paths to dataset indices (by basename stem)."""
    by_stem = {
        os.path.splitext(os.path.basename(f))[0]: i
        for i, f in enumerate(dataset.files)
    }
    out: list[int] = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem in by_stem:
            out.append(by_stem[stem])
        else:
            print(f"[warn] {stem!r} not in split {dataset.split!r}; skipped")
    return out


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)  # arc-length / vertex sampling reproducibility

    dataset = _build_dataset(args)
    if args.files:
        indices = _resolve_file_indices(dataset, args.files)
    else:
        indices = _select_indices(dataset, args.num, args.pick, args.seed)
    if not indices:
        raise SystemExit("No shapes selected.")

    header = (f"{'shape':<42} {'gtV':>4} {'gtE':>4} | "
              f"{'rcV':>4} {'rcE':>4} | {'TA':>5} {'CCD':>6} {'VPE':>6} {'score':>6}")
    print(header)
    print("-" * len(header))

    rows: list[dict[str, Any]] = []
    agg: dict[str, list[float]] = {k: [] for k in
                                   ("ta", "ccd", "vpe", "ccd_score", "vpe_score", "score")}
    for idx in indices:
        sample = dataset[idx]
        name = str(sample["shape_id"])
        gt = _to_np_wf(sample)
        wf_points = sample["wf_points"].numpy()

        pred = reconstruct_wireframe(
            wf_points,
            type_threshold=args.type_threshold,
            merge_radius=args.merge_radius,
            min_votes=args.min_votes,
            num_per_edge=args.num_per_edge,
        )
        m = score_reconstruction(
            pred, gt,
            num_per_edge=args.num_per_edge,
            ccd_tau=args.ccd_tau, vpe_tau=args.vpe_tau,
            match_thresh=args.match_thresh,
            w_ccd=args.w_ccd, w_ta=args.w_ta, w_vpe=args.w_vpe,
        )
        for k in agg:
            agg[k].append(m[k])

        print(f"{name[:42]:<42} {len(gt['vertices']):>4} {len(gt['edge_index']):>4} | "
              f"{len(pred['vertices']):>4} {len(pred['edge_index']):>4} | "
              f"{m['ta']:>5.3f} {m['ccd']:>6.3f} {m['vpe']:>6.3f} {m['score']:>6.3f}")

        rows.append({
            "name": name, "gt": gt, "pred": pred,
            "wf_points": wf_points, "metrics": m,
            "type_threshold": args.type_threshold,
        })

    print("-" * len(header))

    def _ms(vals: list[float]) -> str:
        arr = np.array(vals, dtype=np.float64)
        return f"{arr.mean():.3f}±{arr.std():.3f}"

    print(f"mean±std over {len(rows)} shapes:  "
          f"TA={_ms(agg['ta'])}  CCD={_ms(agg['ccd'])}  "
          f"VPE={_ms(agg['vpe'])}  score={_ms(agg['score'])}")

    if not args.no_viz:
        _visualize(rows, args.out)
        print(f"\nSaved comparison visualization -> {args.out}")


if __name__ == "__main__":
    main()
