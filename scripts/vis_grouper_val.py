#!/usr/bin/env python3
"""Visualize the stage-2 wireframe **grouper** on held-out validation shapes.

The stage-2 grouper (``src/module.py::WireframeGrouperModule`` /
``src/models/wireframe_grouper.py``) is the learned read-out that replaces the
fragile traditional reconstruction: a point set ``(N, 4) = (xyz, type)`` ->
per-point ``{vertex_score, vertex_offset, endpoint_offset, embedding, arclen}``
-> :func:`src.recon.group_wireframe` -> explicit wireframe.

This script checks the trained grouper's reconstruction quality the same way
the training loop's ``validation_step`` does, but renders the result so it can
be eyeballed. For a handful of randomly sampled **validation** shapes it:

  1. builds the *clean* GT-derived point set ``wf_points (N, 4)`` via the real
     dataset code path (``WireframePointDataset``, val split, no augmentation --
     exactly what training validates on);
  2. runs the trained grouper net on it and decodes with ``group_wireframe``
     using the **decode hyper-parameters stored in the checkpoint** (so the
     decode matches the run that produced the reported ``val/score``);
  3. scores Pred vs GT with the same CCD / TA / VPE proxies the training metric
     uses (numpy/scipy port -- no GPU, no pytorch3d, no pytorch-lightning);
  4. renders a 3-column comparison per shape:
     ``GT wireframe | Pred wireframe (grouper) | input point set (by type)``.

Usage (from the project root)::

    # 6 random val shapes with the latest stage-2 run's best checkpoint
    python scripts/vis_grouper_val.py \
        --ckpt logs/pc2wireframe/lpx0b828/checkpoints

    # explicit checkpoint file + more shapes, reproducible pick
    python scripts/vis_grouper_val.py \
        --ckpt logs/pc2wireframe/lpx0b828/checkpoints/last.ckpt \
        --num 8 --pick random --seed 0 --out logs/grouper_val.png

    # the hardest (most-edge) val shapes
    python scripts/vis_grouper_val.py --ckpt <dir-or-ckpt> --pick worst

Only the light parts of ``src`` are imported (the grouper net, the grouped
decoder, the dataset), so this runs without the metric stack's heavy deps.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import re
import sys
from typing import Any

import numpy as np
import torch

# Make the project importable as a package (``src.*``) regardless of cwd.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Light imports only: the grouper net (torch), the grouped decoder
# (numpy/scipy/sklearn) and the dataset (torch/numpy). None of these pull in
# ``src.metrics`` (pytorch3d) or ``src.module`` (pytorch-lightning).
from src.data.dataset import WireframePointDataset  # noqa: E402
from src.models.wireframe_grouper import WireframeGrouper  # noqa: E402
from src.recon import group_wireframe  # noqa: E402


# ----------------------------------------------------------------------
# Checkpoint loading (grouper net + decode hyper-parameters)
# ----------------------------------------------------------------------
def _resolve_ckpt(path: str) -> str:
    """Resolve a checkpoint path: a file is used as-is; a directory is scanned
    for the highest ``val_score`` checkpoint (falling back to ``last.ckpt``)."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        cands = glob.glob(os.path.join(path, "*.ckpt"))
        if not cands:
            raise FileNotFoundError(f"No .ckpt files under {path!r}")
        scored: list[tuple[float, str]] = []
        for c in cands:
            m = re.search(r"val_score=([0-9]*\.?[0-9]+)", os.path.basename(c))
            if m:
                scored.append((float(m.group(1)), c))
        if scored:
            scored.sort()
            return scored[-1][1]
        last = os.path.join(path, "last.ckpt")
        return last if os.path.isfile(last) else cands[0]
    raise FileNotFoundError(f"Checkpoint path not found: {path!r}")


def load_grouper(ckpt_path: str, device: str) -> tuple[WireframeGrouper, dict[str, Any]]:
    """Build the grouper net from a checkpoint and return ``(net, hparams)``.

    The net config (``grouper``) and the decode knobs (``vertex_thresh`` etc.)
    are read straight from the checkpoint's ``hyper_parameters`` so this matches
    the trained run with no manual config plumbing.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters", {}) or {}
    grouper_cfg = hp.get("grouper") or {}
    net = WireframeGrouper(**grouper_cfg)

    state = ck["state_dict"] if "state_dict" in ck else ck
    prefix = "net."
    net_sd = {
        k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
    }
    missing, unexpected = net.load_state_dict(net_sd, strict=False)
    if missing:
        print(f"[warn] missing keys when loading grouper: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys when loading grouper: {unexpected}")

    net.eval().to(device)
    return net, hp


# ----------------------------------------------------------------------
# Decode (grouper fields -> wireframe), mirroring module.decode
# ----------------------------------------------------------------------
@torch.no_grad()
def decode_sample(
    net: WireframeGrouper,
    wf_points: torch.Tensor,
    hp: dict[str, Any],
    device: str,
) -> dict[str, np.ndarray]:
    """Run the grouper on one point set and decode it into a wireframe."""
    pts = wf_points.unsqueeze(0).to(device)  # (1, N, 4)
    out = net(pts)
    xyz = pts[0, :, :3].detach().cpu().numpy()
    fields = {
        "xyz": xyz,
        "vertex_score": out["vertex_score"][0].detach().cpu().numpy(),
        "vertex_offset": out["vertex_offset"][0].detach().cpu().numpy(),
        "endpoint_offset": out["endpoint_offset"][0].detach().cpu().numpy(),
        "embedding": out["embedding"][0].detach().cpu().numpy(),
        "arclen": out["arclen"][0].detach().cpu().numpy(),
    }
    return group_wireframe(
        fields,
        vertex_thresh=float(hp.get("vertex_thresh", 0.5)),
        vertex_merge_radius=float(hp.get("vertex_merge_radius", 0.01)),
        merge_relative=bool(hp.get("vertex_merge_relative", True)),
        split_by_embedding=bool(hp.get("split_by_embedding", True)),
        embed_eps=float(hp.get("embed_eps", 0.5)),
        min_edge_points=int(hp.get("min_edge_points", 3)),
        num_per_edge=int(hp.get("num_per_edge", 32)),
    )


# ----------------------------------------------------------------------
# Lightweight numpy/scipy metrics (same definitions / tau / weights as the
# training metric -- a CPU port that needs neither a GPU nor pytorch3d). This
# is the exact port used by scripts/recon_from_gt.py.
# ----------------------------------------------------------------------
def _flatten_curves(wf: dict[str, np.ndarray], num_per_edge: int) -> np.ndarray:
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
def _select_indices(dataset: Any, num: int, pick: str, seed: int) -> list[int]:
    n = len(dataset)
    num = min(num, n)
    if pick == "first":
        return list(range(num))
    if pick == "worst":
        scored: list[tuple[int, int]] = []
        for i, f in enumerate(dataset.files):
            try:
                with np.load(f, allow_pickle=False) as z:
                    scored.append((int(z["start_verts"].shape[0]), i))
            except Exception:
                continue
        scored.sort(reverse=True)
        return [i for _, i in scored[:num]]
    rng = random.Random(seed)
    return rng.sample(range(n), num)


def _to_np_wf(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "vertices": sample["vertices"].numpy(),
        "edge_index": sample["edge_index"].numpy(),
        "edge_points": sample["edge_points"].numpy(),
    }


# ----------------------------------------------------------------------
# Visualization: GT wireframe | Pred wireframe | input point set (by type)
# ----------------------------------------------------------------------
def _equal_box(ax, pts_list: list[np.ndarray]) -> None:
    allp = [p.reshape(-1, 3) for p in pts_list if p is not None and np.asarray(p).size]
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
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=10, color="#d62728",
                   depthshade=False)


def _visualize(rows: list[dict[str, Any]], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    fig = plt.figure(figsize=(11.5, 3.6 * n))
    col_titles = ["GT wireframe", "Pred wireframe (grouper)", "input point set (type)"]
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

        # col 1: Pred wireframe + metrics
        ax = fig.add_subplot(n, 3, r * 3 + 2, projection="3d")
        _draw_wireframe(ax, pred)
        _equal_box(ax, bounds)
        m = row["metrics"]
        ax.set_title(
            f"{col_titles[1]}: {len(pred['vertices'])} V / "
            f"{len(pred['edge_index'])} E\n"
            f"score={m['score']:.3f}  TA={m['ta']:.3f}  "
            f"CCD={m['ccd']:.3f}  VPE={m['vpe']:.3f}",
            fontsize=8)
        ax.tick_params(labelsize=5)

        # col 2: input point set colored by type
        ax = fig.add_subplot(n, 3, r * 3 + 3, projection="3d")
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
            f"{col_titles[2]}\n{int(is_v.sum())} vtx-pts / "
            f"{int((~is_v).sum())} edge-pts",
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
    # checkpoint
    p.add_argument(
        "--ckpt",
        default="logs/pc2wireframe/lpx0b828/checkpoints",
        help="grouper checkpoint file, or a dir (best val_score is auto-picked)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # data location (defaults mirror configs/grouper_data.yaml + data.yaml)
    p.add_argument("--data-root", default="data")
    p.add_argument("--edge-subdir", default="train/sample_edge")
    p.add_argument("--pc-subdir", default="train/sample_pointcloud")
    p.add_argument("--split-path", default="data/split.json")
    p.add_argument("--split", default="val",
                   choices=["val", "train", "all", "trainval"])
    # selection
    p.add_argument("--num", type=int, default=6, help="number of shapes")
    p.add_argument("--pick", default="random",
                   choices=["random", "first", "worst"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--files", nargs="*", default=None,
                   help="explicit edge npz paths (overrides --pick/--num)")
    # RF target format (must match the trained run / grouper_data.yaml)
    p.add_argument("--wf-num-points", type=int, default=8192)
    p.add_argument("--num-edge-points", type=int, default=32)
    p.add_argument("--type-threshold", type=float, default=0.5,
                   help="only used to color the input point set")
    # metric knobs (defaults match module.py / WireframeScore)
    p.add_argument("--ccd-tau", type=float, default=0.1)
    p.add_argument("--vpe-tau", type=float, default=0.1)
    p.add_argument("--match-thresh", type=float, default=0.1)
    p.add_argument("--w-ccd", type=float, default=0.3)
    p.add_argument("--w-ta", type=float, default=0.4)
    p.add_argument("--w-vpe", type=float, default=0.3)
    # output
    p.add_argument("--out", default="logs/grouper_val.png")
    p.add_argument("--no-viz", action="store_true")
    return p.parse_args()


def _build_dataset(args: argparse.Namespace) -> WireframePointDataset:
    # Val split, no augmentation -- exactly what training validates on.
    return WireframePointDataset(
        split=args.split,
        split_path=args.split_path,
        edge_dir=os.path.join(args.data_root, args.edge_subdir),
        pointcloud_dirs=[os.path.join(args.data_root, args.pc_subdir)],
        auto_build_split=True,
        num_edge_points=args.num_edge_points,
        wf_num_points=args.wf_num_points,
        min_edges=1,
        jitter_std=0.0,
        type_noise_std=0.0,
    )


def _resolve_file_indices(dataset: Any, files: list[str]) -> list[int]:
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

    ckpt_path = _resolve_ckpt(args.ckpt)
    print(f"Loading grouper checkpoint: {ckpt_path}")
    net, hp = load_grouper(ckpt_path, args.device)
    num_per_edge = int(hp.get("num_per_edge", args.num_edge_points))

    dataset = _build_dataset(args)
    if args.files:
        indices = _resolve_file_indices(dataset, args.files)
    else:
        indices = _select_indices(dataset, args.num, args.pick, args.seed)
    if not indices:
        raise SystemExit("No shapes selected.")

    header = (f"{'shape':<42} {'gtV':>4} {'gtE':>4} | "
              f"{'prV':>4} {'prE':>4} | {'TA':>5} {'CCD':>6} {'VPE':>6} {'score':>6}")
    print(header)
    print("-" * len(header))

    rows: list[dict[str, Any]] = []
    agg: dict[str, list[float]] = {k: [] for k in
                                   ("ta", "ccd", "vpe", "ccd_score", "vpe_score", "score")}
    for idx in indices:
        sample = dataset[idx]
        name = str(sample["shape_id"])
        gt = _to_np_wf(sample)
        wf_points = sample["wf_points"]  # torch (N, 4)

        pred = decode_sample(net, wf_points, hp, args.device)
        m = score_reconstruction(
            pred, gt,
            num_per_edge=num_per_edge,
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
            "wf_points": wf_points.numpy(), "metrics": m,
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
        print(f"\nSaved GT-vs-Pred comparison -> {args.out}")


if __name__ == "__main__":
    main()
