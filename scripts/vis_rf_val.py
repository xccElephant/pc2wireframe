#!/usr/bin/env python3
"""Visualize the stage-1 **Rectified-Flow** point set on held-out val shapes.

Stage 1 (``src/module.py::RFWireframeModule``) is

    raw point cloud (coord (P_sum,3), offset (B,))
        -> UtoniaEncoder (frozen PTv3 + trainable compressor) -> latent z
    noise x0 ~ N(0,I) (B,N,3)
        -> RFPointSetVelocity (point-set DiT), conditioned on z
        -> ODE integrate t:0->1 -> wireframe anchor point set x1_hat (B,N,3)=xyz

The training loop only logs the flow-matching ``val/loss`` -- it never actually
*looks* at what the ODE samples. This script does: for a handful of randomly
sampled **validation** shapes it

  1. builds the real dataset sample (``WireframeGraphDataset``, val split, raw
     data, no augmentation -- exactly what training validates on), giving the
     input point cloud + the **GT anchor point set** ``wf_points (N,3)``;
  2. runs the trained encoder + RF velocity net and ODE-samples the **predicted
     anchor set** with the *exact* sampler / hyper-parameters stored in the
     checkpoint (``ode_steps`` / ``ode_method`` / ``sample_seed``);
  3. reports per-shape diagnostics (xyz Chamfer between GT & Pred anchor sets,
     plus the xyz spread to spot a collapsed/blob RF) so a broken stage 1 is
     obvious from the numbers, not just the picture;
  4. renders a 3-column comparison per shape:
     ``input point cloud | GT anchor set | Pred anchor set``.

Only the light parts of ``src`` are imported (the encoder, the velocity net,
the dataset); the LightningModule / metric stack (pytorch3d, pytorch-lightning)
is NOT pulled in. The encoder still needs the Utonia PTv3 backbone deps (spconv
/ flash-attn / torch_scatter), since stage 1 cannot run without encoding the
point cloud.

Usage (from the project root)::

    # 6 random val shapes with this stage-1 checkpoint dir (best val_loss auto-picked)
    python scripts/vis_rf_val.py \
        --ckpt logs/pc2wireframe/v4ay6feo/checkpoints

    # explicit checkpoint file + more shapes, reproducible pick
    python scripts/vis_rf_val.py \
        --ckpt logs/pc2wireframe/v4ay6feo/checkpoints/last.ckpt \
        --num 8 --pick random --seed 0 --out logs/rf_val.png
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

# Light imports only: the encoder + velocity net (torch) and the dataset
# (torch/numpy). None of these pull in ``src.metrics`` (pytorch3d) or
# ``src.module`` (pytorch-lightning). The Utonia backbone deps are still needed
# by the encoder at construction / forward time.
from src.data.dataset import WireframeGraphDataset, collate_rf_batch  # noqa: E402
from src.models.utonia_encoder import UtoniaEncoder  # noqa: E402
from src.models.rf_pointset import RFPointSetVelocity  # noqa: E402


# ----------------------------------------------------------------------
# Checkpoint loading (encoder + RF velocity net + sampler hyper-parameters)
# ----------------------------------------------------------------------
def _resolve_ckpt(path: str) -> str:
    """Resolve a checkpoint path: a file is used as-is; a directory is scanned
    for the lowest ``val_loss`` checkpoint (falling back to ``last.ckpt``)."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        cands = glob.glob(os.path.join(path, "*.ckpt"))
        if not cands:
            raise FileNotFoundError(f"No .ckpt files under {path!r}")
        scored: list[tuple[float, str]] = []
        for c in cands:
            m = re.search(r"val_loss=([0-9]*\.?[0-9]+)", os.path.basename(c))
            if m:
                scored.append((float(m.group(1)), c))
        if scored:
            scored.sort()  # lowest val_loss first
            return scored[0][1]
        last = os.path.join(path, "last.ckpt")
        return last if os.path.isfile(last) else cands[0]
    raise FileNotFoundError(f"Checkpoint path not found: {path!r}")


def _strip_prefix(state: dict[str, torch.Tensor], prefix: str
                  ) -> dict[str, torch.Tensor]:
    return {
        k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
    }


def load_stage1(ckpt_path: str, device: str
                ) -> tuple[UtoniaEncoder, RFPointSetVelocity, dict[str, Any]]:
    """Build the encoder + RF velocity net from a checkpoint.

    The model config (``pc_encoder`` / ``rf_net``) and the sampler knobs
    (``ode_steps`` / ``ode_method`` / ``sample_seed`` / ``wf_num_points``) are
    read straight from the checkpoint's ``hyper_parameters`` so this matches the
    trained run with no manual config plumbing.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters", {}) or {}

    encoder = UtoniaEncoder(**(hp.get("pc_encoder") or {}))
    net = RFPointSetVelocity(**(hp.get("rf_net") or {}))

    state = ck["state_dict"] if "state_dict" in ck else ck
    enc_sd = _strip_prefix(state, "encoder.")
    net_sd = _strip_prefix(state, "net.")

    miss_e, unexp_e = encoder.load_state_dict(enc_sd, strict=False)
    miss_n, unexp_n = net.load_state_dict(net_sd, strict=False)
    # The frozen Utonia backbone is often not re-saved in the ckpt (it is
    # reloaded from logs/utonia/utonia.pth at construction), so missing backbone
    # keys are expected; only warn about the trainable compressor / net.
    comp_missing = [k for k in miss_e if not k.startswith("backbone.")]
    if comp_missing:
        print(f"[warn] missing encoder (compressor) keys: {comp_missing}")
    if unexp_e:
        print(f"[warn] unexpected encoder keys: {unexp_e[:8]}"
              f"{' ...' if len(unexp_e) > 8 else ''}")
    if miss_n:
        print(f"[warn] missing net keys: {miss_n}")
    if unexp_n:
        print(f"[warn] unexpected net keys: {unexp_n}")

    encoder.eval().to(device)
    net.eval().to(device)
    return encoder, net, hp


# ----------------------------------------------------------------------
# Sampling: replicate RFWireframeModule.sample (no pytorch-lightning needed)
# ----------------------------------------------------------------------
@torch.no_grad()
def rf_sample(
    net: RFPointSetVelocity,
    z: torch.Tensor,
    *,
    num_points: int,
    ode_steps: int,
    ode_method: str,
    sample_seed: int,
) -> torch.Tensor:
    """Deterministic ODE sampling ``x0 -> x1`` conditioned on ``z``.

    Mirrors ``src/module.py::RFWireframeModule.sample`` exactly (fixed-seed x0,
    ``dx/dt = net(t, x, z)`` integrated t: 0 -> 1). Returns the pure-xyz anchor
    point set ``x1_hat (B, N, 3)``.
    """
    from torchdiffeq import odeint

    b = z.shape[0]
    n = int(num_points)
    point_dim = int(net.point_dim)
    gen = torch.Generator(device=z.device)
    gen.manual_seed(int(sample_seed))
    x0 = torch.randn(b, n, point_dim, device=z.device, dtype=z.dtype, generator=gen)

    def func(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return net(t.reshape(()).expand(b), x, z)

    t_span = torch.linspace(0.0, 1.0, int(ode_steps) + 1, device=z.device, dtype=z.dtype)
    traj = odeint(func, x0, t_span, method=str(ode_method))
    return traj[-1]


# ----------------------------------------------------------------------
# Diagnostics (numpy/scipy -- no GPU, no pytorch3d)
# ----------------------------------------------------------------------
def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("inf")
    da, _ = cKDTree(b).query(a, k=1)
    db, _ = cKDTree(a).query(b, k=1)
    return 0.5 * (float(np.mean(da)) + float(np.mean(db)))


def diagnose(gt_pts: np.ndarray, pred_pts: np.ndarray) -> dict[str, float]:
    """Per-shape stage-1 diagnostics comparing GT vs Pred anchor sets ``(N,3)``."""
    gt_xyz, pred_xyz = gt_pts[:, :3], pred_pts[:, :3]
    return {
        "cd_all": _chamfer(gt_xyz, pred_xyz),
        # spread of the predicted xyz vs the GT (a collapsed RF often shrinks
        # the cloud toward the mean / a blob -> much smaller std).
        "gt_xyz_std": float(gt_xyz.std()),
        "pred_xyz_std": float(pred_xyz.std()),
    }


# ----------------------------------------------------------------------
# Sample selection
# ----------------------------------------------------------------------
def _select_indices(dataset: Any, num: int, pick: str, seed: int) -> list[int]:
    n = len(dataset)
    num = min(num, n)
    if pick == "first":
        return list(range(num))
    rng = random.Random(seed)
    return rng.sample(range(n), num)


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


# ----------------------------------------------------------------------
# Visualization: input point cloud | GT anchor set | Pred anchor set
#
# The stage-1 target is a pure-xyz anchor cloud, so anchor points are drawn in
# a single color (no vertex/edge type split).
# ----------------------------------------------------------------------
_ANCHOR_C = "#3a7ca5"   # anchor points -- cool blue
_PC_C = "#9aa0a6"       # input surface cloud -- neutral gray


def _equal_box(ax, pts_list: list[np.ndarray]) -> None:
    allp = [p.reshape(-1, 3) for p in pts_list if p is not None and np.asarray(p).size]
    if not allp:
        ax.set_box_aspect((1, 1, 1))
        return
    cat = np.concatenate(allp, axis=0)
    lo, hi = cat.min(0), cat.max(0)
    c = (lo + hi) * 0.5
    r = float((hi - lo).max()) * 0.5 or 1.0
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))


def _style_ax(ax, args) -> None:
    ax.view_init(elev=args.elevation, azim=args.azimuth)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=args.zoom)
    except TypeError:  # older matplotlib without the zoom kwarg
        ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)


def _scatter_anchors(ax, pts: np.ndarray, args) -> None:
    xyz = pts[:, :3]
    if xyz.size:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=args.point_size,
                   color=_ANCHOR_C, alpha=args.alpha, depthshade=False,
                   linewidths=0)


def _visualize(rows: list[dict[str, Any]], out_path: str, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    panel = args.panel
    fig = plt.figure(figsize=(3 * panel, n * panel))
    fig.patch.set_facecolor("white")
    col_titles = ["input point cloud", "GT anchor set", "Pred anchor set (RF)"]
    for r, row in enumerate(rows):
        pc = row["point_cloud"]      # (P, 3)
        gt = row["gt_points"]        # (N, 3)
        pred = row["pred_points"]    # (N, 3)
        d = row["diag"]
        bounds = [pc, gt[:, :3], pred[:, :3]]

        # col 0: input surface point cloud
        ax = fig.add_subplot(n, 3, r * 3 + 1, projection="3d")
        if pc.size:
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=args.point_size,
                       color=_PC_C, alpha=args.alpha, depthshade=False,
                       linewidths=0)
        _equal_box(ax, bounds)
        _style_ax(ax, args)
        ax.set_title(f"{row['name'][:26]}\n{col_titles[0]}: {len(pc)} pts",
                     fontsize=8)

        # col 1: GT anchor point set
        ax = fig.add_subplot(n, 3, r * 3 + 2, projection="3d")
        _scatter_anchors(ax, gt, args)
        _equal_box(ax, bounds)
        _style_ax(ax, args)
        ax.set_title(
            f"{col_titles[1]}\n{len(gt)} pts  (std={d['gt_xyz_std']:.3f})",
            fontsize=8)

        # col 2: Pred anchor point set + diagnostics
        ax = fig.add_subplot(n, 3, r * 3 + 3, projection="3d")
        _scatter_anchors(ax, pred, args)
        _equal_box(ax, bounds)
        _style_ax(ax, args)
        ax.set_title(
            f"{col_titles[2]}\n{len(pred)} pts  (std={d['pred_xyz_std']:.3f})\n"
            f"CD={d['cd_all']:.4f}",
            fontsize=8)

    fig.subplots_adjust(top=0.96, bottom=0.01, left=0.01, right=0.99,
                        hspace=0.18, wspace=0.02)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.1,
                facecolor="white")
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
        default="logs/pc2wireframe/v4ay6feo/checkpoints",
        help="stage-1 checkpoint file, or a dir (best val_loss auto-picked)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # data location (defaults mirror configs/data.yaml)
    p.add_argument("--data-root", default="data")
    p.add_argument("--edge-subdir", default="train/sample_edge")
    p.add_argument("--pc-subdir", default="train/sample_pointcloud")
    p.add_argument("--split-path", default="data/split.json")
    p.add_argument("--split", default="val",
                   choices=["val", "train", "all", "trainval"])
    # selection
    p.add_argument("--num", type=int, default=6, help="number of shapes")
    p.add_argument("--pick", default="random", choices=["random", "first"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--files", nargs="*", default=None,
                   help="explicit edge npz paths/stems (overrides --pick/--num)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="shapes encoded/sampled per forward pass")
    # RF target format (must match the trained run / data.yaml)
    p.add_argument("--wf-num-points", type=int, default=8192)
    p.add_argument("--num-edge-points", type=int, default=32)
    p.add_argument("--max-pc-points", type=int, default=0,
                   help="cap input cloud size (0 = native); only a memory knob")
    # sampler overrides (default: read from ckpt hyper-parameters)
    p.add_argument("--ode-steps", type=int, default=None)
    p.add_argument("--ode-method", default=None)
    p.add_argument("--sample-seed", type=int, default=None)
    # visualization style
    p.add_argument("--azimuth", type=float, default=-60.0)
    p.add_argument("--elevation", type=float, default=18.0)
    p.add_argument("--point-size", type=float, default=1.6)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--panel", type=float, default=4.5,
                   help="size (inches) of each subplot")
    p.add_argument("--zoom", type=float, default=1.2,
                   help="3D fill factor (>1 zooms in)")
    p.add_argument("--dpi", type=int, default=150)
    # output
    p.add_argument("--out", default="logs/rf_val.png")
    p.add_argument("--no-viz", action="store_true")
    return p.parse_args()


def _build_dataset(args: argparse.Namespace) -> WireframeGraphDataset:
    # Val split, raw data, no augmentation -- exactly what training validates on.
    return WireframeGraphDataset(
        split=args.split,
        split_path=args.split_path,
        edge_dir=os.path.join(args.data_root, args.edge_subdir),
        pointcloud_dirs=[os.path.join(args.data_root, args.pc_subdir)],
        auto_build_split=True,
        num_edge_points=args.num_edge_points,
        wf_num_points=args.wf_num_points,
        max_pc_points=args.max_pc_points,
        min_edges=1,
    )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)  # arc-length / vertex sampling reproducibility

    ckpt_path = _resolve_ckpt(args.ckpt)
    print(f"Loading stage-1 checkpoint: {ckpt_path}")
    encoder, net, hp = load_stage1(ckpt_path, args.device)

    num_points = int(hp.get("wf_num_points", args.wf_num_points))
    ode_steps = int(args.ode_steps if args.ode_steps is not None
                    else hp.get("ode_steps", 50))
    ode_method = str(args.ode_method if args.ode_method is not None
                     else hp.get("ode_method", "euler"))
    sample_seed = int(args.sample_seed if args.sample_seed is not None
                      else hp.get("sample_seed", 0))
    print(f"Sampler: N={num_points}  steps={ode_steps}  method={ode_method}  "
          f"seed={sample_seed}")

    dataset = _build_dataset(args)
    if args.files:
        indices = _resolve_file_indices(dataset, args.files)
    else:
        indices = _select_indices(dataset, args.num, args.pick, args.seed)
    if not indices:
        raise SystemExit("No shapes selected.")

    header = (f"{'shape':<42} {'CD':>7} | {'gStd':>6} {'pStd':>6}")
    print(header)
    print("-" * len(header))

    rows: list[dict[str, Any]] = []
    agg: dict[str, list[float]] = {k: [] for k in ("cd_all",)}

    # Process in small batches: encode the packed point cloud + ODE-sample.
    for start in range(0, len(indices), max(1, args.batch_size)):
        chunk = indices[start:start + max(1, args.batch_size)]
        samples = [dataset[i] for i in chunk]
        batch = collate_rf_batch(samples)
        pc = batch["point_cloud"].to(args.device)
        offset = batch["pc_offset"].to(args.device)

        with torch.no_grad():
            z = encoder(pc, offset)
            pred = rf_sample(
                net, z, num_points=num_points, ode_steps=ode_steps,
                ode_method=ode_method, sample_seed=sample_seed)
        pred = pred.detach().cpu().numpy()

        for j, s in enumerate(samples):
            name = str(s["shape_id"])
            gt_pts = s["wf_points"].numpy()        # (N, 3)
            pred_pts = pred[j]                     # (N, 3)
            pc_np = s["point_cloud"].numpy()       # (P, 3)
            d = diagnose(gt_pts, pred_pts)
            for k in agg:
                agg[k].append(d[k])

            print(f"{name[:42]:<42} {d['cd_all']:>7.4f} | "
                  f"{d['gt_xyz_std']:>6.3f} {d['pred_xyz_std']:>6.3f}")

            rows.append({
                "name": name, "point_cloud": pc_np,
                "gt_points": gt_pts, "pred_points": pred_pts, "diag": d,
            })

    print("-" * len(header))

    def _ms(vals: list[float]) -> str:
        arr = np.array(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return "n/a"
        return f"{arr.mean():.4f}±{arr.std():.4f}"

    print(f"mean±std over {len(rows)} shapes:  CD={_ms(agg['cd_all'])}")

    if not args.no_viz:
        _visualize(rows, args.out, args)
        print(f"\nSaved GT-vs-Pred point-set comparison -> {args.out}")


if __name__ == "__main__":
    main()
