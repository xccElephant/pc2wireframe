"""Evaluate / visualise the stage-2 PC2Wireframe model on val.

Loads a stage-2 checkpoint (``PC2WireframeModule``), runs the **validation
protocol** (``sample=False`` -> posterior mode, then ``model._assemble`` with the
chosen ``(ethr, merge_tol)``) over the held-out val shapes, reports the
competition CCD / TA / VPE / weighted-score statistics and saves figures
comparing, per shape:

    input point cloud   |   GT wireframe   |   predicted wireframe

This is headless/server-friendly: it uses the ``Agg`` matplotlib backend and
*saves PNGs* (it never opens a window).

Everything is drawn in the per-shape **normalized** frame (the unit-cube frame
of ``WireframeGraphDataset``), in which the model runs and is supervised, so the
point cloud, GT and prediction are all directly comparable.

Example::

    python scripts/eval_pc2wireframe.py \
        --ckpt logs/pc2wireframe/<run>/checkpoints/epoch=...-val_score=...ckpt \
        --data-config configs/data.yaml \
        --out-dir logs/pc2wireframe/<run>/pc2wireframe_eval
"""
from __future__ import annotations

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never to a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyrootutils  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

root = pyrootutils.setup_root(
    __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
)

from src.data import WireframeDataModule  # noqa: E402
from src.data.dataset import unbatch_wireframe_graphs  # noqa: E402
from src.metrics.functional import (  # noqa: E402
    clamped_distance_to_score,
    curve_chamfer_distance,
    topology_accuracy,
    vertex_position_error,
)
from src.module import PC2WireframeModule  # noqa: E402


# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="stage-2 PC2WireframeModule ckpt")
    ap.add_argument("--curve-vae-ckpt", default=None,
                    help="optional stage-1 ckpt (else frozen weights from ckpt)")
    ap.add_argument("--data-config", default=str(root / "configs/data.yaml"))
    ap.add_argument("--out-dir", default=None,
                    help="default: <ckpt_dir>/../pc2wireframe_eval")
    ap.add_argument("--num-show", type=int, default=12, help="shapes per figure grid")
    ap.add_argument("--max-samples", type=int, default=512,
                    help="cap shapes used for stats")
    ap.add_argument("--ethr", type=float, default=None,
                    help="override module's edge existence threshold")
    ap.add_argument("--merge-tol", type=float, default=None,
                    help="override module's endpoint merge tolerance")
    ap.add_argument("--pc-points-show", type=int, default=2048)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def build_val_loader(data_config: str, num_workers: int, batch_size: int):
    with open(data_config) as f:
        cfg = yaml.safe_load(f)
    init_args = dict(cfg["data"]["init_args"])
    if not os.path.isabs(init_args.get("data_root", "")):
        init_args["data_root"] = str(root / init_args["data_root"])
    if not os.path.isabs(init_args.get("split_path", "")):
        init_args["split_path"] = str(root / init_args["split_path"])
    init_args.update(
        shuffle=False, num_workers=num_workers, batch_size=batch_size,
        persistent_workers=False, use_val=True,
    )
    dm = WireframeDataModule(**init_args)
    dm.setup("validate")
    loader = dm.val_dataloader()
    if loader is None:
        raise SystemExit("val loader is None (use_val=False?); check the data config.")
    return loader


def _gt_from_graph(g: dict) -> dict:
    return {
        "point_cloud": g["point_cloud"].detach().cpu().numpy().astype(np.float32),
        "vertices": g["vertices"].detach().cpu().numpy().astype(np.float32),
        "edge_index": g["edge_index"].detach().cpu().numpy().astype(np.int64),
        "edge_points": g["edge_points"].detach().cpu().numpy().astype(np.float32),
        "shape_id": str(g["shape_id"]),
    }


@torch.no_grad()
def run_eval(model: PC2WireframeModule, loader, device: str, max_samples: int,
             ethr: float, merge_tol: float) -> list[dict]:
    num_per_edge = int(model.hparams.num_per_edge)
    w_ccd = float(model.hparams.eval_w_ccd)
    w_ta = float(model.hparams.eval_w_ta)
    w_vpe = float(model.hparams.eval_w_vpe)
    match_thresh = float(model.hparams.eval_match_thresh)

    results: list[dict] = []
    for batch in loader:
        gts = [_gt_from_graph(g) for g in unbatch_wireframe_graphs(batch)]
        out = model.forward(
            batch["point_cloud"].to(device),
            batch["pc_offset"].to(device),
            sample=False)
        preds = model._assemble(out["preds"], ethr=ethr, merge_tol=merge_tol)
        for gt, pred in zip(gts, preds):
            ccd = curve_chamfer_distance(pred, gt, num_per_edge, device)
            vpe = vertex_position_error(pred, gt, device)
            ta = topology_accuracy(pred, gt, match_thresh, device)
            score = (w_ccd * clamped_distance_to_score(ccd) + w_ta * float(ta)
                     + w_vpe * clamped_distance_to_score(vpe))
            results.append({
                "gt": gt, "pred": pred,
                "ccd": float(ccd) if math.isfinite(ccd) else float("inf"),
                "vpe": float(vpe) if math.isfinite(vpe) else float("inf"),
                "ta": float(ta), "score": float(score),
            })
            if len(results) >= max_samples:
                return results
    return results


# ----------------------------------------------------------------------
def _edge_polylines(wf: dict, num_per_edge: int = 32) -> list[np.ndarray]:
    ep = wf.get("edge_points")
    if ep is not None and len(ep) > 0:
        ep = np.asarray(ep, dtype=np.float32)
        return [ep[i] for i in range(ep.shape[0])]
    verts = np.asarray(wf.get("vertices"), dtype=np.float32).reshape(-1, 3)
    edges = np.asarray(wf.get("edge_index"), dtype=np.int64).reshape(-1, 2)
    t = np.linspace(0.0, 1.0, num_per_edge)[:, None]
    lines = []
    for a, b in edges:
        if a < len(verts) and b < len(verts):
            lines.append(verts[a] * (1.0 - t) + verts[b] * t)
    return lines


def _set_equal_box(ax, pts: np.ndarray) -> None:
    if pts.size == 0:
        return
    c = pts.mean(0)
    r = max(float(np.abs(pts - c).max()), 1e-3)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))


def _plot_pointcloud(ax, pc: np.ndarray, max_points: int, title: str) -> None:
    if pc.shape[0] > max_points:
        idx = np.random.choice(pc.shape[0], size=max_points, replace=False)
        pc = pc[idx]
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1.6, c="#555555", alpha=0.7,
               depthshade=False, linewidths=0)
    _set_equal_box(ax, pc)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=5)
    ax.view_init(elev=22, azim=-60)


def _plot_wireframe(ax, wf: dict, color: str, title: str, num_per_edge: int) -> None:
    verts = np.asarray(wf.get("vertices"), dtype=np.float32).reshape(-1, 3)
    lines = _edge_polylines(wf, num_per_edge)
    for ln in lines:
        ax.plot(ln[:, 0], ln[:, 1], ln[:, 2], "-", color=color, lw=1.2)
    if verts.size:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], color="k", s=3,
                   depthshade=False, linewidths=0)
    allpts = [verts] + lines
    allpts = np.concatenate([p for p in allpts if p.size], axis=0) if any(
        p.size for p in allpts) else np.zeros((0, 3), np.float32)
    _set_equal_box(ax, allpts)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=5)
    ax.view_init(elev=22, azim=-60)


def save_grid(results: list[dict], idx: np.ndarray, out_path: str, suptitle: str,
              pc_points_show: int, num_per_edge: int) -> None:
    n = len(idx)
    fig = plt.figure(figsize=(3 * 3.0, n * 3.0))
    for row, j in enumerate(idx):
        r = results[int(j)]
        gt, pred = r["gt"], r["pred"]
        ax0 = fig.add_subplot(n, 3, row * 3 + 1, projection="3d")
        _plot_pointcloud(ax0, gt["point_cloud"], pc_points_show,
                         f"input PC  ({gt['shape_id']})")
        ax1 = fig.add_subplot(n, 3, row * 3 + 2, projection="3d")
        _plot_wireframe(ax1, gt, "#1f77b4",
                        f"GT  V={len(gt['vertices'])} E={len(gt['edge_index'])}",
                        num_per_edge)
        ax2 = fig.add_subplot(n, 3, row * 3 + 3, projection="3d")
        _plot_wireframe(
            ax2, pred, "#d62728",
            f"Pred  V={len(pred.get('vertices', []))} E={len(pred.get('edge_index', []))}\n"
            f"score={r['score']:.3f} ta={r['ta']:.2f} ccd={r['ccd']:.3f} vpe={r['vpe']:.3f}",
            num_per_edge)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[saved] {out_path}")


def save_hist(results: list[dict], out_path: str) -> None:
    score = np.array([r["score"] for r in results])
    ta = np.array([r["ta"] for r in results])
    ccd = np.array([r["ccd"] for r in results])
    vpe = np.array([r["vpe"] for r in results])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specs = [
        (axes[0, 0], score, "per-shape final score", "#4c72b0", True),
        (axes[0, 1], ta, "topology accuracy (TA)", "#55a868", True),
        (axes[1, 0], ccd[np.isfinite(ccd)], "curve chamfer distance (CCD)", "#c44e52", False),
        (axes[1, 1], vpe[np.isfinite(vpe)], "vertex position error (VPE)", "#8172b3", False),
    ]
    for ax, data, label, color, hi_better in specs:
        if data.size == 0:
            continue
        ax.hist(data, bins=40, color=color, alpha=0.85)
        ax.axvline(float(data.mean()), color="r", ls="--",
                   label=f"mean={data.mean():.4f}")
        ax.axvline(float(np.median(data)), color="g", ls="--",
                   label=f"median={np.median(data):.4f}")
        ax.set_xlabel(label + ("  (higher better)" if hi_better else "  (lower better)"))
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.suptitle("PC2Wireframe per-shape metric distributions on val", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[saved] {out_path}")


# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(args.ckpt)), "pc2wireframe_eval")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[ckpt] {args.ckpt}")
    print(f"[out ] {out_dir}")

    kw = {}
    if args.curve_vae_ckpt:
        kw["curve_vae_ckpt"] = args.curve_vae_ckpt
    model = PC2WireframeModule.load_from_checkpoint(
        args.ckpt, map_location=args.device, **kw)
    model.eval().to(args.device)

    ethr = args.ethr if args.ethr is not None else float(model.hparams.ethr)
    merge_tol = (args.merge_tol if args.merge_tol is not None
                 else float(model.hparams.merge_tol))
    print(f"[decode] ethr={ethr} merge_tol={merge_tol}")

    loader = build_val_loader(args.data_config, args.num_workers, args.batch_size)
    results = run_eval(model, loader, args.device, args.max_samples, ethr, merge_tol)
    if not results:
        raise SystemExit("No val shapes evaluated.")

    score = np.array([r["score"] for r in results])
    ta = np.array([r["ta"] for r in results])
    ccd = np.array([r["ccd"] for r in results])
    vpe = np.array([r["vpe"] for r in results])
    fin_ccd = ccd[np.isfinite(ccd)]
    fin_vpe = vpe[np.isfinite(vpe)]
    print("\n==== PC2Wireframe reconstruction on val (normalized frame) ====")
    print(f"  shapes evaluated : {len(results)}")
    print(f"  score mean/median: {score.mean():.4f} / {np.median(score):.4f}")
    print(f"  TA    mean/median: {ta.mean():.4f} / {np.median(ta):.4f}")
    if fin_ccd.size:
        print(f"  CCD   mean/median: {fin_ccd.mean():.4f} / {np.median(fin_ccd):.4f}")
    if fin_vpe.size:
        print(f"  VPE   mean/median: {fin_vpe.mean():.4f} / {np.median(fin_vpe):.4f}")
    print()

    num_per_edge = int(model.hparams.num_per_edge)
    order = np.argsort(-score)
    n = min(args.num_show, len(results))
    spread = order[np.linspace(0, len(order) - 1, n).astype(int)]
    save_grid(results, spread, os.path.join(out_dir, "recon_spread.png"),
              "PC2Wireframe: input PC | GT (blue) | Pred (red) -- best->worst spread",
              args.pc_points_show, num_per_edge)
    save_grid(results, order[:n], os.path.join(out_dir, "recon_best.png"),
              "PC2Wireframe: input PC | GT (blue) | Pred (red) -- best cases",
              args.pc_points_show, num_per_edge)
    save_grid(results, order[-n:], os.path.join(out_dir, "recon_worst.png"),
              "PC2Wireframe: input PC | GT (blue) | Pred (red) -- worst cases",
              args.pc_points_show, num_per_edge)
    save_hist(results, os.path.join(out_dir, "metrics_hist.png"))


if __name__ == "__main__":
    main()
