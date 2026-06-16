"""Evaluate / visualise the stage-2 PC2Wireframe model on val.

Loads a stage-2 checkpoint (``PC2WireframeModule``) plus the frozen stage-1
curve-VAE checkpoint, runs the **validation protocol** (``sample=False`` ->
posterior mode, then ``model.reconstruct``) over the held-out val shapes,
reports the competition CCD / TA / VPE / weighted-score statistics and saves
figures comparing, per shape:

    input point cloud   |   GT wireframe   |   predicted wireframe

This is headless/server-friendly: it uses the ``Agg`` matplotlib backend and
*saves PNGs* (it never opens a window).

Everything is drawn in the per-shape **normalized** frame (the unit-cube
transform of ``WireframeGraphDataset``), in which the model runs and is
supervised, so the point cloud, GT and prediction are all directly comparable
without any denormalisation.

Example::

    python scripts/eval_pc2wireframe.py \
        --ckpt logs/pc2wireframe/ujtuvalu/checkpoints/epoch=109-val_score=0.1477.ckpt \
        --curve-vae-ckpt logs/pc2wireframe/adpdhtsg/checkpoints/epoch=049-val_loss=0.0058.ckpt \
        --data-config configs/data.yaml \
        --out-dir logs/pc2wireframe/ujtuvalu/pc2wireframe_eval
"""
from __future__ import annotations

import argparse
import glob
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
    curve_chamfer_distance,
    distance_to_score,
    topology_accuracy,
    vertex_position_error,
)
from src.module import PC2WireframeModule  # noqa: E402


# ----------------------------------------------------------------------
def _best_ckpt(log_subdir: str, key: str, mode: str) -> str:
    """Pick the best checkpoint under ``logs/pc2wireframe/<log_subdir>``.

    ``key`` is the metric encoded in the filename (e.g. ``val_score`` /
    ``val_loss``); ``mode`` is ``"max"`` or ``"min"``.
    """
    pat = str(root / f"logs/pc2wireframe/{log_subdir}/checkpoints/epoch=*{key}=*.ckpt")
    cands = glob.glob(pat)
    if not cands:
        raise SystemExit(f"No checkpoint found matching {pat!r}; pass it explicitly.")

    def _metric(p: str) -> float:
        try:
            return float(p.split(f"{key}=")[-1].split(".ckpt")[0])
        except ValueError:
            return float("inf") if mode == "min" else float("-inf")

    return (min if mode == "min" else max)(cands, key=_metric)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None,
                    help="stage-2 ckpt (default: best val_score under logs/.../ujtuvalu)")
    ap.add_argument("--curve-vae-ckpt", default=None,
                    help="stage-1 curve VAE ckpt (default: best val_loss under logs/.../adpdhtsg)")
    ap.add_argument("--data-config", default=str(root / "configs/data.yaml"))
    ap.add_argument("--out-dir", default=None,
                    help="default: <ckpt_dir>/../pc2wireframe_eval")
    ap.add_argument("--num-show", type=int, default=12, help="shapes per figure grid")
    ap.add_argument("--max-samples", type=int, default=512,
                    help="cap shapes used for stats")
    ap.add_argument("--vertex-threshold", type=float, default=None,
                    help="override module's vertex existence threshold")
    ap.add_argument("--edge-threshold", type=float, default=None,
                    help="override module's edge existence threshold")
    ap.add_argument("--pc-points-show", type=int, default=2048,
                    help="point-cloud points to scatter per panel")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


# ----------------------------------------------------------------------
def build_val_loader(data_config: str, num_workers: int, batch_size: int):
    with open(data_config) as f:
        cfg = yaml.safe_load(f)
    init_args = dict(cfg["data"]["init_args"])
    # data_root / split_path in the config are relative to the project root.
    if not os.path.isabs(init_args.get("data_root", "")):
        init_args["data_root"] = str(root / init_args["data_root"])
    if not os.path.isabs(init_args.get("split_path", "")):
        init_args["split_path"] = str(root / init_args["split_path"])
    # Deterministic, lighter loading for evaluation.
    init_args.update(
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
        persistent_workers=False,
        use_val=True,
    )
    dm = WireframeDataModule(**init_args)
    dm.setup("validate")
    loader = dm.val_dataloader()
    if loader is None:
        raise SystemExit("val loader is None (use_val=False?); check the data config.")
    return loader


def _gt_from_graph(g: dict) -> dict:
    """One unbatched GT graph -> numpy wireframe dict (normalized frame)."""
    return {
        "point_cloud": g["point_cloud"].detach().cpu().numpy().astype(np.float32),
        "vertices": g["vertices"].detach().cpu().numpy().astype(np.float32),
        "edge_index": g["edge_index"].detach().cpu().numpy().astype(np.int64),
        "edge_points": g["edge_points"].detach().cpu().numpy().astype(np.float32),
        "shape_id": str(g["shape_id"]),
    }


@torch.no_grad()
def run_eval(model: PC2WireframeModule, loader, device: str, max_samples: int,
             vertex_threshold: float, edge_threshold: float) -> list[dict]:
    """Validation-protocol reconstruction + per-shape metrics over val shapes.

    Returns a list of per-shape dicts holding the input point cloud, GT
    wireframe, predicted wireframe and the CCD / TA / VPE / score metrics.
    """
    net = model.model
    num_per_edge = int(model.hparams.eval_num_per_edge)
    w_ccd = float(model.hparams.eval_w_ccd)
    w_ta = float(model.hparams.eval_w_ta)
    w_vpe = float(model.hparams.eval_w_vpe)
    ccd_tau = float(model.hparams.eval_ccd_tau)
    vpe_tau = float(model.hparams.eval_vpe_tau)
    match_thresh = float(model.hparams.eval_match_thresh)

    results: list[dict] = []
    for batch in loader:
        gts = [_gt_from_graph(g) for g in unbatch_wireframe_graphs(batch)]
        point_cloud = batch["point_cloud"].to(device)
        out = net(point_cloud, sample=False)
        preds = net.reconstruct(
            out["preds"],
            vertex_threshold=vertex_threshold,
            edge_threshold=edge_threshold,
            num_points=num_per_edge,
        )
        for gt, pred in zip(gts, preds):
            ccd = curve_chamfer_distance(pred, gt, num_per_edge, device)
            vpe = vertex_position_error(pred, gt, device)
            ta = topology_accuracy(pred, gt, match_thresh, device)
            ccd_score = distance_to_score(ccd, ccd_tau)
            vpe_score = distance_to_score(vpe, vpe_tau)
            score = w_ccd * ccd_score + w_ta * ta + w_vpe * vpe_score
            results.append({
                "gt": gt,
                "pred": pred,
                "ccd": float(ccd) if math.isfinite(ccd) else float("inf"),
                "vpe": float(vpe) if math.isfinite(vpe) else float("inf"),
                "ta": float(ta),
                "score": float(score),
            })
            if len(results) >= max_samples:
                return results
    return results


# ----------------------------------------------------------------------
def _edge_polylines(wf: dict, num_per_edge: int = 32) -> list[np.ndarray]:
    """Per-edge polylines for a wireframe: dense ``edge_points`` if present,
    else straight segments between the endpoints of ``edge_index``."""
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
            f"Pred  V={pred.get('num_vertices', 0)} E={pred.get('num_edges', 0)}\n"
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

    ckpt = args.ckpt or _best_ckpt("ujtuvalu", "val_score", "max")
    curve_vae_ckpt = args.curve_vae_ckpt or _best_ckpt("adpdhtsg", "val_loss", "min")
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(ckpt)), "pc2wireframe_eval")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[ckpt]          {ckpt}")
    print(f"[curve_vae_ckpt]{curve_vae_ckpt}")
    print(f"[out ]          {out_dir}")

    # Override the saved curve_vae_ckpt hparam so loading does not depend on the
    # (possibly stale) path captured at train time -- the frozen curve VAE
    # weights are restored from the stage-2 state_dict anyway.
    model = PC2WireframeModule.load_from_checkpoint(
        ckpt, map_location=args.device, curve_vae_ckpt=curve_vae_ckpt)
    model.eval().to(args.device)

    v_thr = args.vertex_threshold if args.vertex_threshold is not None \
        else float(model.hparams.vertex_threshold)
    e_thr = args.edge_threshold if args.edge_threshold is not None \
        else float(model.hparams.edge_threshold)
    print(f"[thresholds] vertex={v_thr} edge={e_thr}")

    loader = build_val_loader(args.data_config, args.num_workers, args.batch_size)
    results = run_eval(model, loader, args.device, args.max_samples, v_thr, e_thr)
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
    print(f"  CCD   mean/median: {fin_ccd.mean():.4f} / {np.median(fin_ccd):.4f}"
          if fin_ccd.size else "  CCD   : all non-finite")
    print(f"  VPE   mean/median: {fin_vpe.mean():.4f} / {np.median(fin_vpe):.4f}"
          if fin_vpe.size else "  VPE   : all non-finite")
    print(f"  score p10/p90    : {np.percentile(score, 10):.4f} / "
          f"{np.percentile(score, 90):.4f}\n")

    num_per_edge = int(model.hparams.eval_num_per_edge)
    # rank by score (higher is better)
    order = np.argsort(-score)
    n = min(args.num_show, len(results))

    # representative spread: evenly spaced over the (best->worst) ranking
    spread = order[np.linspace(0, len(order) - 1, n).astype(int)]
    save_grid(results, spread, os.path.join(out_dir, "recon_spread.png"),
              "PC2Wireframe: input PC | GT (blue) | Pred (red) -- best->worst spread",
              args.pc_points_show, num_per_edge)

    # best cases
    save_grid(results, order[:n], os.path.join(out_dir, "recon_best.png"),
              "PC2Wireframe: input PC | GT (blue) | Pred (red) -- best cases",
              args.pc_points_show, num_per_edge)

    # worst cases
    save_grid(results, order[-n:], os.path.join(out_dir, "recon_worst.png"),
              "PC2Wireframe: input PC | GT (blue) | Pred (red) -- worst cases",
              args.pc_points_show, num_per_edge)

    save_hist(results, os.path.join(out_dir, "metrics_hist.png"))


if __name__ == "__main__":
    main()
