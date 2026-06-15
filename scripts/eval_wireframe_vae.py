"""Evaluate / visualise the stage-2 wireframe VAE (``AutoencoderKLWireframe``).

Loads a stage-2 checkpoint and runs the **validation protocol** of
``WireframeVAEModule`` (encode GT wireframe -> posterior mode -> decode ->
reconstruct) over the held-out val wireframes, reports the competition metrics
(CCD / TA / VPE + weighted score) and saves 3-D figures comparing each decoded
wireframe against its ground truth.

This mirrors ``scripts/eval_curve_vae.py`` but for the full wireframe VAE: the
curve VAE is loaded (frozen) from the same checkpoint and is only used to encode
GT curves into the per-curve latent targets and to decode them back into 3-D
polylines during ``reconstruct``.

This is headless/server-friendly: it uses the ``Agg`` matplotlib backend and
*saves PNGs* (it never opens a window).

Example::

    python scripts/eval_wireframe_vae.py \
        --ckpt logs/pc2wireframe/6d7yjfdy/checkpoints/epoch=099-val_score=0.1133.ckpt \
        --data-config configs/data.yaml \
        --out-dir logs/pc2wireframe/6d7yjfdy/wireframe_vae_eval
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never to a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyrootutils  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from einops import rearrange  # noqa: E402

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
from src.module import WireframeVAEModule  # noqa: E402


# ----------------------------------------------------------------------
def _default_ckpt() -> str:
    """Pick the best (highest val_score) checkpoint under the default log dir."""
    pat = str(root / "logs/pc2wireframe/*/checkpoints/epoch=*val_score=*.ckpt")
    cands = glob.glob(pat)
    if not cands:
        raise SystemExit(
            f"No checkpoint found matching {pat!r}; pass --ckpt explicitly."
        )

    # filename encodes val_score; higher is better.
    def _val_score(p: str) -> float:
        try:
            return float(p.split("val_score=")[-1].split(".ckpt")[0])
        except ValueError:
            return float("-inf")

    return max(cands, key=_val_score)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None, help="stage-2 ckpt (default: best under logs/)")
    ap.add_argument("--data-config", default=str(root / "configs/data.yaml"))
    ap.add_argument("--out-dir", default=None, help="default: <ckpt_dir>/../wireframe_vae_eval")
    ap.add_argument("--num-show", type=int, default=12, help="wireframes per figure grid")
    ap.add_argument("--max-shapes", type=int, default=512, help="cap shapes used for stats")
    ap.add_argument("--num-per-edge", type=int, default=32, help="points/edge for decode + CCD")
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


def _batch_to_device(batch: dict, device: str) -> dict:
    """Move all tensor entries of a packed batch onto ``device`` (in place)."""
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def _gt_wireframe_numpy(graph: dict) -> dict:
    """One GT graph (from ``unbatch_wireframe_graphs``) -> numpy wireframe dict."""
    return {
        "vertices": graph["vertices"].detach().cpu().numpy(),
        "edge_index": graph["edge_index"].detach().cpu().numpy(),
        "edge_points": graph["edge_points"].detach().cpu().numpy(),
    }


@torch.no_grad()
def run_eval(
    model: WireframeVAEModule,
    loader,
    device: str,
    max_shapes: int,
    num_per_edge: int,
):
    """Validation-protocol reconstruction over val wireframes.

    Returns dict with per-shape metrics (numpy arrays of length N) and the raw
    pred / GT wireframes (lists) needed for plotting.
    """
    inner = model.model  # ClrWireframeBase

    preds_all: list[dict] = []
    gts_all: list[dict] = []
    ccd_l, vpe_l, ta_l = [], [], []
    ccd_s_l, vpe_s_l = [], []
    n_pred_e, n_gt_e = [], []
    shape_ids: list[str] = []

    seen = 0
    for batch in loader:
        batch = _batch_to_device(batch, device)

        targets = inner.graph_to_node_inputs(batch)
        posterior = inner.encode_target(targets)
        z = rearrange(posterior.mode(), "b d n -> b n d")
        dec = inner.decode_latent(z)
        preds_wf = inner.reconstruct_graph(
            dec, recon_curves=True, num_points=num_per_edge
        )

        graphs = unbatch_wireframe_graphs(batch)
        for pred_wf, graph in zip(preds_wf, graphs):
            gt_wf = _gt_wireframe_numpy(graph)

            ccd = curve_chamfer_distance(pred_wf, gt_wf, num_per_edge, device)
            vpe = vertex_position_error(pred_wf, gt_wf, device)
            ta = topology_accuracy(
                pred_wf, gt_wf, model.hparams.eval_match_thresh, device
            )

            ccd_l.append(ccd)
            vpe_l.append(vpe)
            ta_l.append(float(ta))
            ccd_s_l.append(distance_to_score(ccd, model.hparams.eval_ccd_tau))
            vpe_s_l.append(distance_to_score(vpe, model.hparams.eval_vpe_tau))
            n_pred_e.append(int(pred_wf["num_edges"]))
            n_gt_e.append(int(gt_wf["edge_index"].reshape(-1, 2).shape[0]))
            shape_ids.append(str(graph.get("shape_id", seen)))

            preds_all.append(pred_wf)
            gts_all.append(gt_wf)
            seen += 1
            if seen >= max_shapes:
                break
        if seen >= max_shapes:
            break

    return dict(
        preds=preds_all,
        gts=gts_all,
        ccd=np.asarray(ccd_l, dtype=np.float64),
        vpe=np.asarray(vpe_l, dtype=np.float64),
        ta=np.asarray(ta_l, dtype=np.float64),
        ccd_score=np.asarray(ccd_s_l, dtype=np.float64),
        vpe_score=np.asarray(vpe_s_l, dtype=np.float64),
        n_pred_e=np.asarray(n_pred_e, dtype=np.int64),
        n_gt_e=np.asarray(n_gt_e, dtype=np.int64),
        shape_ids=shape_ids,
    )


# ----------------------------------------------------------------------
def _wireframe_segments(wf: dict, num_per_edge: int) -> np.ndarray:
    """Return polylines ``(E, P, 3)`` for a wireframe (dense curves or straight)."""
    ep = wf.get("edge_points")
    if ep is not None and len(ep) > 0:
        return np.asarray(ep, dtype=np.float32).reshape(len(ep), -1, 3)
    verts = np.asarray(wf["vertices"], dtype=np.float32)
    edges = np.asarray(wf["edge_index"], dtype=np.int64).reshape(-1, 2)
    if edges.shape[0] == 0 or verts.shape[0] == 0:
        return np.zeros((0, num_per_edge, 3), dtype=np.float32)
    t = np.linspace(0.0, 1.0, num_per_edge)[None, :, None]
    a = verts[edges[:, 0]][:, None, :]
    b = verts[edges[:, 1]][:, None, :]
    return (a * (1.0 - t) + b * t).astype(np.float32)


def _plot_wireframe_cell(ax, gt: dict, pred: dict, num_per_edge: int, title: str):
    """3-D overlay of one GT wireframe (blue) vs reconstruction (red dashed)."""
    gt_seg = _wireframe_segments(gt, num_per_edge)
    pr_seg = _wireframe_segments(pred, num_per_edge)

    for k, c in enumerate(gt_seg):
        ax.plot(c[:, 0], c[:, 1], c[:, 2], "-", color="#1f77b4", lw=1.2,
                label="GT" if k == 0 else None)
    for k, c in enumerate(pr_seg):
        ax.plot(c[:, 0], c[:, 1], c[:, 2], "--", color="#d62728", lw=1.2,
                label="recon" if k == 0 else None)

    gt_v = np.asarray(gt["vertices"], dtype=np.float32).reshape(-1, 3)
    pr_v = np.asarray(pred["vertices"], dtype=np.float32).reshape(-1, 3)
    if gt_v.shape[0]:
        ax.scatter(gt_v[:, 0], gt_v[:, 1], gt_v[:, 2], color="#1f77b4", s=10,
                   depthshade=False)
    if pr_v.shape[0]:
        ax.scatter(pr_v[:, 0], pr_v[:, 1], pr_v[:, 2], color="#d62728", s=10,
                   marker="x", depthshade=False)

    pts = [p.reshape(-1, 3) for p in (gt_seg, pr_seg) if p.size]
    pts = np.concatenate(pts, axis=0) if pts else np.zeros((1, 3))
    c = pts.mean(0)
    r = max(float(np.abs(pts - c).max()), 1e-3)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=5)
    ax.view_init(elev=22, azim=-60)


def save_grid(res: dict, idx: np.ndarray, num_per_edge: int, out_path: str, suptitle: str):
    n = len(idx)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(cols * 3.2, rows * 3.2))
    for i, j in enumerate(idx):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        title = (f"#{int(j)} ccd={res['ccd'][j]:.3f}\n"
                 f"ta={res['ta'][j]:.2f} vpe={res['vpe'][j]:.3f}\n"
                 f"E:{res['n_pred_e'][j]}/{res['n_gt_e'][j]}")
        _plot_wireframe_cell(ax, res["gts"][j], res["preds"][j], num_per_edge, title)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[saved] {out_path}")


def save_hists(res: dict, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, name, color in (
        (axes[0], "ccd", "CCD (lower better)", "#4c72b0"),
        (axes[1], "vpe", "VPE (lower better)", "#55a868"),
        (axes[2], "ta", "TA (higher better)", "#c44e52"),
    ):
        vals = res[key]
        finite = vals[np.isfinite(vals)]
        ax.hist(finite, bins=40, color=color, alpha=0.85)
        if finite.size:
            ax.axvline(float(finite.mean()), color="k", ls="--",
                       label=f"mean={finite.mean():.4f}")
            ax.axvline(float(np.median(finite)), color="orange", ls="--",
                       label=f"median={np.median(finite):.4f}")
        ax.set_xlabel(name)
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.suptitle("Wireframe VAE reconstruction metrics on val", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[saved] {out_path}")


# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt = args.ckpt or _default_ckpt()
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(ckpt)), "wireframe_vae_eval"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[ckpt] {ckpt}")
    print(f"[out ] {out_dir}")

    # curve_vae_ckpt=None: the (frozen) curve VAE weights are already inside this
    # stage-2 checkpoint, so we don't need (and shouldn't depend on) the original
    # stage-1 path that may no longer exist.
    model = WireframeVAEModule.load_from_checkpoint(
        ckpt, map_location=args.device, curve_vae_ckpt=None
    )
    model.eval().to(args.device)

    loader = build_val_loader(args.data_config, args.num_workers, args.batch_size)
    res = run_eval(model, loader, args.device, args.max_shapes, args.num_per_edge)

    n = len(res["ccd"])
    if n == 0:
        raise SystemExit("No shapes evaluated; check the val split / data config.")

    ccd, vpe, ta = res["ccd"], res["vpe"], res["ta"]
    ccd_s, vpe_s = res["ccd_score"], res["vpe_score"]
    w_ccd = float(model.hparams.eval_w_ccd)
    w_ta = float(model.hparams.eval_w_ta)
    w_vpe = float(model.hparams.eval_w_vpe)
    score = w_ccd * ccd_s.mean() + w_ta * ta.mean() + w_vpe * vpe_s.mean()

    def _stat(a):
        fa = a[np.isfinite(a)]
        if fa.size == 0:
            return "nan"
        return (f"mean={fa.mean():.4f} median={np.median(fa):.4f} "
                f"p90={np.percentile(fa, 90):.4f} max={fa.max():.4f}")

    print("\n==== wireframe VAE reconstruction on val (encode GT -> decode) ====")
    print(f"  shapes evaluated : {n}")
    print(f"  CCD  : {_stat(ccd)}")
    print(f"  VPE  : {_stat(vpe)}")
    print(f"  TA   : {_stat(ta)}")
    print(f"  CCD score (exp -d/tau) mean: {ccd_s.mean():.4f}")
    print(f"  VPE score (exp -d/tau) mean: {vpe_s.mean():.4f}")
    print(f"  edges pred/gt mean : {res['n_pred_e'].mean():.1f} / {res['n_gt_e'].mean():.1f}")
    edge_acc = float((res["n_pred_e"] == res["n_gt_e"]).mean())
    print(f"  exact edge-count match : {edge_acc * 100:.1f}%")
    print(f"\n  weighted val/score (w_ccd={w_ccd}, w_ta={w_ta}, w_vpe={w_vpe})"
          f" = {score:.4f}\n")

    # Rank shapes best -> worst by the per-shape weighted score.
    per_score = w_ccd * ccd_s + w_ta * ta + w_vpe * vpe_s
    order = np.argsort(-per_score)  # high score (best) first
    k = min(args.num_show, n)

    # representative spread: evenly spaced percentiles best -> worst
    spread = order[np.linspace(0, len(order) - 1, k).astype(int)]
    save_grid(res, spread, args.num_per_edge,
              os.path.join(out_dir, "recon_spread.png"),
              "Wireframe VAE: GT (blue) vs recon (red) -- best->worst spread")

    # worst cases (lowest per-shape score)
    worst = order[-k:][::-1]
    save_grid(res, worst, args.num_per_edge,
              os.path.join(out_dir, "recon_worst.png"),
              "Wireframe VAE: GT (blue) vs recon (red) -- worst cases")

    save_hists(res, os.path.join(out_dir, "recon_metrics_hist.png"))


if __name__ == "__main__":
    main()
