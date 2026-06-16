"""Visualise **baseline** vs **Ours** on the *test* set (no GT, no metrics).

The competition test split ships only point clouds (no GT edges), so this
script just renders, per test shape:

    input point cloud   |   baseline wireframe   |   Ours wireframe

  * **point cloud** -- ``data/test/sample_pointcloud/<stem>.npz``.
  * **baseline**    -- the exported submission
                       ``pc2wireframe_baseline/.../submission/sample_edge/<stem>.npz``.
  * **Ours**        -- run the stage-2 ``PC2WireframeModule`` on the point cloud
                       and denormalise the prediction back to the data frame
                       (exactly ``predict_step``).

The render set is the test shapes that *also* have a baseline submission (so
both methods can be shown side by side); ``--num-viz`` shapes are sampled from
it at random.

Headless: uses the ``Agg`` backend and saves PNGs.

Example::

    python scripts/eval_compare.py \
        --ckpt logs/pc2wireframe/ujtuvalu/checkpoints/epoch=109-val_score=0.1477.ckpt \
        --curve-vae-ckpt logs/pc2wireframe/adpdhtsg/checkpoints/epoch=049-val_loss=0.0058.ckpt \
        --num-viz 12
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

root = pyrootutils.setup_root(
    __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
)

from src.data.dataset import (  # noqa: E402
    _apply_unit_cube,
    _resample_polyline,
    _sample_points,
    _unit_cube_transform,
)
from src.module import PC2WireframeModule  # noqa: E402


# ----------------------------------------------------------------------
def _best_ckpt(log_subdir: str, key: str, mode: str) -> str:
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
                    help="stage-2 (Ours) ckpt (default: best val_score under logs/.../ujtuvalu)")
    ap.add_argument("--curve-vae-ckpt", default=None,
                    help="stage-1 curve VAE ckpt (default: best val_loss under logs/.../adpdhtsg)")
    ap.add_argument("--submission-dir", default=str(
        root / "pc2wireframe_baseline/test_submission/submission/sample_edge"),
        help="baseline submission sample_edge dir")
    ap.add_argument("--pc-dir", default=str(root / "data/test/sample_pointcloud"),
                    help="test point-cloud dir")
    ap.add_argument("--out-dir", default=None,
                    help="default: <ckpt_dir>/../compare_test_viz")
    ap.add_argument("--num-viz", type=int, default=12,
                    help="number of randomly sampled test shapes to visualise")
    ap.add_argument("--pc-num-points", type=int, default=4096)
    ap.add_argument("--pc-points-show", type=int, default=2048)
    ap.add_argument("--vertex-threshold", type=float, default=None)
    ap.add_argument("--edge-threshold", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


# ----------------------------------------------------------------------
def _load_npz(path: str) -> dict:
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except ValueError:
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}


def _get(d: dict, keys: tuple[str, ...]) -> np.ndarray:
    for k in keys:
        if k in d:
            return np.asarray(d[k])
    raise KeyError(f"none of {keys!r} in npz")


def load_submission_wireframe(sub_path: str, num_per_edge: int) -> dict:
    """Read a baseline submission npz into a wireframe dict (data frame)."""
    d = _load_npz(sub_path)
    verts = np.asarray(d["vertices"], np.float32).reshape(-1, 3)
    eidx = np.asarray(d["edge_index"], np.int64).reshape(-1, 2)
    ep = np.asarray(d.get("edge_points"), np.float32) if "edge_points" in d else None
    if ep is not None and ep.size and ep.shape[1] != num_per_edge:
        ep = np.stack([_resample_polyline(ep[e], num_per_edge)
                       for e in range(ep.shape[0])], 0)
    return {
        "vertices": verts,
        "edge_index": eidx,
        "edge_points": ep if ep is not None else np.zeros(
            (0, num_per_edge, 3), np.float32),
    }


# ----------------------------------------------------------------------
# Ours inference (normalize PC -> model -> denormalise to data frame)
# ----------------------------------------------------------------------
@torch.no_grad()
def predict_ours(net, stems: list[str], pcs: dict[str, np.ndarray], device: str,
                 batch_size: int, pc_num_points: int, num_per_edge: int,
                 v_thr: float, e_thr: float, seed: int) -> dict[str, dict]:
    """Run our model on each stem's point cloud, returning data-frame wireframes."""
    out: dict[str, dict] = {}
    buf_pc, buf_meta = [], []

    def flush():
        if not buf_pc:
            return
        pc = torch.from_numpy(np.stack(buf_pc, 0)).to(device)
        res = net(pc, sample=False)
        wfs = net.reconstruct(res["preds"], vertex_threshold=v_thr,
                              edge_threshold=e_thr, num_points=num_per_edge)
        for (stem, center, scale), wf in zip(buf_meta, wfs):
            # inverse unit-cube transform -> native data frame (== predict_step)
            if wf.get("vertices") is not None and wf["vertices"].size:
                wf["vertices"] = wf["vertices"] * scale + center
            if wf.get("edge_points") is not None and wf["edge_points"].size:
                wf["edge_points"] = wf["edge_points"] * scale + center
            out[stem] = wf
        buf_pc.clear()
        buf_meta.clear()

    rng = np.random.RandomState(seed)
    for stem in stems:
        # deterministic resample so the per-shape transform is reproducible.
        np.random.seed(int(rng.randint(0, 2**31 - 1)))
        pc = _sample_points(pcs[stem], pc_num_points)
        center, scale = _unit_cube_transform(pc)
        buf_pc.append(_apply_unit_cube(pc, center, scale))
        buf_meta.append((stem, center.astype(np.float32), float(scale)))
        if len(buf_pc) >= batch_size:
            flush()
    flush()
    return out


# ----------------------------------------------------------------------
# visualization
# ----------------------------------------------------------------------
def _edge_polylines(wf: dict, num_per_edge: int) -> list[np.ndarray]:
    ep = wf.get("edge_points")
    if ep is not None and len(ep) > 0:
        ep = np.asarray(ep, np.float32)
        return [ep[i] for i in range(ep.shape[0])]
    verts = np.asarray(wf.get("vertices"), np.float32).reshape(-1, 3)
    edges = np.asarray(wf.get("edge_index"), np.int64).reshape(-1, 2)
    t = np.linspace(0.0, 1.0, num_per_edge)[:, None]
    return [verts[a] * (1.0 - t) + verts[b] * t
            for a, b in edges if a < len(verts) and b < len(verts)]


def _set_box(ax, pts: np.ndarray):
    if pts.size == 0:
        return
    c = pts.mean(0)
    r = max(float(np.abs(pts - c).max()), 1e-3)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))


def _plot_pc(ax, pc, max_points, title):
    pc = np.asarray(pc, np.float32).reshape(-1, 3)
    if pc.shape[0] > max_points:
        idx = np.random.choice(pc.shape[0], size=max_points, replace=False)
        pc = pc[idx]
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1.6, c="#555555", alpha=0.7,
               depthshade=False, linewidths=0)
    _set_box(ax, pc)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=5)
    ax.view_init(elev=22, azim=-60)


def _plot_wf(ax, wf, color, title, num_per_edge):
    verts = np.asarray(wf.get("vertices"), np.float32).reshape(-1, 3)
    lines = _edge_polylines(wf, num_per_edge)
    for ln in lines:
        ax.plot(ln[:, 0], ln[:, 1], ln[:, 2], "-", color=color, lw=1.2)
    if verts.size:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], color="k", s=3,
                   depthshade=False, linewidths=0)
    allpts = [p for p in ([verts] + lines) if p.size]
    pts = np.concatenate(allpts, 0) if allpts else np.zeros((0, 3), np.float32)
    _set_box(ax, pts)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=5)
    ax.view_init(elev=22, azim=-60)


def save_viz_grid(stems, pcs, baselines, ours, out_path, num_per_edge, pc_points_show):
    n = len(stems)
    fig = plt.figure(figsize=(3 * 3.2, n * 3.2))
    for r, stem in enumerate(stems):
        ax0 = fig.add_subplot(n, 3, r * 3 + 1, projection="3d")
        _plot_pc(ax0, pcs[stem], pc_points_show, f"PC  {stem[:20]}")
        base = baselines[stem]
        ax1 = fig.add_subplot(n, 3, r * 3 + 2, projection="3d")
        _plot_wf(ax1, base, "#ff7f0e",
                 f"Baseline  V={len(base['vertices'])} E={len(base['edge_index'])}",
                 num_per_edge)
        wf = ours[stem]
        ax2 = fig.add_subplot(n, 3, r * 3 + 3, projection="3d")
        _plot_wf(ax2, wf, "#d62728",
                 f"Ours  V={wf.get('num_vertices', len(wf['vertices']))} "
                 f"E={wf.get('num_edges', len(wf['edge_index']))}", num_per_edge)
    fig.suptitle("Test set:  input PC | Baseline (orange) | Ours (red)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=120)
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
        os.path.dirname(os.path.dirname(ckpt)), "compare_test_viz")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[ckpt]           {ckpt}")
    print(f"[curve_vae_ckpt] {curve_vae_ckpt}")
    print(f"[submission]     {args.submission_dir}")
    print(f"[pc-dir]         {args.pc_dir}")
    print(f"[out]            {out_dir}")

    # render set: test shapes that also have a baseline submission.
    sub_stems = {os.path.splitext(f)[0] for f in os.listdir(args.submission_dir)
                 if f.endswith(".npz")}
    pc_stems = {os.path.splitext(f)[0] for f in os.listdir(args.pc_dir)
                if f.endswith(".npz")}
    pool = sorted(sub_stems & pc_stems)
    if not pool:
        raise SystemExit("No test shapes have both a point cloud and a baseline "
                         "submission; check --pc-dir / --submission-dir.")
    print(f"[set] {len(pool)} test shapes with both a point cloud and a baseline result")

    n_viz = min(args.num_viz, len(pool))
    rng = np.random.RandomState(args.seed)
    stems = [pool[i] for i in sorted(rng.choice(len(pool), size=n_viz, replace=False))]
    print(f"[viz] sampling {n_viz} shapes: "
          + ", ".join(s[:16] for s in stems[:8]) + (" ..." if n_viz > 8 else ""))

    # --- load Ours model ---
    model = PC2WireframeModule.load_from_checkpoint(
        ckpt, map_location=args.device, curve_vae_ckpt=curve_vae_ckpt)
    model.eval().to(args.device)
    net = model.model
    num_per_edge = int(model.hparams.eval_num_per_edge)
    v_thr = (args.vertex_threshold if args.vertex_threshold is not None
             else float(model.hparams.vertex_threshold))
    e_thr = (args.edge_threshold if args.edge_threshold is not None
             else float(model.hparams.edge_threshold))
    print(f"[thresholds] vertex={v_thr} edge={e_thr}")

    # --- load point clouds + baseline, run Ours ---
    pcs = {s: np.asarray(_get(_load_npz(os.path.join(args.pc_dir, s + ".npz")),
                              ("surface_points", "points", "point_cloud", "pc")),
                         np.float32).reshape(-1, 3) for s in stems}
    baselines = {s: load_submission_wireframe(
        os.path.join(args.submission_dir, s + ".npz"), num_per_edge) for s in stems}
    print("[infer] running Ours ...")
    ours = predict_ours(net, stems, pcs, args.device, args.batch_size,
                        args.pc_num_points, num_per_edge, v_thr, e_thr, args.seed)

    save_viz_grid(stems, pcs, baselines, ours,
                  os.path.join(out_dir, "compare_test_samples.png"),
                  num_per_edge, args.pc_points_show)


if __name__ == "__main__":
    main()
