#!/usr/bin/env python3
"""Visualize the single-stage WireframeAE end to end.

For a handful of shapes from the requested ``--split`` it:

  1. loads the input surface point cloud (and, for train/val, the GT wireframe)
     via the real dataset code path -- exactly what training validated on;
  2. runs the encoder (point cloud -> latent z (16x256)) and the WireframeAE
     decoder (latent -> vertex queries + pairwise edges) and decodes a
     wireframe with the thresholds stored in the checkpoint;
  3. renders a per-shape comparison row:
     ``input point cloud | predicted wireframe | GT wireframe``
     (the GT column is dropped for the ``test`` split, which has no edges).

``test`` ships only point clouds, so it is loaded via the point-cloud-only
dataset and GT scoring is skipped. For ``test`` the precomputed **baseline
submission** wireframe (``--baseline-dir``) is drawn as an extra column.

Usage (from the project root)::

    python scripts/vis_ae_val.py --ckpt logs/pc2wireframe/<run>/checkpoints \
        --split val --num 6

    python scripts/vis_ae_val.py --split test --num 8 \
        --out logs/ae_test.png
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

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.dataset import (  # noqa: E402
    WireframeGraphDataset,
    PointCloudDataset,
    collate_ae_batch,
)
from src.models.utonia_encoder import UtoniaEncoder  # noqa: E402
from src.models.wireframe_ae import WireframeAE  # noqa: E402
from src.recon import decode_wireframe  # noqa: E402


# ----------------------------------------------------------------------
# Checkpoint loading
# ----------------------------------------------------------------------
def _resolve_ckpt(path: str, metric: str = "val_score", best: str = "max") -> str:
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        cands = glob.glob(os.path.join(path, "*.ckpt"))
        if not cands:
            raise FileNotFoundError(f"No .ckpt files under {path!r}")
        scored: list[tuple[float, str]] = []
        pat = re.compile(rf"{re.escape(metric)}=([0-9]*\.?[0-9]+)")
        for c in cands:
            m = pat.search(os.path.basename(c))
            if m:
                scored.append((float(m.group(1)), c))
        if scored:
            scored.sort()
            return scored[0][1] if best == "min" else scored[-1][1]
        last = os.path.join(path, "last.ckpt")
        return last if os.path.isfile(last) else cands[0]
    raise FileNotFoundError(f"Checkpoint path not found: {path!r}")


def _strip_prefix(state: dict[str, torch.Tensor], prefix: str
                  ) -> dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def load_model(ckpt_path: str, device: str
               ) -> tuple[UtoniaEncoder, WireframeAE, dict[str, Any]]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters", {}) or {}
    encoder = UtoniaEncoder(**(hp.get("pc_encoder") or {}))
    decoder = WireframeAE(**(hp.get("decoder") or {}))
    state = ck["state_dict"] if "state_dict" in ck else ck
    encoder.load_state_dict(_strip_prefix(state, "encoder."), strict=False)
    decoder.load_state_dict(_strip_prefix(state, "decoder."), strict=False)
    encoder.eval().to(device)
    decoder.eval().to(device)
    return encoder, decoder, hp


# ----------------------------------------------------------------------
# Decode (mirrors module.decode / export_submission.decode_batch)
# ----------------------------------------------------------------------
@torch.no_grad()
def decode_batch(
    decoder, out, hp,
    *,
    vertex_thresh: float | None = None,
    edge_thresh: float | None = None,
    max_edges: int = 0,
) -> list[dict[str, np.ndarray]]:
    """Decode a batch of decoder outputs into wireframes (numpy).

    ``vertex_thresh`` / ``edge_thresh`` override the values baked into the
    checkpoint hyper-parameters when provided. ``max_edges`` caps the number of
    edges per shape: among the pairs that clear ``edge_thresh`` only the top-k
    by existence probability are kept (0 = no cap). Mirrors
    ``export_submission.decode_batch`` so visualizations match the submission.
    """
    vertex_logit = out["vertex_logit"]
    vertex_xyz = out["vertex_xyz"]
    hidden = out["hidden"]
    global_vec = out["global"]
    device = vertex_logit.device
    b = vertex_logit.shape[0]
    vt = float(vertex_thresh if vertex_thresh is not None
               else hp.get("vertex_thresh", 0.5))
    et = float(edge_thresh if edge_thresh is not None
               else hp.get("edge_thresh", 0.5))
    cap = int(hp.get("max_decode_vertices", 512))
    npe = int(hp.get("num_per_edge", 32))
    me = max(0, int(max_edges))

    wfs: list[dict[str, np.ndarray]] = []
    for i in range(b):
        prob = torch.sigmoid(vertex_logit[i])
        alive = torch.nonzero(prob >= vt, as_tuple=False).reshape(-1)
        if alive.numel() > cap > 0:
            alive = alive[torch.topk(prob[alive], cap).indices]
        if alive.numel() < 2:
            wfs.append({
                "vertices": vertex_xyz[i][alive].cpu().numpy()
                .astype(np.float32).reshape(-1, 3),
                "edge_index": np.zeros((0, 2), dtype=np.int64),
                "edge_points": np.zeros((0, npe, 3), dtype=np.float32),
            })
            continue
        verts = vertex_xyz[i][alive]
        va = alive.shape[0]
        iu, ju = torch.triu_indices(va, va, offset=1, device=device)
        h = hidden[i][alive]
        ehead = decoder.edge_logits(
            h[iu], h[ju], global_vec[i][None, :].expand(iu.shape[0], -1))

        # Threshold + top-k edge selection on-device (mirrors the official
        # baseline): keep edges above ``et`` and, if still too many, the
        # top-``me`` by probability.
        edge_prob = torch.sigmoid(ehead["exist"])      # (M,)
        keep = edge_prob >= et
        if me > 0 and int(keep.sum().item()) > me:
            surv = torch.nonzero(keep, as_tuple=False).reshape(-1)
            top = torch.topk(edge_prob[surv], me, largest=True).indices
            keep = torch.zeros_like(keep)
            keep[surv[top]] = True
        sel = torch.nonzero(keep, as_tuple=False).reshape(-1)

        iu_s, ju_s = iu[sel], ju[sel]
        fields = {
            "vertices": verts.cpu().numpy().astype(np.float32),
            "pair_index": torch.stack([iu_s, ju_s], dim=1).cpu().numpy(),
            "edge_prob": edge_prob[sel].cpu().numpy(),
            "edge_type": ehead["type"][sel].argmax(dim=-1).cpu().numpy(),
            "q1": ehead["params"][sel, 0].cpu().numpy(),
            "q2": ehead["params"][sel, 1].cpu().numpy(),
        }
        wfs.append(decode_wireframe(fields, edge_thresh=et, num_per_edge=npe))
    return wfs


# ----------------------------------------------------------------------
# Lightweight numpy/scipy metrics (CPU port; same defs as the training metric)
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


def _topology_accuracy(pred, gt, match_thresh: float) -> float:
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


def score_reconstruction(pred, gt, *, num_per_edge, ccd_tau, vpe_tau,
                         match_thresh, w_ccd, w_ta, w_vpe) -> dict[str, float]:
    ccd = _chamfer(
        _flatten_curves(pred, num_per_edge), _flatten_curves(gt, num_per_edge))
    vpe = _chamfer(pred.get("vertices"), gt.get("vertices"))
    ta = _topology_accuracy(pred, gt, match_thresh)
    ccd_s = _dist_to_score(ccd, ccd_tau)
    vpe_s = _dist_to_score(vpe, vpe_tau)
    score = w_ccd * ccd_s + w_ta * ta + w_vpe * vpe_s
    return {"ccd": ccd, "vpe": vpe, "ta": ta,
            "ccd_score": ccd_s, "vpe_score": vpe_s, "score": score}


# ----------------------------------------------------------------------
# Dataset + sample selection
# ----------------------------------------------------------------------
def _build_dataset(args: argparse.Namespace) -> Any:
    if args.split == "test":
        return PointCloudDataset(
            pointcloud_dir=args.test_pc_dir, max_pc_points=args.max_pc_points)
    return WireframeGraphDataset(
        split=args.split,
        split_path=args.split_path,
        edge_dir=os.path.join(args.data_root, args.edge_subdir),
        pointcloud_dirs=[os.path.join(args.data_root, args.pc_subdir)],
        auto_build_split=True,
        num_edge_points=args.num_edge_points,
        max_pc_points=args.max_pc_points,
        min_edges=1,
    )


def _select_indices(dataset, num, pick, seed, pool=None) -> list[int]:
    cands = list(range(len(dataset))) if pool is None else list(pool)
    num = min(num, len(cands))
    if pick == "first":
        return cands[:num]
    rng = random.Random(seed)
    return rng.sample(cands, num)


def _baseline_stems(baseline_dir: str) -> set[str]:
    if not os.path.isdir(baseline_dir):
        return set()
    return {os.path.splitext(f)[0]
            for f in os.listdir(baseline_dir) if f.endswith(".npz")}


def _load_baseline_wf(stem, baseline_dir):
    path = os.path.join(baseline_dir, f"{stem}.npz")
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=True) as z:
            return {
                "vertices": np.asarray(z["vertices"], dtype=np.float32).reshape(-1, 3),
                "edge_index": np.asarray(z["edge_index"], dtype=np.int64).reshape(-1, 2),
                "edge_points": np.asarray(z["edge_points"], dtype=np.float32),
            }
    except Exception as exc:
        print(f"[warn] failed to load baseline for {stem!r}: {exc}")
        return None


def _resolve_file_indices(dataset, files) -> list[int]:
    by_stem = {os.path.splitext(os.path.basename(f))[0]: i
               for i, f in enumerate(dataset.files)}
    out: list[int] = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem in by_stem:
            out.append(by_stem[stem])
        else:
            print(f"[warn] {stem!r} not in split; skipped")
    return out


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------
_PC_C = "#9aa0a6"
_AXIS = {"x": 0, "y": 1, "z": 2}
_CUSTOM_CMAPS = {
    "soft": ["#6b8fd6", "#56b6c2", "#86d6a8"],
    "warmsoft": ["#e8c98a", "#e09b7d", "#cf8aa6"],
}


def _get_cmap(name: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if name in _CUSTOM_CMAPS:
        return LinearSegmentedColormap.from_list(name, _CUSTOM_CMAPS[name])
    return plt.get_cmap(name)


def _gradient_rgb(t, args) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    tt = args.color_lo + t * (args.color_hi - args.color_lo)
    rgb = np.asarray(_get_cmap(args.cmap)(tt))[..., :3]
    lum = np.tensordot(rgb, np.array([0.2126, 0.7152, 0.0722]), axes=([-1], [0]))
    return lum[..., None] + args.sat * (rgb - lum[..., None])


def _equal_box(ax, pts_list) -> None:
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
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)


def _draw_wireframe(ax, wf, args) -> None:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    axis = _AXIS[args.color_axis]
    ep = np.asarray(wf.get("edge_points"))
    if ep.size:
        polys = [ep[i].reshape(-1, 3) for i in range(ep.shape[0])]
        polys = [p for p in polys if p.shape[0] >= 2]
        if polys:
            cat = np.concatenate(polys, axis=0)
            vmin, vmax = float(cat[:, axis].min()), float(cat[:, axis].max())
            seg_list, val_list = [], []
            for p in polys:
                seg_list.append(np.stack([p[:-1], p[1:]], axis=1))
                mid = (p[:-1] + p[1:]) * 0.5
                val_list.append(mid[:, axis])
            segs = np.concatenate(seg_list, 0)
            vals = np.concatenate(val_list, 0)
            t = (vals - vmin) / (vmax - vmin + 1e-12)
            lc = Line3DCollection(
                segs, colors=_gradient_rgb(t, args), linewidths=args.linewidth)
            ax.add_collection3d(lc)
    if args.show_vertices:
        v = np.asarray(wf.get("vertices")).reshape(-1, 3)
        if v.size:
            ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=6, color="#555555",
                       depthshade=False)


def _visualize(rows, out_path, args, has_gt, has_baseline) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    ncol = 2 + (1 if has_gt else 0) + (1 if has_baseline else 0)
    panel = args.panel
    fig = plt.figure(figsize=(ncol * panel, n * panel))
    fig.patch.set_facecolor("white")

    for r, row in enumerate(rows):
        pc = row["point_cloud"]
        pred = row["pred"]
        gt = row.get("gt")
        base = row.get("baseline")
        pipe_bounds = [pc, pred.get("edge_points"), pred.get("vertices")]
        if gt is not None:
            pipe_bounds += [gt.get("edge_points"), gt.get("vertices")]

        off = r * ncol
        col = 0

        col += 1
        ax = fig.add_subplot(n, ncol, off + col, projection="3d")
        if pc.size:
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=args.point_size,
                       color=_PC_C, alpha=args.alpha, depthshade=False,
                       linewidths=0)
        _equal_box(ax, pipe_bounds)
        _style_ax(ax, args)
        ax.set_title(f"{row['name'][:26]}\ninput point cloud: {len(pc)} pts",
                     fontsize=8)

        col += 1
        ax = fig.add_subplot(n, ncol, off + col, projection="3d")
        _draw_wireframe(ax, pred, args)
        _equal_box(ax, pipe_bounds)
        _style_ax(ax, args)
        sub = (f"{len(pred['vertices'])} V / {len(pred['edge_index'])} E")
        m = row.get("metrics")
        if m is not None:
            sub += (f"\nscore={m['score']:.3f}  TA={m['ta']:.3f}  "
                    f"CCD={m['ccd']:.3f}  VPE={m['vpe']:.3f}")
        ax.set_title(f"WireframeAE (ours)\n{sub}", fontsize=8)

        if has_gt and gt is not None:
            col += 1
            ax = fig.add_subplot(n, ncol, off + col, projection="3d")
            _draw_wireframe(ax, gt, args)
            _equal_box(ax, pipe_bounds)
            _style_ax(ax, args)
            ax.set_title(
                f"GT wireframe\n{len(gt['vertices'])} V / "
                f"{len(gt['edge_index'])} E", fontsize=8)

        if has_baseline:
            col += 1
            ax = fig.add_subplot(n, ncol, off + col, projection="3d")
            if base is not None:
                _draw_wireframe(ax, base, args)
                _equal_box(ax, [base.get("edge_points"), base.get("vertices")])
                ax.set_title(
                    f"baseline submission\n{len(base['vertices'])} V / "
                    f"{len(base['edge_index'])} E", fontsize=8)
            else:
                ax.set_title("baseline submission\n(missing)", fontsize=8)
            _style_ax(ax, args)

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
    p.add_argument("--ckpt", default="logs/pc2wireframe/checkpoints",
                   help="WireframeAE ckpt file or dir (best val_score auto-picked)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-root", default="data")
    p.add_argument("--edge-subdir", default="train/sample_edge")
    p.add_argument("--pc-subdir", default="train/sample_pointcloud")
    p.add_argument("--split-path", default="data/split.json")
    p.add_argument("--test-pc-dir", default="data/test/sample_pointcloud")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument(
        "--baseline-dir",
        default="pc2wireframe_baseline/test_submission/submission/sample_edge",
        help="dir of baseline submission wireframe npz (test split only)")
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--num", type=int, default=6)
    p.add_argument("--pick", default="random", choices=["random", "first"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--files", nargs="*", default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-edge-points", type=int, default=32)
    p.add_argument("--max-pc-points", type=int, default=0)
    # decode post-processing (override the thresholds baked into the ckpt)
    p.add_argument("--vertex-thresh", type=float, default=0.5,
                   help="keep a vertex iff sigmoid(alive) >= this "
                        "(default: use ckpt value; higher = fewer vertices)")
    p.add_argument("--edge-thresh", type=float, default=0.5,
                   help="keep an edge iff sigmoid(exist) >= this "
                        "(default: use ckpt value)")
    p.add_argument("--max-edges", type=int, default=1024,
                   help="hard cap on edges per shape: among edges passing "
                        "--edge-thresh, keep the top-k by probability "
                        "(0 = no cap). Mirrors the official top-k strategy.")
    p.add_argument("--ccd-tau", type=float, default=0.1)
    p.add_argument("--vpe-tau", type=float, default=0.1)
    p.add_argument("--match-thresh", type=float, default=0.1)
    p.add_argument("--w-ccd", type=float, default=0.3)
    p.add_argument("--w-ta", type=float, default=0.4)
    p.add_argument("--w-vpe", type=float, default=0.3)
    p.add_argument("--cmap", default="warmsoft")
    p.add_argument("--color-axis", default="z", choices=["x", "y", "z"])
    p.add_argument("--sat", type=float, default=0.9)
    p.add_argument("--color-lo", type=float, default=0.0)
    p.add_argument("--color-hi", type=float, default=1.0)
    p.add_argument("--linewidth", type=float, default=1.2)
    p.add_argument("--show-vertices", action="store_true")
    p.add_argument("--azimuth", type=float, default=-60.0)
    p.add_argument("--elevation", type=float, default=18.0)
    p.add_argument("--point-size", type=float, default=1.6)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--panel", type=float, default=4.5)
    p.add_argument("--zoom", type=float, default=1.2)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--out", default=None)
    p.add_argument("--no-viz", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    has_gt = args.split != "test"
    out_path = args.out or f"logs/ae_{args.split}.png"

    ckpt = _resolve_ckpt(args.ckpt, "val_score", "max")
    print(f"Loading checkpoint: {ckpt}")
    encoder, decoder, hp = load_model(ckpt, args.device)
    vt = args.vertex_thresh if args.vertex_thresh is not None \
        else hp.get("vertex_thresh", 0.5)
    et = args.edge_thresh if args.edge_thresh is not None \
        else hp.get("edge_thresh", 0.5)
    print(f"decode: vertex_thresh={vt} edge_thresh={et} "
          f"max_edges={args.max_edges or 'inf'}")

    has_baseline = (args.split == "test" and not args.no_baseline)
    baseline_stems: set[str] = set()
    if has_baseline:
        baseline_stems = _baseline_stems(args.baseline_dir)
        if not baseline_stems:
            print(f"[warn] no baseline npz under {args.baseline_dir!r}; "
                  f"disabling baseline column")
            has_baseline = False

    dataset = _build_dataset(args)
    if args.files:
        indices = _resolve_file_indices(dataset, args.files)
    else:
        pool: list[int] | None = None
        if has_baseline:
            pool = [i for i, f in enumerate(dataset.files)
                    if os.path.splitext(os.path.basename(f))[0] in baseline_stems]
            print(f"{len(pool)}/{len(dataset)} test shapes have a baseline result.")
            if not pool:
                raise SystemExit("No test shapes overlap the baseline submission.")
        indices = _select_indices(dataset, args.num, args.pick, args.seed, pool)
    if not indices:
        raise SystemExit("No shapes selected.")

    header = f"{'shape':<42}"
    if has_gt:
        header += (f" {'prV':>4} {'prE':>4} | {'TA':>5} {'CCD':>6} "
                   f"{'VPE':>6} {'score':>6}")
    else:
        header += f" {'prV':>4} {'prE':>4}"
    print(header)
    print("-" * len(header))

    rows: list[dict[str, Any]] = []
    agg: dict[str, list[float]] = {k: [] for k in ("ta", "ccd", "vpe", "score")}

    bs = max(1, args.batch_size)
    for start in range(0, len(indices), bs):
        chunk = indices[start:start + bs]
        samples = [dataset[i] for i in chunk]
        batch = collate_ae_batch(samples)
        pc = batch["point_cloud"].to(args.device)
        offset = batch["pc_offset"].to(args.device)

        with torch.no_grad():
            z = encoder(pc, offset)
            out = decoder(z)
            preds = decode_batch(
                decoder, out, hp,
                vertex_thresh=args.vertex_thresh,
                edge_thresh=args.edge_thresh,
                max_edges=args.max_edges)

        for j, s in enumerate(samples):
            name = str(s["shape_id"])
            pred = preds[j]
            pc_np = s["point_cloud"].numpy()
            row: dict[str, Any] = {
                "name": name, "point_cloud": pc_np, "pred": pred,
            }
            if has_baseline:
                row["baseline"] = _load_baseline_wf(name, args.baseline_dir)
            if has_gt:
                gt = {
                    "vertices": s["vertices"].numpy(),
                    "edge_index": s["edge_index"].numpy(),
                    "edge_points": s["edge_points"].numpy(),
                }
                m = score_reconstruction(
                    pred, gt, num_per_edge=args.num_edge_points,
                    ccd_tau=args.ccd_tau, vpe_tau=args.vpe_tau,
                    match_thresh=args.match_thresh,
                    w_ccd=args.w_ccd, w_ta=args.w_ta, w_vpe=args.w_vpe)
                row["gt"] = gt
                row["metrics"] = m
                for k in agg:
                    agg[k].append(m[k])
                print(f"{name[:42]:<42} {len(pred['vertices']):>4} "
                      f"{len(pred['edge_index']):>4} | {m['ta']:>5.3f} "
                      f"{m['ccd']:>6.3f} {m['vpe']:>6.3f} {m['score']:>6.3f}")
            else:
                print(f"{name[:42]:<42} {len(pred['vertices']):>4} "
                      f"{len(pred['edge_index']):>4}")
            rows.append(row)

    print("-" * len(header))
    if has_gt:
        def _ms(vals: list[float]) -> str:
            arr = np.array(vals, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            return "n/a" if arr.size == 0 else f"{arr.mean():.3f}±{arr.std():.3f}"
        print(f"mean±std over {len(rows)} shapes:  TA={_ms(agg['ta'])}  "
              f"CCD={_ms(agg['ccd'])}  VPE={_ms(agg['vpe'])}  "
              f"score={_ms(agg['score'])}")

    if not args.no_viz:
        _visualize(rows, out_path, args, has_gt, has_baseline)
        print(f"\nSaved WireframeAE comparison -> {out_path}")


if __name__ == "__main__":
    main()
