#!/usr/bin/env python3
"""Sweep decode thresholds on the validation set to pick the best ones.

The WireframeAE val score is sensitive to the *post-processing* decode params
(``edge_thresh`` / ``tau_merge`` / ``topk_edges``) used by
:func:`src.recon.aggregate_wireframe`, but those are cheap to vary once the
network has run. This script therefore:

  1. loads a checkpoint (encoder + RVQ + edge-set decoder), exactly like
     ``export_submission.py`` (round-trip through the *submitted* indices);
  2. runs the (expensive) forward pass **once** over the val split and caches
     each shape's per-edge existence probabilities + ordered curve points and
     its GT wireframe;
  3. sweeps a grid of ``(edge_thresh, tau_merge, max_edges)`` over the cache,
     scoring every combination with the *same* :class:`WireframeScore` metric
     used during training, and reports the best.

The score definition (CCD / TA / VPE -> weighted final, higher is better) and
its weights / ``match_thresh`` are read from the checkpoint hyper-parameters so
the numbers line up with ``val/score`` from training.

Usage (from the project root)::

    python scripts/find_best_thresh.py \
        --ckpt logs/pc2wireframe/kh95mmuk/checkpoints/last.ckpt

    # custom grids + save the full table
    python scripts/find_best_thresh.py --ckpt <ckpt> \
        --edge-thresh-grid 0.05:0.9:0.05 \
        --tau-merge-grid 0.0,0.01,0.015,0.02 \
        --max-edges-grid 0,128,256 \
        --out logs/thresh_sweep.csv

Once you have the winning ``(edge_thresh, tau_merge, max_edges)`` feed them
straight to ``export_submission.py`` via ``--edge-thresh`` / ``--tau-merge`` /
``--max-edges``.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import torch

# Make the project importable as a package (``src.*``) and reuse the
# checkpoint-loading helpers from the export script.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_PROJECT_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export_submission import _resolve_ckpt, load_model  # noqa: E402

from src.data.dataset import WireframeGraphDataset, collate_ae_batch  # noqa: E402
from src.metrics import WireframeScore  # noqa: E402
from src.recon import aggregate_wireframe  # noqa: E402


# ----------------------------------------------------------------------
# grid parsing
# ----------------------------------------------------------------------
def _parse_float_grid(spec: str) -> list[float]:
    """Parse ``a,b,c`` or ``start:stop:step`` (inclusive) into a float list."""
    spec = spec.strip()
    if ":" in spec:
        lo, hi, step = (float(x) for x in spec.split(":"))
        if step <= 0:
            raise ValueError(f"step must be > 0 in {spec!r}")
        n = int(round((hi - lo) / step)) + 1
        vals = [round(lo + i * step, 10) for i in range(max(1, n))]
        return [v for v in vals if v <= hi + 1e-9]
    return [float(x) for x in spec.split(",") if x.strip() != ""]


def _parse_int_grid(spec: str) -> list[int]:
    return [int(float(x)) for x in spec.strip().split(",") if x.strip() != ""]


# ----------------------------------------------------------------------
# val dataset (mirror scripts/vis_ae_val._build_dataset for split=val)
# ----------------------------------------------------------------------
def build_val_dataset(args: argparse.Namespace) -> WireframeGraphDataset:
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


# ----------------------------------------------------------------------
# forward pass once -> cache per-shape (exist_prob, edge_points, gt)
# ----------------------------------------------------------------------
@torch.no_grad()
def run_forward_cache(
    encoder, quantizer, decoder, dataset, args
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, np.ndarray]]]:
    """Run the (expensive) network once; return cached decoder outputs + GT.

    Returns three parallel lists (one entry per val shape):
        exist_probs : (Q,)        float32   sigmoid(edge_exist_logit)
        edge_points : (Q, P, 3)   float32   ordered curve samples
        gts         : dict        {vertices, edge_index, edge_points} numpy
    """
    n_total = len(dataset)
    indices = list(range(n_total))
    if args.limit and args.limit > 0:
        indices = indices[: args.limit]

    exist_probs: list[np.ndarray] = []
    edge_pts: list[np.ndarray] = []
    gts: list[dict[str, np.ndarray]] = []

    bs = max(1, args.batch_size)
    pbar = None
    if not args.no_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(indices), unit="shape", desc="forward")
        except Exception:  # noqa: BLE001
            pbar = None

    for start in range(0, len(indices), bs):
        chunk = indices[start:start + bs]
        samples = [dataset[i] for i in chunk]
        batch = collate_ae_batch(samples)
        pc = batch["point_cloud"].to(args.device)
        offset = batch["pc_offset"].to(args.device)

        z_list = encoder(pc, offset)
        # Round-trip through the submitted indices (indices -> z_q -> decoder)
        # so the sweep matches what the exported submission will score.
        flat_idx = quantizer(z_list)["indices"]
        z_q = quantizer.decode_indices(flat_idx)
        out = decoder(z_q)

        ep = torch.sigmoid(out["edge_exist_logit"]).detach().cpu().numpy()
        cp = out["edge_points"].detach().cpu().numpy().astype(np.float32)

        for j, g in enumerate(batch["gt_wireframes"]):
            exist_probs.append(ep[j].astype(np.float32))
            edge_pts.append(cp[j])
            gts.append({
                "vertices": g["vertices"].detach().cpu().numpy(),
                "edge_index": g["edge_index"].detach().cpu().numpy(),
                "edge_points": g["edge_points"].detach().cpu().numpy(),
            })
        if pbar is not None:
            pbar.update(len(chunk))

    if pbar is not None:
        pbar.close()
    return exist_probs, edge_pts, gts


# ----------------------------------------------------------------------
# score one (edge_thresh, tau_merge, max_edges) over the cache
# ----------------------------------------------------------------------
def score_combo(
    exist_probs: list[np.ndarray],
    edge_pts: list[np.ndarray],
    gts: list[dict[str, np.ndarray]],
    *,
    edge_thresh: float,
    tau_merge: float,
    max_edges: int,
    min_edges: int,
    num_per_edge: int,
    metric_kwargs: dict[str, Any],
    device: str,
) -> dict[str, float]:
    metric = WireframeScore(num_per_edge=num_per_edge, **metric_kwargs).to(device)
    metric.reset()
    n_nonempty = 0
    pred_edges = 0
    for prob, cp, gt in zip(exist_probs, edge_pts, gts):
        pred = aggregate_wireframe(
            cp, prob,
            edge_threshold=edge_thresh, tau_merge=tau_merge,
            topk_edges=max_edges, min_edges=min_edges,
            num_per_edge=num_per_edge,
        )
        ne = int(np.asarray(pred["edge_index"]).reshape(-1, 2).shape[0])
        n_nonempty += 1 if ne > 0 else 0
        pred_edges += ne
        metric.update([pred], [gt])
    res = metric.compute()
    n = max(1, len(gts))
    return {
        "edge_thresh": float(edge_thresh),
        "tau_merge": float(tau_merge),
        "max_edges": int(max_edges),
        "score": float(res["score"]),
        "ccd": float(res["ccd"]),
        "ta": float(res["ta"]),
        "vpe": float(res["vpe"]),
        "ccd_score": float(res["ccd_score"]),
        "vpe_score": float(res["vpe_score"]),
        "nonempty_frac": n_nonempty / n,
        "mean_pred_edges": pred_edges / n,
    }


# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--ckpt",
        default="logs/pc2wireframe/checkpoints",
        help="WireframeAE ckpt file or dir (best val_score auto-picked)")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    # ---- val data (mirrors scripts/vis_ae_val.py) ----
    p.add_argument("--data-root", default="data")
    p.add_argument("--edge-subdir", default="train/sample_edge")
    p.add_argument("--pc-subdir", default="train/sample_pointcloud")
    p.add_argument("--split-path", default="data/split.json")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--num-edge-points", type=int, default=32)
    p.add_argument("--max-pc-points", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=0,
                   help="only score the first N val shapes (0 = all)")
    p.add_argument("--no-progress", action="store_true")
    # ---- sweep grids ----
    p.add_argument("--edge-thresh-grid", default="0.05:0.9:0.05",
                   help="'a,b,c' or 'start:stop:step' (inclusive)")
    p.add_argument("--tau-merge-grid", default="0.015",
                   help="'a,b,c' or 'start:stop:step' (inclusive)")
    p.add_argument("--max-edges-grid", default="0",
                   help="comma list of top-k caps (0 = no cap)")
    p.add_argument("--min-edges", type=int, default=1,
                   help="floor on edges per shape (fall back to top-k if fewer "
                        "clear the threshold; never empty). Default mirrors the "
                        "training / export default.")
    # ---- eval metric overrides (default: read from ckpt hparams) ----
    p.add_argument("--w-ccd", type=float, default=None)
    p.add_argument("--w-ta", type=float, default=None)
    p.add_argument("--w-vpe", type=float, default=None)
    p.add_argument("--match-thresh", type=float, default=None)
    p.add_argument("--out", default=None,
                   help="optional path to write the full sweep table (.csv)")
    p.add_argument("--top", type=int, default=15,
                   help="how many top rows to print")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt = _resolve_ckpt(args.ckpt, "val_score", "max")
    print(f"ckpt: {ckpt}")
    encoder, quantizer, decoder, hp = load_model(ckpt, args.device)

    # Eval metric params: prefer CLI overrides, else the ckpt's training values,
    # else the WireframeScore defaults -- so the sweep score matches val/score.
    metric_kwargs = {
        "w_ccd": args.w_ccd if args.w_ccd is not None
        else float(hp.get("eval_w_ccd", 0.3)),
        "w_ta": args.w_ta if args.w_ta is not None
        else float(hp.get("eval_w_ta", 0.4)),
        "w_vpe": args.w_vpe if args.w_vpe is not None
        else float(hp.get("eval_w_vpe", 0.3)),
        "match_thresh": args.match_thresh if args.match_thresh is not None
        else float(hp.get("eval_match_thresh", 0.1)),
    }
    num_per_edge = int(hp.get("num_per_edge", args.num_edge_points))
    baked = (float(hp.get("edge_thresh", 0.5)),
             float(hp.get("tau_merge", 0.015)),
             int(hp.get("topk_edges", 0)))
    print(f"metric: w_ccd={metric_kwargs['w_ccd']} w_ta={metric_kwargs['w_ta']} "
          f"w_vpe={metric_kwargs['w_vpe']} "
          f"match_thresh={metric_kwargs['match_thresh']} "
          f"num_per_edge={num_per_edge}")
    print(f"ckpt baked-in decode: edge_thresh={baked[0]} tau_merge={baked[1]} "
          f"topk_edges={baked[2]}")

    et_grid = _parse_float_grid(args.edge_thresh_grid)
    tau_grid = _parse_float_grid(args.tau_merge_grid)
    me_grid = _parse_int_grid(args.max_edges_grid)
    n_combos = len(et_grid) * len(tau_grid) * len(me_grid)
    print(f"grid: {len(et_grid)} edge_thresh x {len(tau_grid)} tau_merge x "
          f"{len(me_grid)} max_edges = {n_combos} combos")

    dataset = build_val_dataset(args)
    print(f"val shapes: {len(dataset)}"
          f"{f' (limited to {args.limit})' if args.limit else ''}")

    exist_probs, edge_pts, gts = run_forward_cache(
        encoder, quantizer, decoder, dataset, args)
    print(f"cached {len(gts)} shapes; sweeping {n_combos} combos...")

    results: list[dict[str, Any]] = []
    sweep_bar = None
    if not args.no_progress:
        try:
            from tqdm import tqdm
            sweep_bar = tqdm(total=n_combos, unit="combo", desc="sweep")
        except Exception:  # noqa: BLE001
            sweep_bar = None
    for et in et_grid:
        for tau in tau_grid:
            for me in me_grid:
                results.append(score_combo(
                    exist_probs, edge_pts, gts,
                    edge_thresh=et, tau_merge=tau, max_edges=me,
                    min_edges=args.min_edges,
                    num_per_edge=num_per_edge, metric_kwargs=metric_kwargs,
                    device=args.device))
                if sweep_bar is not None:
                    sweep_bar.update(1)
    if sweep_bar is not None:
        sweep_bar.close()

    results.sort(key=lambda r: r["score"], reverse=True)

    # ---- print the leaderboard ----
    hdr = (f"{'et':>5} {'tau':>6} {'maxE':>5} | {'score':>7} {'TA':>6} "
           f"{'CCD':>6} {'VPE':>6} | {'!empty':>6} {'predE':>6}")
    print("\n" + "=" * len(hdr))
    print("THRESHOLD SWEEP (sorted by score, higher is better)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results[: max(1, args.top)]:
        print(f"{r['edge_thresh']:>5.2f} {r['tau_merge']:>6.3f} "
              f"{r['max_edges']:>5d} | {r['score']:>7.4f} {r['ta']:>6.3f} "
              f"{r['ccd']:>6.3f} {r['vpe']:>6.3f} | "
              f"{r['nonempty_frac']:>6.2f} {r['mean_pred_edges']:>6.1f}")
    print("-" * len(hdr))
    best = results[0]
    print(f"BEST: edge_thresh={best['edge_thresh']:.3f} "
          f"tau_merge={best['tau_merge']:.4f} max_edges={best['max_edges']} "
          f"-> score={best['score']:.4f}")
    print("Reproduce in export_submission.py:")
    print(f"  --edge-thresh {best['edge_thresh']:g} "
          f"--tau-merge {best['tau_merge']:g} "
          f"--max-edges {best['max_edges']} "
          f"--min-edges {args.min_edges}")

    # ---- optional CSV dump ----
    if args.out:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        cols = ["edge_thresh", "tau_merge", "max_edges", "score", "ta", "ccd",
                "vpe", "ccd_score", "vpe_score", "nonempty_frac",
                "mean_pred_edges"]
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in results:
                w.writerow({c: r[c] for c in cols})
        print(f"\nfull table ({len(results)} rows) -> {args.out}")


if __name__ == "__main__":
    main()
