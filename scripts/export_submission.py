#!/usr/bin/env python3
"""Export a competition submission zip from the PC2Wireframe model.

Serialises every test shape into the layout the official evaluator expects::

    submission/
        latent_pack.npz                 # stems (N,)  + latents (N, K)
        sample_edge/<stem>.npz
            latent       : (K,) float32, K <= 4096   (flat 16x256 PC latent)
            vertices     : (V, 3) float32
            edge_index   : (E, 2) int32   (indexes into vertices)
            edge_points  : (E, 32, 3) float32
            num_vertices : () int32
            num_edges    : () int32

and finally zips ``submission/`` into ``submission.zip``.

Pipeline per shape::

    raw test point cloud
        --[PTv3 encoder + latent compressor -> Z (16, 256)]-->   # the submission
    latent Z
        --[EdgeSetDecoder -> per-edge (confidence, endpoints, curve latent)]-->
        --[frozen curve VAE decode + assemble_wireframe]-->
    wireframe {vertices, edge_index, edge_points}

The per-sample ``latent`` is the **flat 16 x 256 = 4096 float32** point-cloud
latent (no RVQ indices). The wireframe is decoded from the same latent the
evaluator stores, so the export is a faithful round-trip.

Coordinate frame
----------------
The model works directly in the **normalized** point-cloud frame (the dataset is
already unit-normalized), so vertices / curves are written as decoded.

Usage (from the project root)::

    # single GPU
    python scripts/export_submission.py \
        --ckpt logs/pc2wireframe/<run>/checkpoints \
        --test-pc-dir data/test/sample_pointcloud \
        --out-dir logs/submission

    # 8-GPU data-parallel (one worker process per GPU, then auto-merge + zip)
    python scripts/export_submission.py --spawn 8 --out-dir logs/submission

Resume an interrupted run with ``--resume`` (skips shapes that already have an
output npz and does NOT wipe the out-dir).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import zipfile
from typing import Any

import numpy as np
import torch

# Make the project importable as a package (``src.*``).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.dataset import PointCloudDataset, collate_ae_batch  # noqa: E402
from src.module import PC2WireframeModule  # noqa: E402

NUM_EDGE_POINTS = 32
LATENT_BUDGET = 4096


# ----------------------------------------------------------------------
# Checkpoint loading
# ----------------------------------------------------------------------
def _resolve_ckpt(path: str, metric: str = "val_score", best: str = "max") -> str:
    """Resolve a checkpoint file or pick the best one in a directory."""
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


def load_model(ckpt_path: str, device: str) -> PC2WireframeModule:
    """Load a ``PC2WireframeModule`` from a Lightning checkpoint.

    ``curve_vae_ckpt`` is overridden to ``None`` so loading does not depend on a
    (possibly stale) stage-1 path -- the frozen curve VAE weights live in the
    stage-2 state_dict.
    """
    model = PC2WireframeModule.load_from_checkpoint(
        ckpt_path, map_location="cpu", curve_vae_ckpt=None)
    model.eval().to(device)
    return model


# ----------------------------------------------------------------------
# Serialisation helpers
# ----------------------------------------------------------------------
def _resample_curve(curve: np.ndarray, num_points: int) -> np.ndarray:
    """Resample one ``(U, 3)`` polyline to exactly ``num_points`` points."""
    curve = np.asarray(curve, dtype=np.float64).reshape(-1, 3)
    if curve.shape[0] == num_points:
        return curve.astype(np.float32)
    if curve.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if curve.shape[0] == 1:
        return np.repeat(curve.astype(np.float32), num_points, axis=0)
    t_old = np.linspace(0.0, 1.0, curve.shape[0])
    t_new = np.linspace(0.0, 1.0, num_points)
    out = np.stack([np.interp(t_new, t_old, curve[:, c]) for c in range(3)],
                   axis=-1)
    return out.astype(np.float32)


def _build_sample_npz(
    pred: dict[str, np.ndarray],
    latent: np.ndarray,
    num_per_edge: int,
) -> dict[str, np.ndarray]:
    """Convert one decoded wireframe (+ latent) to submission npz arrays."""
    latent = np.asarray(latent, dtype=np.float32).reshape(-1)
    if latent.size == 0:
        raise ValueError("empty latent")
    if latent.size > LATENT_BUDGET:
        raise ValueError(
            f"latent size {latent.size} exceeds the {LATENT_BUDGET} budget")

    vertices = np.asarray(pred.get("vertices"), dtype=np.float32).reshape(-1, 3)
    edge_index = np.asarray(pred.get("edge_index"), dtype=np.int64).reshape(-1, 2)

    raw_ep = np.asarray(pred.get("edge_points"), dtype=np.float32)
    n_edges = int(edge_index.shape[0])
    if raw_ep.size and raw_ep.ndim == 3 and raw_ep.shape[0] == n_edges:
        ep = np.empty((n_edges, num_per_edge, 3), dtype=np.float32)
        for e in range(n_edges):
            ep[e] = _resample_curve(raw_ep[e], num_per_edge)
    else:
        ep = np.zeros((n_edges, num_per_edge, 3), dtype=np.float32)
        if n_edges and vertices.shape[0]:
            t = np.linspace(0.0, 1.0, num_per_edge)[None, :, None]
            a = vertices[edge_index[:, 0]][:, None, :]
            b = vertices[edge_index[:, 1]][:, None, :]
            ep = (a * (1.0 - t) + b * t).astype(np.float32)

    return {
        "latent": latent,
        "vertices": vertices,
        "edge_index": edge_index.astype(np.int32),
        "edge_points": ep.astype(np.float32),
        "num_vertices": np.int32(vertices.shape[0]),
        "num_edges": np.int32(n_edges),
    }


# ----------------------------------------------------------------------
# Multi-GPU layout
# ----------------------------------------------------------------------
def _sub_dir(out_dir: str) -> str:
    return os.path.join(out_dir, "submission")


def _sample_dir(out_dir: str) -> str:
    return os.path.join(_sub_dir(out_dir), "sample_edge")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--ckpt",
        default="logs/pc2wireframe/checkpoints",
        help="PC2WireframeModule ckpt file or dir (best val_score auto-picked)")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--test-pc-dir", default="data/test/sample_pointcloud",
                   help="point-cloud dir for the GT-less test split")
    p.add_argument("--out-dir", default="logs/submission")
    p.add_argument("--zip-name", default="submission.zip")
    p.add_argument("--batch-size", type=int, default=4,
                   help="shapes encoded/decoded per forward pass")
    p.add_argument("--max-pc-points", type=int, default=0,
                   help="cap input cloud size (0 = native); memory knob only")
    # decode post-processing (override the values baked into the ckpt)
    p.add_argument("--ethr", type=float, default=None,
                   help="keep an edge iff sigmoid(confidence) >= this "
                        "(default: the value baked into the checkpoint)")
    p.add_argument("--merge-tol", type=float, default=None,
                   help="endpoint merge tolerance (default: from the checkpoint)")
    p.add_argument("--min-edges", type=int, default=1,
                   help="floor on edges per shape (top-k fallback; 0 = allow empty)")
    p.add_argument("--limit", type=int, default=0,
                   help="only export the first N shapes (0 = all; debugging)")
    p.add_argument("--no-progress", action="store_true")
    # multi-GPU parallelism
    p.add_argument("--spawn", type=int, default=0,
                   help="launch N worker processes (one per GPU) then merge; "
                        "0 = run in this single process")
    p.add_argument("--gpus", default=None,
                   help="comma-separated GPU ids for --spawn (default 0..spawn-1)")
    p.add_argument("--world-size", type=int, default=1,
                   help="[worker] total number of shards")
    p.add_argument("--rank", type=int, default=0,
                   help="[worker] this shard id in [0, world-size)")
    p.add_argument("--merge-only", action="store_true",
                   help="skip inference; just build latent_pack + zip")
    p.add_argument("--resume", action="store_true",
                   help="skip shapes that already have an output npz")
    return p.parse_args()


# ----------------------------------------------------------------------
# Worker: run the pipeline on this rank's shard, write loose npz
# ----------------------------------------------------------------------
def run_worker(args: argparse.Namespace) -> None:
    world = max(1, int(args.world_size))
    rank = int(args.rank)

    ckpt = _resolve_ckpt(args.ckpt, "val_score", "max")
    print(f"[rank {rank}] ckpt: {ckpt}")
    print(f"[rank {rank}] decode: ethr={args.ethr} merge_tol={args.merge_tol} "
          f"min_edges={args.min_edges}")
    model = load_model(ckpt, args.device)
    num_per_edge = int(model.hparams.num_per_edge)
    ethr = (args.ethr if args.ethr is not None else float(model.hparams.ethr))
    merge_tol = (args.merge_tol if args.merge_tol is not None
                 else float(model.hparams.merge_tol))
    # min_edges override applied through the module's hparam.
    model.hparams.min_edges = int(args.min_edges)
    k_latent = int(model.pc_encoder.compressor.latent_budget)

    dataset = PointCloudDataset(
        pointcloud_dir=args.test_pc_dir, max_pc_points=args.max_pc_points)
    n_total = len(dataset)
    indices = list(range(n_total))
    if args.limit and args.limit > 0:
        indices = indices[: args.limit]
    shard = indices[rank::world]

    sample_dir = _sample_dir(args.out_dir)
    os.makedirs(sample_dir, exist_ok=True)

    n_skipped = 0
    if args.resume:
        pending: list[int] = []
        for i in shard:
            stem = os.path.splitext(
                os.path.basename(dataset.files[i % len(dataset.files)]))[0]
            if os.path.isfile(os.path.join(sample_dir, f"{stem}.npz")):
                n_skipped += 1
            else:
                pending.append(i)
        shard = pending
    print(f"[rank {rank}] {len(shard)} shapes to do "
          f"(skipped {n_skipped} existing, world_size={world})")

    stems: list[str] = []
    n_failed = 0
    bs = max(1, args.batch_size)
    use_tqdm = (world == 1 and rank == 0 and not args.no_progress)
    pbar = None
    if use_tqdm:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(shard), unit="shape", desc="export")
        except Exception:  # noqa: BLE001
            pbar = None
    heartbeat = max(1, len(shard) // 20)
    done = 0

    def _tick() -> None:
        nonlocal done
        done += 1
        if pbar is not None:
            pbar.update(1)
        elif not args.no_progress and (
                done % heartbeat == 0 or done == len(shard)):
            print(f"[rank {rank}] {done}/{len(shard)}", flush=True)

    def _emit(stem: str, sample_npz: dict[str, np.ndarray]) -> None:
        np.savez(os.path.join(sample_dir, f"{stem}.npz"), **sample_npz)
        stems.append(stem)

    def _fallback_npz() -> dict[str, np.ndarray]:
        return {
            "latent": np.zeros(k_latent, dtype=np.float32),
            "vertices": np.zeros((0, 3), dtype=np.float32),
            "edge_index": np.zeros((0, 2), dtype=np.int32),
            "edge_points": np.zeros((0, NUM_EDGE_POINTS, 3), dtype=np.float32),
            "num_vertices": np.int32(0),
            "num_edges": np.int32(0),
        }

    for start in range(0, len(shard), bs):
        chunk = shard[start:start + bs]

        samples: list[dict[str, Any]] = []
        for i in chunk:
            try:
                s = dataset[i]
            except Exception as exc:  # noqa: BLE001
                stem = os.path.splitext(
                    os.path.basename(dataset.files[i % len(dataset.files)]))[0]
                print(f"[rank {rank}] load failed {stem}: {exc}; "
                      f"writing empty fallback", file=sys.stderr)
                _emit(stem, _fallback_npz())
                n_failed += 1
                _tick()
                continue
            if int(s["point_cloud"].shape[0]) < 1:
                stem = str(s["shape_id"])
                print(f"[rank {rank}] empty point cloud {stem}; "
                      f"writing empty fallback", file=sys.stderr)
                _emit(stem, _fallback_npz())
                n_failed += 1
                _tick()
                continue
            samples.append(s)

        if not samples:
            continue

        batch = collate_ae_batch(samples)
        pc = batch["point_cloud"].to(args.device)
        offset = batch["pc_offset"].to(args.device)

        with torch.no_grad():
            out = model.forward(pc, offset, sample=False)
            preds = model._assemble(out["preds"], ethr=ethr, merge_tol=merge_tol)
            z = out["z"].reshape(out["z"].shape[0], -1).detach().cpu().numpy()

        for j, s in enumerate(samples):
            stem = str(s["shape_id"])
            latent = z[j].reshape(-1)
            try:
                sample_npz = _build_sample_npz(preds[j], latent, num_per_edge)
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                print(f"[rank {rank}] decode failed {stem}: {exc}; "
                      f"writing empty wireframe (latent kept)", file=sys.stderr)
                sample_npz = _fallback_npz()
                lat = np.asarray(latent, dtype=np.float32).reshape(-1)
                if lat.size == k_latent:
                    sample_npz["latent"] = lat
            _emit(stem, sample_npz)
            _tick()

    if pbar is not None:
        pbar.close()
    print(f"[rank {rank}] wrote {len(stems)} sample(s) "
          f"({n_failed} failed) -> {sample_dir}")


# ----------------------------------------------------------------------
# Merge: gather latents -> latent_pack.npz, then zip submission/
# ----------------------------------------------------------------------
def run_merge(args: argparse.Namespace) -> None:
    sample_dir = _sample_dir(args.out_dir)
    files = sorted(glob.glob(os.path.join(sample_dir, "*.npz")))
    if not files:
        raise RuntimeError(
            f"No sample npz under {sample_dir!r}; did the workers run?")

    all_stems: list[str] = []
    all_latents: list[np.ndarray] = []
    widths: set[int] = set()
    n_no_edges = 0
    n_no_verts = 0
    n_zero_latent = 0
    empty_stems: list[str] = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        with np.load(f, allow_pickle=True) as z:
            if "latent" not in z.files:
                print(f"[merge] {stem}: no 'latent' field; skipping in pack",
                      file=sys.stderr)
                continue
            lat = np.asarray(z["latent"], dtype=np.float32).reshape(-1)
            n_v = int(z["num_vertices"]) if "num_vertices" in z.files else int(
                np.asarray(z["vertices"]).reshape(-1, 3).shape[0]
                if "vertices" in z.files else 0)
            n_e = int(z["num_edges"]) if "num_edges" in z.files else int(
                np.asarray(z["edge_index"]).reshape(-1, 2).shape[0]
                if "edge_index" in z.files else 0)
        if n_e == 0:
            n_no_edges += 1
            if len(empty_stems) < 20:
                empty_stems.append(stem)
        if n_v == 0:
            n_no_verts += 1
        if lat.size and not np.any(lat):
            n_zero_latent += 1
        all_stems.append(stem)
        all_latents.append(lat)
        widths.add(int(lat.size))
    if not all_stems:
        raise RuntimeError("No latents found in sample npz; nothing to merge.")

    n = len(all_stems)
    if n_no_edges or n_no_verts or n_zero_latent:
        pct = 100.0 * n_no_edges / max(1, n)
        print("=" * 64, file=sys.stderr)
        print("[merge][WARN] empty/fallback wireframes in submission:",
              file=sys.stderr)
        print(f"  no edges    : {n_no_edges}/{n} ({pct:.1f}%)  <- score ~0",
              file=sys.stderr)
        print(f"  no vertices : {n_no_verts}/{n}", file=sys.stderr)
        print(f"  zero latent : {n_zero_latent}/{n}", file=sys.stderr)
        if empty_stems:
            more = " ..." if n_no_edges > len(empty_stems) else ""
            print(f"  e.g. {empty_stems}{more}", file=sys.stderr)
        print("=" * 64, file=sys.stderr)
    else:
        print(f"[merge] all {n} wireframes non-empty.")
    if len(widths) != 1:
        raise RuntimeError(
            f"Inconsistent latent widths across samples: {sorted(widths)}. "
            "All shapes must share the same K for latent_pack.")

    latents_arr = np.stack(all_latents, axis=0).astype(np.float32)
    max_stem_len = max(len(s) for s in all_stems)
    stems_arr = np.array(all_stems, dtype=f"<U{max_stem_len}")
    np.savez(os.path.join(_sub_dir(args.out_dir), "latent_pack.npz"),
             stems=stems_arr, latents=latents_arr)

    sub_dir = _sub_dir(args.out_dir)
    out_zip = os.path.join(args.out_dir, args.zip_name)
    if os.path.isfile(out_zip):
        os.remove(out_zip)
    n_files = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(sub_dir):
            for f in files:
                full = os.path.join(root, f)
                arc = os.path.relpath(full, args.out_dir)  # -> submission/...
                zf.write(full, arc)
                n_files += 1
    print(f"[merge] {len(all_stems)} sample(s), {n_files} files -> {out_zip}")


# ----------------------------------------------------------------------
# Orchestrate: launch one worker subprocess per GPU, then merge
# ----------------------------------------------------------------------
def _clean_out_dir(out_dir: str) -> None:
    import shutil

    sd = _sample_dir(out_dir)
    if os.path.isdir(sd):
        shutil.rmtree(sd)
    os.makedirs(sd, exist_ok=True)


def orchestrate(args: argparse.Namespace) -> None:
    import subprocess

    if args.gpus:
        gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    else:
        gpu_ids = [str(i) for i in range(int(args.spawn))]
    world = len(gpu_ids)
    if world == 0:
        raise SystemExit("--spawn must be >= 1 (or pass --gpus).")

    os.makedirs(args.out_dir, exist_ok=True)
    if not args.resume:
        _clean_out_dir(args.out_dir)
    print(f"[spawn] launching {world} worker(s) on GPU(s) {gpu_ids}"
          f"{' (resume)' if args.resume else ''}")

    base = [
        sys.executable, os.path.abspath(__file__),
        "--world-size", str(world),
        "--device", "cuda",
        "--ckpt", args.ckpt,
        "--test-pc-dir", args.test_pc_dir,
        "--out-dir", args.out_dir,
        "--batch-size", str(args.batch_size),
        "--max-pc-points", str(args.max_pc_points),
        "--min-edges", str(args.min_edges),
    ]
    if args.ethr is not None:
        base += ["--ethr", str(args.ethr)]
    if args.merge_tol is not None:
        base += ["--merge-tol", str(args.merge_tol)]
    if args.limit and args.limit > 0:
        base += ["--limit", str(args.limit)]
    if args.resume:
        base += ["--resume"]

    procs = []
    for rank, gpu in enumerate(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = base + ["--rank", str(rank)]
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    failed = [i for i, c in enumerate(codes) if c != 0]
    if failed:
        raise SystemExit(f"worker rank(s) {failed} failed (exit codes "
                         f"{[codes[i] for i in failed]}).")

    run_merge(args)


def main() -> None:
    args = parse_args()

    if args.merge_only:
        run_merge(args)
        return

    if args.spawn and args.spawn > 0:
        orchestrate(args)
        return

    if args.world_size <= 1 and args.rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        if not args.resume:
            _clean_out_dir(args.out_dir)
        run_worker(args)
        run_merge(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
