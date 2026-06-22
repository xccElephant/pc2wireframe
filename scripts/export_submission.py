#!/usr/bin/env python3
"""Export a competition submission zip from the **two-stage** PC2Wireframe model.

This stitches the two trained checkpoints together exactly like
``scripts/vis_pipeline_val.py`` -- only instead of rendering a comparison figure
it serialises every test shape into the layout the official evaluator (and the
``pc2wireframe_baseline`` submission) expect::

    submission/
        latent_pack.npz                 # stems (N,)  + latents (N, K)
        sample_edge/<stem>.npz
            latent       : (K,) float32, K <= 4096
            vertices     : (V, 3) float32
            edge_index   : (E, 2) int32   (indexes into vertices)
            edge_points  : (E, 32, 3) float32
            num_vertices : () int32
            num_edges    : () int32

and finally zips ``submission/`` into ``submission.zip``.

Pipeline per shape (identical to ``vis_pipeline_val.py``)::

    raw test point cloud
        --[stage 1: UtoniaEncoder -> latent z; RFPointSetVelocity ODE 0->1]-->
    predicted anchor set x1_hat (N, 3)
        --[stage 2: WireframeGrouper -> src.recon.group_wireframe]-->
    wireframe {vertices, edge_index, edge_points}

Coordinate frame
----------------
The dataset normalises every shape into a unit cube (``(x - center) / scale``).
The grouper therefore predicts in that normalised frame, so before writing we
**de-normalise back** to the on-disk test frame via ``x * pc_scale + pc_center``
-- the same frame the released baseline submission lives in (and the frame the
server-side evaluator scores against).

The per-sample ``latent`` is the stage-1 encoder latent ``z`` flattened to
``(K,) = latent_num * latent_dim`` floats (must be <= 4096, which the
``LatentCompressor`` budget already guarantees).

Usage (from the project root)::

    # single GPU (defaults mirror scripts/vis_pipeline_val.py)
    python scripts/export_submission.py \
        --stage1-ckpt logs/pc2wireframe/f2yry4qc/checkpoints \
        --stage2-ckpt logs/pc2wireframe/ro1tlty1/checkpoints \
        --test-pc-dir data/test/sample_pointcloud \
        --out-dir logs/submission

    # 8-GPU data-parallel (one worker process per GPU, then auto-merge + zip)
    python scripts/export_submission.py --spawn 8 --out-dir logs/submission

Multi-GPU layout: each worker handles a strided shard
(``indices[rank::world_size]``) and writes loose ``submission/sample_edge/*.npz``
files; the merge step rebuilds ``latent_pack.npz`` by scanning those files and
zips the folder. The per-shape result is seeded by stem, so it is identical no
matter how many GPUs are used.

Resume an interrupted run with ``--resume`` (skips shapes that already have an
output npz and does NOT wipe the out-dir)::

    python scripts/export_submission.py --spawn 8 --out-dir logs/submission --resume

Manual launch (instead of ``--spawn``)::

    CUDA_VISIBLE_DEVICES=r python scripts/export_submission.py \
        --world-size 8 --rank r --out-dir logs/submission   # for r in 0..7
    python scripts/export_submission.py --merge-only --out-dir logs/submission
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from typing import Any

import numpy as np
import torch

# Make the project importable as a package (``src.*``) and let us reuse the
# proven stage-stitching helpers from the visualisation script.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_PROJECT_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data.dataset import PointCloudDataset, collate_rf_batch  # noqa: E402
from vis_pipeline_val import (  # noqa: E402
    _resolve_ckpt,
    decode_grouper,
    load_stage1,
    load_stage2,
    rf_sample,
)

NUM_EDGE_POINTS = 32
LATENT_BUDGET = 4096


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


def _bytes_for_npz(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _denormalize(x: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """Invert the dataset unit-cube transform: ``x * scale + center``."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32)
    return (x * float(scale) + center.reshape(1, -1)).astype(np.float32)


def _build_sample_npz(
    pred: dict[str, np.ndarray],
    latent: np.ndarray,
    center: np.ndarray,
    scale: float,
    num_per_edge: int,
) -> dict[str, np.ndarray]:
    """Convert one decoded wireframe (+ latent) to submission npz arrays.

    ``pred`` is in the normalised frame; ``vertices`` / ``edge_points`` are
    de-normalised back to the on-disk test frame before writing.
    """
    latent = np.asarray(latent, dtype=np.float32).reshape(-1)
    if latent.size == 0:
        raise ValueError("empty latent")
    if latent.size > LATENT_BUDGET:
        raise ValueError(
            f"latent size {latent.size} exceeds the {LATENT_BUDGET} budget")

    vertices = np.asarray(pred.get("vertices"), dtype=np.float32).reshape(-1, 3)
    vertices = _denormalize(vertices, center, scale)

    edge_index = np.asarray(pred.get("edge_index"), dtype=np.int64).reshape(-1, 2)

    raw_ep = np.asarray(pred.get("edge_points"), dtype=np.float32)
    n_edges = int(edge_index.shape[0])
    if raw_ep.size and raw_ep.ndim == 3 and raw_ep.shape[0] == n_edges:
        ep = np.empty((n_edges, num_per_edge, 3), dtype=np.float32)
        for e in range(n_edges):
            ep[e] = _resample_curve(raw_ep[e], num_per_edge)
        ep = _denormalize(ep.reshape(-1, 3), center, scale).reshape(
            n_edges, num_per_edge, 3)
    else:
        # No usable curves: synthesise straight 32-point polylines from the
        # endpoint vertices so the field stays consistent with edge_index.
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
# Workers write *loose* npz files straight to ``<out_dir>/submission/`` and a
# per-rank latent shard under ``<out_dir>/_parts/``; the merge step then builds
# ``latent_pack.npz`` and zips the whole folder. Sharding is strided
# (``indices[rank::world_size]``) so the load is balanced, and every shape is
# decoded under a stable per-stem ``np.random`` seed so the result is identical
# regardless of how many GPUs / shards are used.
def _sub_dir(out_dir: str) -> str:
    return os.path.join(out_dir, "submission")


def _sample_dir(out_dir: str) -> str:
    return os.path.join(_sub_dir(out_dir), "sample_edge")


def _stem_seed(stem: str) -> int:
    import zlib
    return int(zlib.crc32(stem.encode("utf-8")) & 0xFFFFFFFF)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--stage1-ckpt",
        default="logs/pc2wireframe/f2yry4qc/checkpoints",
        help="stage-1 (RF) ckpt file or dir (best val_loss auto-picked)")
    p.add_argument(
        "--stage2-ckpt",
        default="logs/pc2wireframe/ro1tlty1/checkpoints",
        help="stage-2 (grouper) ckpt file or dir (best val_score auto-picked)")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--test-pc-dir", default="data/test/sample_pointcloud",
                   help="point-cloud dir for the GT-less test split")
    p.add_argument("--out-dir", default="logs/submission")
    p.add_argument("--zip-name", default="submission.zip")
    p.add_argument("--batch-size", type=int, default=4,
                   help="shapes encoded/sampled per stage-1 forward pass")
    p.add_argument("--max-pc-points", type=int, default=0,
                   help="cap input cloud size (0 = native); memory knob only")
    # stage-1 sampler overrides (default: read from ckpt hyper-parameters)
    p.add_argument("--ode-steps", type=int, default=None)
    p.add_argument("--ode-method", default=None)
    p.add_argument("--sample-seed", type=int, default=None)
    p.add_argument("--limit", type=int, default=0,
                   help="only export the first N shapes (0 = all; debugging)")
    p.add_argument("--no-progress", action="store_true")
    # multi-GPU parallelism
    p.add_argument("--spawn", type=int, default=0,
                   help="launch N worker processes (one per GPU) then merge; "
                        "0 = run in this single process")
    p.add_argument("--gpus", default=None,
                   help="comma-separated GPU ids for --spawn "
                        "(default 0..spawn-1)")
    p.add_argument("--world-size", type=int, default=1,
                   help="[worker] total number of shards")
    p.add_argument("--rank", type=int, default=0,
                   help="[worker] this shard id in [0, world-size)")
    p.add_argument("--merge-only", action="store_true",
                   help="skip inference; just build latent_pack + zip from "
                        "an already-populated --out-dir")
    p.add_argument("--resume", action="store_true",
                   help="skip shapes that already have "
                        "submission/sample_edge/<stem>.npz (do NOT wipe the "
                        "out-dir); continue an interrupted export")
    return p.parse_args()


# ----------------------------------------------------------------------
# Worker: run the pipeline on this rank's shard, write loose npz + latent part
# ----------------------------------------------------------------------
def run_worker(args: argparse.Namespace) -> None:
    world = max(1, int(args.world_size))
    rank = int(args.rank)

    s1_ckpt = _resolve_ckpt(args.stage1_ckpt, "val_loss", "min")
    s2_ckpt = _resolve_ckpt(args.stage2_ckpt, "val_score", "max")
    print(f"[rank {rank}] stage-1: {s1_ckpt}")
    print(f"[rank {rank}] stage-2: {s2_ckpt}")
    encoder, rf_net, hp1 = load_stage1(s1_ckpt, args.device)
    grouper, hp2 = load_stage2(s2_ckpt, args.device)

    # Latent width (K = num_tokens * latent_dim) -- needed to emit a correctly
    # shaped fallback latent for degenerate (empty) point clouds.
    k_latent = int(encoder.compressor.num_tokens * encoder.compressor.latent_dim)

    num_points = int(hp1.get("wf_num_points", 8192))
    ode_steps = int(args.ode_steps if args.ode_steps is not None
                    else hp1.get("ode_steps", 50))
    ode_method = str(args.ode_method if args.ode_method is not None
                     else hp1.get("ode_method", "euler"))
    sample_seed = int(args.sample_seed if args.sample_seed is not None
                      else hp1.get("sample_seed", 0))
    if rank == 0:
        print(f"Stage-1 sampler: N={num_points}  steps={ode_steps}  "
              f"method={ode_method}  seed={sample_seed}")

    dataset = PointCloudDataset(
        pointcloud_dir=args.test_pc_dir, max_pc_points=args.max_pc_points)
    n_total = len(dataset)
    indices = list(range(n_total))
    if args.limit and args.limit > 0:
        indices = indices[: args.limit]
    shard = indices[rank::world]

    sample_dir = _sample_dir(args.out_dir)
    os.makedirs(sample_dir, exist_ok=True)

    # Resume: drop shapes that already have an output npz so an interrupted
    # run continues instead of recomputing everything.
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
    latents_per_sample: list[np.ndarray] = []
    n_failed = 0

    bs = max(1, args.batch_size)
    # A single-process run gets a tqdm bar; under multi-GPU spawn the bars of
    # different ranks would clobber each other on the shared terminal, so every
    # worker instead prints a periodic flushed heartbeat line.
    use_tqdm = (world == 1 and rank == 0 and not args.no_progress)
    pbar = None
    if use_tqdm:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(shard), unit="shape", desc="export")
        except Exception:  # noqa: BLE001 - tqdm is optional
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
        latents_per_sample.append(sample_npz["latent"])

    def _fallback_npz() -> dict[str, np.ndarray]:
        """Empty wireframe + zero latent (keeps a degenerate stem covered)."""
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

        # Load this chunk defensively: a corrupt / empty point cloud must not
        # crash the whole export. Such shapes get a fallback (empty) entry so
        # the stem stays present in the submission, and never reach the model
        # (an empty cloud would make PTv3 / the compressor produce NaNs).
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

        batch = collate_rf_batch(samples)
        pc = batch["point_cloud"].to(args.device)
        offset = batch["pc_offset"].to(args.device)

        with torch.no_grad():
            z = encoder(pc, offset)                       # (b, K, D)
            anchors = rf_sample(
                rf_net, z, num_points=num_points, ode_steps=ode_steps,
                ode_method=ode_method, sample_seed=sample_seed)  # (b, N, 3)

        for j, s in enumerate(samples):
            stem = str(s["shape_id"])
            center = s["pc_center"].numpy().astype(np.float64)
            scale = float(s["pc_scale"])
            latent = z[j].detach().cpu().numpy().reshape(-1)
            # Stable per-shape seed -> result is independent of shard layout.
            np.random.seed(_stem_seed(stem))
            try:
                pred = decode_grouper(grouper, anchors[j], hp2)
                # The grouper may decode curves at its own resolution
                # (hp2.num_per_edge); the submission requires exactly
                # NUM_EDGE_POINTS per edge, so resample on write.
                sample_npz = _build_sample_npz(
                    pred, latent, center, scale, NUM_EDGE_POINTS)
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                print(f"[rank {rank}] decode failed {stem}: {exc}; "
                      f"writing empty wireframe (latent kept)", file=sys.stderr)
                sample_npz = _fallback_npz()
                # The cloud was valid, so keep the real encoder latent.
                lat = np.asarray(latent, dtype=np.float32).reshape(-1)
                if lat.size == k_latent:
                    sample_npz["latent"] = lat
            _emit(stem, sample_npz)
            _tick()

    if pbar is not None:
        pbar.close()

    # No per-rank latent shard is written: latent_pack is rebuilt at merge time
    # by scanning every submission/sample_edge/*.npz, which makes both
    # multi-GPU and --resume runs self-consistent.
    print(f"[rank {rank}] wrote {len(stems)} sample(s) "
          f"({n_failed} failed) -> {sample_dir}")


# ----------------------------------------------------------------------
# Merge: gather latent shards -> latent_pack.npz, then zip submission/
# ----------------------------------------------------------------------
def run_merge(args: argparse.Namespace) -> None:
    import glob

    sample_dir = _sample_dir(args.out_dir)
    files = sorted(glob.glob(os.path.join(sample_dir, "*.npz")))
    if not files:
        raise RuntimeError(
            f"No sample npz under {sample_dir!r}; did the workers run?")

    # Rebuild latent_pack by reading the latent out of every sample npz, so the
    # pack always matches whatever sample files are present (multi-GPU shards,
    # resumed runs, etc.).
    all_stems: list[str] = []
    all_latents: list[np.ndarray] = []
    widths: set[int] = set()
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        with np.load(f, allow_pickle=True) as z:
            if "latent" not in z.files:
                print(f"[merge] {stem}: no 'latent' field; skipping in pack",
                      file=sys.stderr)
                continue
            lat = np.asarray(z["latent"], dtype=np.float32).reshape(-1)
        all_stems.append(stem)
        all_latents.append(lat)
        widths.add(int(lat.size))
    if not all_stems:
        raise RuntimeError("No latents found in sample npz; nothing to merge.")
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
        "--stage1-ckpt", args.stage1_ckpt,
        "--stage2-ckpt", args.stage2_ckpt,
        "--test-pc-dir", args.test_pc_dir,
        "--out-dir", args.out_dir,
        "--batch-size", str(args.batch_size),
        "--max-pc-points", str(args.max_pc_points),
    ]
    if args.ode_steps is not None:
        base += ["--ode-steps", str(args.ode_steps)]
    if args.ode_method is not None:
        base += ["--ode-method", str(args.ode_method)]
    if args.sample_seed is not None:
        base += ["--sample-seed", str(args.sample_seed)]
    if args.limit and args.limit > 0:
        base += ["--limit", str(args.limit)]
    if args.resume:
        base += ["--resume"]

    procs = []
    for rank, gpu in enumerate(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        # Each rank prints periodic full-line heartbeats (not a tqdm bar), so
        # interleaving them on the shared terminal stays readable and lets you
        # spot a single stuck rank.
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

    # Single-process path (optionally one shard of a manual distributed run).
    if args.world_size <= 1 and args.rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        if not args.resume:
            _clean_out_dir(args.out_dir)
        run_worker(args)
        run_merge(args)
    else:
        # Manual distributed worker: write this shard only; run --merge-only
        # once all ranks finish.
        run_worker(args)


if __name__ == "__main__":
    main()
