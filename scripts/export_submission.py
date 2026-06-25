#!/usr/bin/env python3
"""Export a competition submission zip from the VQVAE WireframeAE model.

Serialises every test shape into the layout the official evaluator (and the
``pc2wireframe_baseline`` submission) expect::

    submission/
        latent_pack.npz                 # stems (N,)  + latents (N, K)
        sample_edge/<stem>.npz
            latent       : (K,) float32, K <= 4096   (flat RVQ indices)
            vertices     : (V, 3) float32
            edge_index   : (E, 2) int32   (indexes into vertices)
            edge_points  : (E, 32, 3) float32
            num_vertices : () int32
            num_edges    : () int32

and finally zips ``submission/`` into ``submission.zip``.

Pipeline per shape::

    raw test point cloud
        --[UtoniaEncoder -> multi-scale z_s]-->
        --[MultiScaleResidualVQ -> flat indices (K,)]-->     # the submission
    flat indices
        --[decode_indices -> z_q -> JointSetDecoder -> assemble_wireframe]-->
    wireframe {vertices, edge_index, edge_points}

The wireframe is reconstructed from the **submitted indices** alone
(indices -> codebooks -> z_q -> decoder), so the export is a guaranteed
round-trip with the evaluator's view of the latent.

Coordinate frame
----------------
The model works directly in the **raw** point-cloud frame (no per-shape
normalization), so vertices / curves are written as decoded.

The per-sample ``latent`` is the flat RVQ index vector
``(K,) = sum_s N_s * n_q`` floats (<= 4096, guaranteed by the
``MultiScaleResidualVQ`` budget check at construction).

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

Manual launch (instead of ``--spawn``)::

    CUDA_VISIBLE_DEVICES=r python scripts/export_submission.py \
        --world-size 8 --rank r --out-dir logs/submission   # for r in 0..7
    python scripts/export_submission.py --merge-only --out-dir logs/submission
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
from src.models.joint_set_decoder import JointSetDecoder  # noqa: E402
from src.models.quantizer import MultiScaleResidualVQ  # noqa: E402
from src.models.utonia_encoder import UtoniaEncoder  # noqa: E402
from src.models.vae import AutoencoderKL1D  # noqa: E402
from src.models.vae.curve_packing import decode_curve_latent  # noqa: E402
from src.recon import assemble_wireframe  # noqa: E402

NUM_EDGE_POINTS = 32
LATENT_BUDGET = 4096


# ----------------------------------------------------------------------
# Checkpoint loading (build encoder + decoder directly; no pytorch3d import)
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


def _strip_prefix(state: dict[str, torch.Tensor], prefix: str
                  ) -> dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def load_model(ckpt_path: str, device: str
               ) -> tuple[UtoniaEncoder, MultiScaleResidualVQ,
                          JointSetDecoder, AutoencoderKL1D, dict[str, Any]]:
    """Build the encoder + RVQ + joint decoder + curve VAE from a checkpoint."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters", {}) or {}
    state = ck["state_dict"] if "state_dict" in ck else ck

    encoder = UtoniaEncoder(**(hp.get("pc_encoder") or {}))
    quantizer = MultiScaleResidualVQ(
        scale_tokens=encoder.scale_tokens,
        dim=encoder.latent_dim,
        **(hp.get("quantizer") or {}),
    )
    curve_vae = AutoencoderKL1D(**(hp.get("curve_vae") or {}))
    dec_cfg = dict(hp.get("decoder") or {})
    dec_cfg.setdefault("num_scales", len(encoder.scale_tokens))
    dec_cfg.setdefault("latent_dim", encoder.latent_dim)
    dec_cfg["curve_latent_dim"] = int(
        curve_vae.config.latent_channels * curve_vae.latent_len)
    decoder = JointSetDecoder(**dec_cfg)

    miss_e, unexp_e = encoder.load_state_dict(
        _strip_prefix(state, "encoder."), strict=False)
    miss_q, unexp_q = quantizer.load_state_dict(
        _strip_prefix(state, "quantizer."), strict=False)
    miss_d, unexp_d = decoder.load_state_dict(
        _strip_prefix(state, "decoder."), strict=False)
    miss_c, unexp_c = curve_vae.load_state_dict(
        _strip_prefix(state, "curve_vae."), strict=False)

    comp_missing = [k for k in miss_e if not k.startswith("backbone.")]
    if comp_missing:
        print(f"[warn] missing encoder (compressor) keys: {comp_missing}")
    if unexp_e:
        print(f"[warn] unexpected encoder keys: {unexp_e[:8]}"
              f"{' ...' if len(unexp_e) > 8 else ''}")
    if miss_q:
        print(f"[warn] missing quantizer keys: {miss_q[:8]}"
              f"{' ...' if len(miss_q) > 8 else ''}")
    if miss_d:
        print(f"[warn] missing decoder keys: {miss_d}")
    if unexp_d:
        print(f"[warn] unexpected decoder keys: {unexp_d}")
    if miss_c:
        print(f"[warn] missing curve_vae keys: {miss_c[:8]}"
              f"{' ...' if len(miss_c) > 8 else ''}")
    if unexp_c:
        print(f"[warn] unexpected curve_vae keys: {unexp_c[:8]}"
              f"{' ...' if len(unexp_c) > 8 else ''}")

    encoder.eval().to(device)
    quantizer.eval().to(device)
    decoder.eval().to(device)
    curve_vae.eval().to(device)
    return encoder, quantizer, decoder, curve_vae, hp


# ----------------------------------------------------------------------
# Decode: encoder latent + decoder fields -> wireframe (mirrors module.decode)
# ----------------------------------------------------------------------
@torch.no_grad()
def decode_batch(
    out: dict[str, torch.Tensor],
    curve_vae: AutoencoderKL1D,
    hp: dict[str, Any],
    *,
    vthr: float | None = None,
    ethr: float | None = None,
    min_edges: int | None = None,
) -> list[dict[str, np.ndarray]]:
    """Assemble joint vertex+edge decoder outputs into wireframes (numpy).

    ``vthr`` / ``ethr`` (vertex / edge existence thresholds) override the values
    baked into the checkpoint when provided. Each edge's endpoints are the top-2
    vertices under the association matrix; the decoded canonical curve is then
    denormalised onto them. ``min_edges`` is the floor so the wireframe is never
    empty.
    """
    npe = int(hp.get("num_per_edge", NUM_EDGE_POINTS))
    vt = float(vthr if vthr is not None else hp.get("vthr", 0.5))
    et = float(ethr if ethr is not None else hp.get("ethr", 0.5))
    mn = int(min_edges if min_edges is not None else hp.get("min_edges", 1))
    mv = int(hp.get("min_vertices", 2))

    vprob = torch.sigmoid(out["vertex_exist_logit"]).cpu().numpy()
    vcoord = out["vertex_coord"].cpu().numpy()
    eprob = torch.sigmoid(out["edge_exist_logit"]).cpu().numpy()
    assoc = torch.sigmoid(out["assoc_logit"]).cpu().numpy()
    b, ne, _ = out["curve_latent"].shape
    flat = out["curve_latent"].reshape(b * ne, -1)
    curves = decode_curve_latent(
        curve_vae, flat, num_points=npe, pin_endpoints=True)
    curves = curves.reshape(b, ne, npe, 3).cpu().numpy()
    return [
        assemble_wireframe(
            vprob[i], vcoord[i], eprob[i], assoc[i], curves[i],
            vthr=vt, ethr=et, num_per_edge=npe,
            min_vertices=mv, min_edges=mn,
        )
        for i in range(b)
    ]


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
        help="WireframeAE ckpt file or dir (best val_score auto-picked)")
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
    # decode post-processing (override the thresholds baked into the ckpt)
    p.add_argument("--vthr", type=float, default=None,
                   help="keep a vertex iff sigmoid(exist) >= this "
                        "(default: the value baked into the checkpoint)")
    p.add_argument("--ethr", type=float, default=None,
                   help="keep an edge iff sigmoid(exist) >= this "
                        "(default: the value baked into the checkpoint)")
    p.add_argument("--min-edges", type=int, default=1,
                   help="floor on edges per shape: if fewer pass --ethr, fall "
                        "back to the top-k by probability so the wireframe is "
                        "never empty (0 = allow empty).")
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
# Worker: run the pipeline on this rank's shard, write loose npz
# ----------------------------------------------------------------------
def run_worker(args: argparse.Namespace) -> None:
    world = max(1, int(args.world_size))
    rank = int(args.rank)

    ckpt = _resolve_ckpt(args.ckpt, "val_score", "max")
    print(f"[rank {rank}] ckpt: {ckpt}")
    print(f"[rank {rank}] decode: vthr={args.vthr} ethr={args.ethr} "
          f"min_edges={args.min_edges}")
    encoder, quantizer, decoder, curve_vae, hp = load_model(ckpt, args.device)

    k_latent = int(quantizer.total_indices)

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
            z_list = encoder(pc, offset)                  # list[(b, N_s, D)]
            indices = quantizer(z_list)["indices"]        # (b, T) submission
            # Decode from the *submitted* indices (round-trip): indices ->
            # codebooks -> z_q -> graph decoder.
            z_q = quantizer.decode_indices(indices)
            out = decoder(z_q)
            preds = decode_batch(
                out, curve_vae, hp,
                vthr=args.vthr, ethr=args.ethr, min_edges=args.min_edges)

        for j, s in enumerate(samples):
            stem = str(s["shape_id"])
            latent = indices[j].detach().cpu().numpy().reshape(-1)
            try:
                sample_npz = _build_sample_npz(
                    preds[j], latent, NUM_EDGE_POINTS)
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
    n_no_edges = 0          # empty wireframes (no edges => zero score)
    n_no_verts = 0          # empty wireframes (no vertices at all)
    n_zero_latent = 0       # latent is all-zeros (decode/load fallback)
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

    # Surface empty / fallback wireframes loudly: they serialise fine and never
    # abort the export, but score ~0, so a silent pile of them quietly tanks the
    # submission.
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
    if args.vthr is not None:
        base += ["--vthr", str(args.vthr)]
    if args.ethr is not None:
        base += ["--ethr", str(args.ethr)]
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
