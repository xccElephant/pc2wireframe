"""Smoke-test + profile one stage-2 (PC2Wireframe) training iteration.

What it does
------------
Builds the **real** stage-2 model from ``configs/pc2wireframe.yaml`` (PTv3
encoder + latent compressor + transformer wireframe decoder + frozen curve VAE),
runs a few ``forward + loss + backward + optimizer`` steps and prints a per-stage
wall-clock breakdown so you can see where one iteration's time goes:

    data            -- batch fetch (only with --data real)
    enc.ptv3        -- PTv3 backbone forward
    enc.group+comp  -- per-voxel feature grouping + cross-attn latent compressor
    decoder         -- node + edge transformer decoder forward
    build_targets   -- slice packed batch into per-sample GT
    criterion       -- Hungarian matching + matched losses (total)
      .hungarian    -- (subset of criterion) scipy linear_sum_assignment time
      .curve_decode -- (subset of criterion) frozen curve-VAE curve decode time
    backward        -- loss.backward()
    optim           -- optimizer.step()

Examples
--------
    # synthetic batch (no dataset needed), default config, GPU
    python scripts/profile_pc2wireframe.py --steps 5

    # real data loader (times data fetch too)
    python scripts/profile_pc2wireframe.py --data real --steps 5

    # stress the matching cost (max-size graphs)
    python scripts/profile_pc2wireframe.py --nv 768 --ne 1024 --batch-size 16
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pyrootutils
import torch
import yaml

root = pyrootutils.setup_root(
    __file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
)

import src.models.criterion as crit_mod  # noqa: E402
from src.module import PC2WireframeModule  # noqa: E402

# ----------------------------------------------------------------------
# instrument scipy matching + curve decode (no library edits needed)
SUBTIMES: dict[str, float] = {}


def _install_probes() -> None:
    """Wrap the two likely hot-spots so the script can attribute their time."""
    import scipy.optimize as sopt

    orig_lsa = sopt.linear_sum_assignment

    def timed_lsa(*a, **k):
        t = time.perf_counter()
        r = orig_lsa(*a, **k)
        SUBTIMES["hungarian"] = SUBTIMES.get("hungarian", 0.0) + (
            time.perf_counter() - t)
        return r

    crit_mod.linear_sum_assignment = timed_lsa  # used inside _match_*

    orig_decode = crit_mod.decode_curve_latent

    def timed_decode(*a, **k):
        t = time.perf_counter()
        r = orig_decode(*a, **k)
        SUBTIMES["curve_decode"] = SUBTIMES.get("curve_decode", 0.0) + (
            time.perf_counter() - t)
        return r

    crit_mod.decode_curve_latent = timed_decode


# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(root / "configs/pc2wireframe.yaml"),
                    help="stage-2 config (model.init_args is used to build the model)")
    ap.add_argument("--data", choices=["synthetic", "real"], default="synthetic",
                    help="synthetic batch (default) or the real WireframeDataModule")
    ap.add_argument("--data-config", default=str(root / "configs/data.yaml"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=5, help="timed steps")
    ap.add_argument("--warmup", type=int, default=2, help="untimed warmup steps")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--pc-points", type=int, default=4096)
    ap.add_argument("--nv", type=int, default=64, help="synthetic GT vertices/sample")
    ap.add_argument("--ne", type=int, default=96, help="synthetic GT edges/sample")
    ap.add_argument("--no-backward", action="store_true", help="time forward only")
    ap.add_argument("--precision", choices=["32", "bf16", "16"], default="32")
    return ap.parse_args()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


class Stop:
    """Cumulative wall-clock timer keyed by stage name."""

    def __init__(self, device: torch.device):
        self.device = device
        self.t: dict[str, float] = {}

    def __call__(self, name: str):
        dev = self.device

        class _Ctx:
            def __enter__(_s):
                _sync(dev)
                _s.t0 = time.perf_counter()

            def __exit__(_s, *exc):
                _sync(dev)
                self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - _s.t0)

        return _Ctx()


# ----------------------------------------------------------------------
def synth_batch(args, device: torch.device) -> dict:
    """A packed batch in the dataloader's collate layout (random values)."""
    b, n, u = args.batch_size, args.pc_points, 32
    nv, ne = args.nv, args.ne
    pc = torch.rand(b, n, 3, device=device) * 2 - 1
    nv_t = torch.full((b,), nv, dtype=torch.long)
    ne_t = torch.full((b,), ne, dtype=torch.long)
    vertices = torch.rand(b * nv, 3, device=device) * 2 - 1
    vptr = torch.arange(0, b * nv + 1, nv, dtype=torch.long)
    eptr = torch.arange(0, b * ne + 1, ne, dtype=torch.long)
    # random local endpoints per sample, offset into the packed vertex space.
    ei_parts = []
    for s in range(b):
        a = torch.randint(0, nv, (ne,))
        c = (a + torch.randint(1, nv, (ne,))) % nv  # != a
        ei_parts.append(torch.stack([a, c], dim=0) + s * nv)
    edge_index = torch.cat(ei_parts, dim=1).to(device)
    edge_points_norm = torch.rand(b * ne, u, 3, device=device) * 2 - 1
    return {
        "point_cloud": pc,
        "vertices": vertices,
        "vertex_ptr": vptr,
        "edge_index": edge_index,
        "edge_ptr": eptr,
        "edge_points_norm": edge_points_norm,
        "num_vertices": nv_t,
        "num_edges": ne_t,
        "num_graphs": b,
    }


def real_loader(args):
    from src.data import WireframeDataModule

    with open(args.data_config) as f:
        cfg = yaml.safe_load(f)
    init = dict(cfg["data"]["init_args"])
    import os
    for key in ("data_root", "split_path"):
        if key in init and not os.path.isabs(init[key]):
            init[key] = str(root / init[key])
    init.update(batch_size=args.batch_size, num_workers=4,
                persistent_workers=False, shuffle=True)
    dm = WireframeDataModule(**init)
    dm.setup("fit")
    return dm.train_dataloader()


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


# ----------------------------------------------------------------------
def build_module(args, device: torch.device) -> PC2WireframeModule:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    init = dict(cfg["model"]["init_args"])
    init["curve_vae_ckpt"] = None  # random-init curve VAE is fine for timing
    module = PC2WireframeModule(**init)
    return module.to(device).train()


def run(args) -> None:
    _install_probes()
    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)

    print(f"[device] {device}  precision={args.precision}")
    module = build_module(args, device)
    model = module.model
    enc = model.pc_encoder
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params: {n_params/1e6:.1f}M  "
          f"latent={enc.compressor.num_tokens}x{enc.compressor.latent_dim}  "
          f"node_q={model.decoder.num_node_queries} rln={model.decoder.num_rln_tokens}")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4)

    loader = None
    data_iter = None
    if args.data == "real":
        loader = real_loader(args)
        data_iter = iter(loader)

    amp_dtype = {"32": None, "bf16": torch.bfloat16, "16": torch.float16}[args.precision]

    def get_batch(stop):
        nonlocal data_iter
        if args.data == "synthetic":
            return synth_batch(args, device)
        with stop("data"):
            try:
                b = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                b = next(data_iter)
            b = to_device(b, device)
        return b

    def one_step(stop, timed: bool):
        batch = get_batch(stop)
        opt.zero_grad(set_to_none=True)
        ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
               if amp_dtype is not None else _nullctx())
        with ctx:
            with stop("enc.ptv3"):
                point = enc.backbone(enc._to_data_dict(batch["point_cloud"]))
            with stop("enc.group+comp"):
                tokens, mask = enc._group_by_batch(
                    point["feat"], point["batch"], batch["point_cloud"].shape[0])
                mu, logvar = enc.compressor(tokens, key_padding_mask=mask)
                z = enc.compressor.reparameterize(mu, logvar) if logvar is not None else mu
            with stop("decoder"):
                preds = model.decoder(z)
            with stop("build_targets"):
                from src.models.packing import build_targets
                targets = build_targets(batch)
            with stop("criterion"):
                total, parts = module.criterion(preds, targets, model)
                if logvar is not None:
                    total = total + module.hparams.kl_weight * (
                        -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
        if not args.no_backward:
            with stop("backward"):
                total.backward()
            with stop("optim"):
                opt.step()
        return float(total.detach()), int(parts["n_match_edges"])

    # warmup (build CUDA kernels / autotune)
    print(f"[warmup] {args.warmup} steps ...")
    warm = Stop(device)
    for _ in range(max(0, args.warmup)):
        one_step(warm, timed=False)
    SUBTIMES.clear()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"[run] {args.steps} timed steps ...")
    stop = Stop(device)
    loss_v = 0.0
    n_edges = 0
    t0 = time.perf_counter()
    for _ in range(args.steps):
        loss_v, n_edges = one_step(stop, timed=True)
    total_wall = time.perf_counter() - t0

    # ---- report ----
    s = max(1, args.steps)
    print("\n==== per-iteration time breakdown (mean over "
          f"{args.steps} steps) ====")
    order = sorted(stop.t.items(), key=lambda kv: -kv[1])
    measured = sum(stop.t.values())
    for name, tot in order:
        ms = tot / s * 1e3
        pct = 100 * tot / max(measured, 1e-9)
        print(f"  {name:16s} {ms:9.2f} ms/it   {pct:5.1f}%")
    print("  " + "-" * 44)
    for name in ("hungarian", "curve_decode"):
        if name in SUBTIMES:
            ms = SUBTIMES[name] / s * 1e3
            print(f"  (criterion.{name:11s}{ms:9.2f} ms/it)")
    print(f"\n  total measured : {measured/s*1e3:9.2f} ms/it")
    print(f"  full wall      : {total_wall/s*1e3:9.2f} ms/it  "
          f"({s/total_wall:.2f} it/s)")
    if device.type == "cuda":
        print(f"  peak GPU mem   : {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"\n  last loss={loss_v:.4f}  matched_edges/iter~{n_edges}")
    print("\nTip: --data real also times the dataloader; raise --nv/--ne or "
          "--batch-size to see how the Hungarian matching scales.")


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    run(parse_args())
