"""Evaluate / visualise the stage-1 curve VAE (``AutoencoderKL1D``) on val.

Loads a stage-1 checkpoint, runs the **validation protocol** of
``CurveVAEModule`` (posterior mode + uniform ``t``) over the held-out val
curves, reports reconstruction-error statistics and saves figures comparing the
decoded curves against the ground-truth canonical curves.

This is headless/server-friendly: it uses the ``Agg`` matplotlib backend and
*saves PNGs* (it never opens a window).

Everything happens in the curve VAE's canonical frame: GT curves have their
endpoints pinned to ``[-1,0,0]`` / ``[1,0,0]`` (see ``geometry.normalize_curves``),
so GT and decode are directly comparable without any denormalisation.

Example::

    python scripts/eval_curve_vae.py \
        --ckpt logs/pc2wireframe/f09xd826/checkpoints/epoch=044-val_loss=0.0150.ckpt \
        --data-config configs/data.yaml \
        --out-dir logs/pc2wireframe/f09xd826/curve_vae_eval
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
from src.models.packing import normalized_curves_from_batch  # noqa: E402
from src.models.vae.torch_tools import interpolate_1d  # noqa: E402
from src.module import CurveVAEModule  # noqa: E402


# ----------------------------------------------------------------------
def _default_ckpt() -> str:
    """Pick the best (lowest val_loss) checkpoint under the default log dir."""
    pat = str(root / "logs/pc2wireframe/*/checkpoints/epoch=*val_loss=*.ckpt")
    cands = glob.glob(pat)
    if not cands:
        raise SystemExit(
            f"No checkpoint found matching {pat!r}; pass --ckpt explicitly."
        )

    # filename encodes val_loss; lower is better.
    def _val_loss(p: str) -> float:
        try:
            return float(p.split("val_loss=")[-1].split(".ckpt")[0])
        except ValueError:
            return float("inf")

    return min(cands, key=_val_loss)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None, help="stage-1 ckpt (default: best under logs/)")
    ap.add_argument("--data-config", default=str(root / "configs/data.yaml"))
    ap.add_argument("--out-dir", default=None, help="default: <ckpt_dir>/../curve_vae_eval")
    ap.add_argument("--num-show", type=int, default=16, help="curves per figure grid")
    ap.add_argument("--max-curves", type=int, default=4000, help="cap curves used for stats")
    ap.add_argument("--decode-points", type=int, default=64, help="t-resolution for plotting")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


# ----------------------------------------------------------------------
def build_val_loader(data_config: str, num_workers: int, batch_size: int):
    with open(data_config) as f:
        cfg = yaml.safe_load(f)
    init_args = dict(cfg["data"]["init_args"])
    # data_root in the config is relative to the project root.
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


@torch.no_grad()
def run_eval(model: CurveVAEModule, loader, device: str, max_curves: int, decode_points: int):
    """Validation-protocol reconstruction over val curves.

    Returns dict of numpy arrays:
        gt        (N, U_in, 3)  ground-truth canonical curve polylines
        dec       (N, P, 3)     decoded curves (P = decode_points)
        mse       (N,)          per-curve raw MSE at the U-point val grid
        lat_std   (N,)          per-curve mean posterior std
    """
    vae = model.curve_vae
    U = int(vae.sample_points_num)

    gt_list, dec_list, mse_list, std_list = [], [], [], []
    seen = 0
    for batch in loader:
        curves = normalized_curves_from_batch(batch)  # (E, U_in, 3)
        if curves is None or curves.shape[0] == 0:
            continue
        curves = curves.to(device)
        data = rearrange(curves, "b n c -> b c n")  # (E, 3, U_in)

        posterior = vae.encode(data)
        z = posterior.mode()

        # --- error at the val grid (uniform t, U points) ---
        t_u = torch.linspace(0.0, 1.0, U, device=device).unsqueeze(0).expand(z.shape[0], -1)
        dec_u = vae.decode(z, t_u)          # (E, 3, U)
        gt_u = interpolate_1d(t_u, data)    # (E, 3, U)
        per_curve_mse = (dec_u - gt_u).pow(2).mean(dim=[1, 2])  # (E,)

        # --- high-res decode for plotting ---
        t_p = torch.linspace(0.0, 1.0, decode_points, device=device)
        t_p = t_p.unsqueeze(0).expand(z.shape[0], -1)
        dec_p = vae.decode(z, t_p)          # (E, 3, P)

        gt_list.append(curves.cpu().numpy())                       # (E, U_in, 3)
        dec_list.append(rearrange(dec_p, "b c p -> b p c").cpu().numpy())
        mse_list.append(per_curve_mse.cpu().numpy())
        std_list.append(posterior.std.mean(dim=[1, 2]).cpu().numpy())

        seen += curves.shape[0]
        if seen >= max_curves:
            break

    gt = np.concatenate(gt_list, axis=0)[:max_curves]
    dec = np.concatenate(dec_list, axis=0)[:max_curves]
    mse = np.concatenate(mse_list, axis=0)[:max_curves]
    lat_std = np.concatenate(std_list, axis=0)[:max_curves]
    return dict(gt=gt, dec=dec, mse=mse, lat_std=lat_std)


# ----------------------------------------------------------------------
def _plot_curve_cell(ax, gt: np.ndarray, dec: np.ndarray, title: str):
    """3-D overlay of one GT curve (blue) vs decoded curve (red dashed)."""
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], "-", color="#1f77b4", lw=1.6, label="GT")
    ax.plot(dec[:, 0], dec[:, 1], dec[:, 2], "--", color="#d62728", lw=1.6, label="decode")
    # pinned endpoints
    ax.scatter(gt[[0, -1], 0], gt[[0, -1], 1], gt[[0, -1], 2],
               color="k", s=12, depthshade=False)

    pts = np.concatenate([gt, dec], axis=0)
    c = pts.mean(0)
    r = max(float(np.abs(pts - c).max()), 1e-3)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=5)
    ax.view_init(elev=22, azim=-60)


def save_grid(res: dict, idx: np.ndarray, out_path: str, suptitle: str):
    n = len(idx)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(cols * 3.0, rows * 3.0))
    for i, j in enumerate(idx):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        _plot_curve_cell(
            ax, res["gt"][j], res["dec"][j],
            f"#{int(j)}  mse={res['mse'][j]:.4f}",
        )
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[saved] {out_path}")


def save_hist(mse: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.sqrt(mse), bins=60, color="#4c72b0", alpha=0.85)
    ax.axvline(float(np.sqrt(mse).mean()), color="r", ls="--",
               label=f"mean RMSE={np.sqrt(mse).mean():.4f}")
    ax.axvline(float(np.median(np.sqrt(mse))), color="g", ls="--",
               label=f"median RMSE={np.median(np.sqrt(mse)):.4f}")
    ax.set_xlabel("per-curve RMSE (canonical frame, span=2)")
    ax.set_ylabel("count")
    ax.set_title("Curve VAE reconstruction error on val")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[saved] {out_path}")


# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt = args.ckpt or _default_ckpt()
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.dirname(ckpt)), "curve_vae_eval")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[ckpt] {ckpt}")
    print(f"[out ] {out_dir}")

    model = CurveVAEModule.load_from_checkpoint(ckpt, map_location=args.device)
    model.eval().to(args.device)

    loader = build_val_loader(args.data_config, args.num_workers, args.batch_size)
    res = run_eval(model, loader, args.device, args.max_curves, args.decode_points)

    mse = res["mse"]
    rmse = np.sqrt(mse)
    print("\n==== curve VAE reconstruction on val (canonical frame, endpoint span = 2.0) ====")
    print(f"  curves evaluated : {len(mse)}")
    print(f"  MSE   mean/median: {mse.mean():.5f} / {np.median(mse):.5f}")
    print(f"  RMSE  mean/median: {rmse.mean():.5f} / {np.median(rmse):.5f}")
    print(f"  RMSE  p90 / p99  : {np.percentile(rmse, 90):.5f} / {np.percentile(rmse, 99):.5f}")
    print(f"  RMSE  max        : {rmse.max():.5f}")
    print(f"  latent std mean  : {res['lat_std'].mean():.5f}")
    print(f"  (RMSE / span)    : {rmse.mean() / 2.0 * 100:.2f}% mean, "
          f"{np.median(rmse) / 2.0 * 100:.2f}% median\n")

    order = np.argsort(mse)
    n = min(args.num_show, len(mse))

    # representative spread: evenly spaced percentiles of the error distribution
    spread = order[np.linspace(0, len(order) - 1, n).astype(int)]
    save_grid(res, spread, os.path.join(out_dir, "recon_spread.png"),
              "Curve VAE: GT (blue) vs decode (red) -- best->worst spread")

    # worst cases
    worst = order[-n:][::-1]
    save_grid(res, worst, os.path.join(out_dir, "recon_worst.png"),
              "Curve VAE: GT (blue) vs decode (red) -- worst cases")

    save_hist(mse, os.path.join(out_dir, "recon_rmse_hist.png"))


if __name__ == "__main__":
    main()
