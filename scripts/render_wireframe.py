#!/usr/bin/env python3
"""Render GT wireframes as nice gradient images. Two backends, pick with ``--backend``.

* ``matplotlib`` (default): every GT edge polyline is drawn as a thick, round-
  capped, height-gradient stroke on a white background. Flat-but-pretty, needs
  only ``numpy`` + ``matplotlib`` (Agg, pure CPU). No GL/X/VTK/root -- runs
  anywhere, instantly.
* ``mitsuba``: every edge becomes a chain of real 3D **tubes** (cylinders +
  joint spheres), colored by a height gradient and rendered with Mitsuba 3
  (``scalar_rgb``, pure CPU -- no GL/X needed) for a **pure-white background
  with soft contact shadows** -- the physically-shaded look. Slower, needs
  ``pip install mitsuba``.

It reads the GT **edge npz** files directly (the same
``data/train/sample_edge/*.npz`` the dataloader uses).

NPZ schema it expects (aliases handled):
    start_verts  (E, 3)        edge start points
    end_verts    (E, 3)        edge end points
    edge_points  (E, U, 3)     per-edge polyline samples   <- this is what we draw

Usage (from the project root)
-----------------------------
Fast matplotlib preview, 6 random shapes -> logs/render/::

    python scripts/render_wireframe.py --num 6

Mitsuba (real tubes + soft shadows), montage of 9::

    python scripts/render_wireframe.py --backend mitsuba --num 9 --seed 0 --montage

Mitsuba quality / look knobs::

    python scripts/render_wireframe.py --backend mitsuba --num 4 \
        --radius 0.01 --spp 256 --cmap coolwarm --color-axis z

Install
-------
    pip install numpy matplotlib    # matplotlib backend (already present)
    pip install mitsuba             # mitsuba backend (CPU, no GL needed)
"""
from __future__ import annotations

import argparse
import glob
import os
import random
from typing import Any

import numpy as np


# ----------------------------------------------------------------------
# IO: read GT edge polylines straight from the npz (no torch)
# ----------------------------------------------------------------------
_EDGE_POINT_KEYS = ("edge_points", "curve_points")


def _load_npz_dict(path: str) -> dict[str, Any]:
    """Load an npz into a dict, falling back to pickle for object arrays."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except ValueError as e:
        if "pickle" not in str(e).lower() and "object" not in str(e).lower():
            raise
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}


def _get(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in data:
            return data[k]
    raise KeyError(f"none of {keys} found in npz (have {list(data)})")


def load_polylines(path: str) -> list[np.ndarray]:
    """Return the GT edges as a list of ``(Ni, 3)`` polylines.

    Handles both a regular ``(E, U, 3)`` float array and an object array of
    variable-length polylines.
    """
    data = _load_npz_dict(path)
    ep = _get(data, _EDGE_POINT_KEYS)
    polys: list[np.ndarray] = []
    if isinstance(ep, np.ndarray) and ep.dtype != object and ep.ndim == 3:
        for i in range(ep.shape[0]):
            p = np.asarray(ep[i], dtype=np.float64).reshape(-1, 3)
            if p.shape[0] >= 2:
                polys.append(p)
    else:  # object array / ragged
        for i in range(len(ep)):
            p = np.asarray(ep[i], dtype=np.float64).reshape(-1, 3)
            if p.shape[0] >= 2:
                polys.append(p)
    return polys


def _edge_count(path: str) -> int:
    """Cheap edge count for the ``worst`` pick (avoids unpickling polylines)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            for k in ("start_verts", "start_vertices", "sv"):
                if k in z.files:
                    return int(z[k].shape[0])
            for k in _EDGE_POINT_KEYS:
                if k in z.files:
                    return int(z[k].shape[0])
    except Exception:
        return 0
    return 0


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------
def list_npz(directory: str) -> list[str]:
    return sorted(glob.glob(os.path.join(directory, "*.npz")))


def select_files(
    files: list[str], num: int, pick: str, seed: int
) -> list[str]:
    num = min(num, len(files))
    if pick == "first":
        return files[:num]
    if pick == "worst":
        pool = files
        if len(files) > 2000:  # don't stat the whole corpus
            random.seed(seed)
            pool = random.sample(files, 2000)
        scored = sorted(((_edge_count(f), f) for f in pool), reverse=True)
        return [f for _, f in scored[:num]]
    rng = random.Random(seed)
    return rng.sample(files, num)


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
def normalize_to_unit_cube(polys: list[np.ndarray]) -> list[np.ndarray]:
    """Center on the bbox center and scale the largest extent to [-1, 1]."""
    cat = np.concatenate(polys, axis=0)
    lo, hi = cat.min(0), cat.max(0)
    center = (lo + hi) * 0.5
    scale = float((hi - lo).max()) * 0.5 or 1.0
    return [(p - center) / scale for p in polys]


# ----------------------------------------------------------------------
# Rendering (matplotlib 3d, Agg backend)
# ----------------------------------------------------------------------
_AXIS = {"x": 0, "y": 1, "z": 2}


# Built-in low-contrast palettes (analogous hues + similar lightness => calm).
_CUSTOM_CMAPS = {
    # cool periwinkle -> teal -> mint: gentle, pretty, low contrast
    "soft": ["#6b8fd6", "#56b6c2", "#86d6a8"],
    # warm sand -> coral -> dusty rose: muted warm variant
    "warmsoft": ["#e8c98a", "#e09b7d", "#cf8aa6"],
}


def _get_cmap(name: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if name in _CUSTOM_CMAPS:
        return LinearSegmentedColormap.from_list(name, _CUSTOM_CMAPS[name])
    return plt.get_cmap(name)


def _gradient_rgb(t, args) -> np.ndarray:
    """Map normalized scalars ``t in [0,1]`` to softened sRGB colors.

    ``--color-lo/--color-hi`` clip to a sub-range of the colormap (skip the
    over-saturated ends -> lower contrast) and ``--sat`` blends toward luminance
    gray (lower saturation -> calmer colors). Works for scalars or arrays.
    """
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    tt = args.color_lo + t * (args.color_hi - args.color_lo)
    rgb = np.asarray(_get_cmap(args.cmap)(tt))[..., :3]
    lum = np.tensordot(rgb, np.array([0.2126, 0.7152, 0.0722]), axes=([-1], [0]))
    return lum[..., None] + args.sat * (rgb - lum[..., None])


def _segments_and_values(polys: list[np.ndarray], axis: int):
    """Flatten polylines into per-segment endpoints + per-segment color value."""
    seg_list, val_list = [], []
    for p in polys:
        seg = np.stack([p[:-1], p[1:]], axis=1)          # (M, 2, 3)
        mid = (p[:-1] + p[1:]) * 0.5                      # (M, 3)
        seg_list.append(seg)
        val_list.append(mid[:, axis])
    if not seg_list:
        return np.zeros((0, 2, 3)), np.zeros((0,))
    return np.concatenate(seg_list, 0), np.concatenate(val_list, 0)


def _equal_aspect(ax, pts: np.ndarray) -> None:
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) * 0.5
    r = float((hi - lo).max()) * 0.5 or 1.0
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _draw_on_ax(ax, polys: list[np.ndarray], args, norm=None) -> None:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    axis = _AXIS[args.color_axis]
    cat = np.concatenate(polys, axis=0)
    segs, vals = _segments_and_values(polys, axis)
    vmin, vmax = float(cat[:, axis].min()), float(cat[:, axis].max())
    t = (vals - vmin) / (vmax - vmin + 1e-12)
    colors = _gradient_rgb(t, args)

    lc = Line3DCollection(
        segs, colors=colors,
        linewidths=args.linewidth, capstyle="round", joinstyle="round",
    )
    ax.add_collection3d(lc)

    _equal_aspect(ax, cat)
    ax.view_init(elev=args.elevation, azim=args.azimuth)
    ax.set_axis_off()
    # transparent panes so only the strokes show on the figure background
    ax.patch.set_alpha(0.0)


def render_single(polys: list[np.ndarray], out_path: str, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.normalize:
        polys = normalize_to_unit_cube(polys)

    dpi = 100
    fig = plt.figure(figsize=(args.res / dpi, args.res / dpi), dpi=dpi)
    fig.patch.set_facecolor(args.bg)
    ax = fig.add_subplot(111, projection="3d")
    _draw_on_ax(ax, polys, args)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, dpi=dpi, facecolor=args.bg)
    plt.close(fig)


def render_montage(items: list[tuple[str, list[np.ndarray]]], out_path: str,
                   args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(items)
    cols = args.cols or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    dpi = 100
    fig = plt.figure(figsize=(args.res * cols / dpi, args.res * rows / dpi),
                     dpi=dpi)
    fig.patch.set_facecolor(args.bg)
    for i, (name, polys) in enumerate(items):
        if args.normalize:
            polys = normalize_to_unit_cube(polys)
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        _draw_on_ax(ax, polys, args)
        ax.set_title(name[:24], fontsize=8, color="gray")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.96,
                        wspace=0.02, hspace=0.06)
    fig.savefig(out_path, dpi=dpi, facecolor=args.bg)
    plt.close(fig)


# ----------------------------------------------------------------------
# Rendering (Mitsuba 3, scalar_rgb -- pure CPU, no GL/X needed)
#
# Each edge polyline is turned into a chain of `cylinder` primitives (round
# tubes) with a small `sphere` at every joint, colored by a height gradient.
# A constant white environment gives a pure-white background, and a big
# overhead area light over a faint ground plane produces the soft contact
# shadow.
# ----------------------------------------------------------------------
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _resample_to(p: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to at most ``n`` points by arc length (n<=0: keep)."""
    if n <= 0 or p.shape[0] <= n:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-12:
        return p[:1]
    dst = np.linspace(0.0, total, n)
    return np.stack([np.interp(dst, cum, p[:, c]) for c in range(3)], axis=-1)


def _mitsuba_scene_dict(polys: list[np.ndarray], args) -> dict:
    axis = _AXIS[args.color_axis]
    cat = np.concatenate(polys, axis=0)
    center = (cat.min(0) + cat.max(0)) * 0.5
    bs = float(np.linalg.norm(cat - center, axis=1).max()) or 1.0
    vmin = float(cat[:, axis].min())
    vmax = float(cat[:, axis].max())
    zmin = float(cat[:, 2].min())
    zmax = float(cat[:, 2].max())

    # camera (z-up); azimuth/elevation in degrees, distance from bounding sphere
    az, el = np.radians(args.azimuth), np.radians(args.elevation)
    direction = np.array([np.cos(el) * np.cos(az),
                          np.cos(el) * np.sin(az),
                          np.sin(el)])
    origin = center + direction * bs * args.dist_scale

    import mitsuba as mi
    T = mi.ScalarTransform4f

    levels = max(2, args.color_levels)

    def color_id(val: float) -> str:
        t = (val - vmin) / (vmax - vmin + 1e-12)
        return f"c{int(np.clip(t * levels, 0, levels - 1))}"

    scene: dict[str, Any] = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 8},
        "sensor": {
            "type": "perspective",
            "fov": args.fov,
            "to_world": T.look_at(origin=origin.tolist(),
                                  target=center.tolist(), up=[0, 0, 1]),
            "sampler": {"type": "independent", "sample_count": args.spp},
            "film": {
                "type": "hdrfilm", "width": args.res, "height": args.res,
                "pixel_format": "rgb", "rfilter": {"type": "gaussian"},
            },
        },
        # pure-white background + soft ambient fill
        "env": {"type": "constant",
                "radiance": {"type": "rgb", "value": args.ambient}},
        # overhead key light (large => soft shadow), faces down (-z)
        "key": {
            "type": "rectangle",
            "to_world": (T.translate([center[0], center[1], zmax + bs * 2.0])
                         @ T.rotate(axis=[1, 0, 0], angle=180)
                         @ T.scale([bs * 2.5, bs * 2.5, bs * 2.5])),
            "emitter": {"type": "area",
                        "radiance": {"type": "rgb", "value": args.light}},
        },
    }

    # palette of reusable diffuse BSDFs (softened sRGB colormap -> linear)
    for i in range(levels):
        rgb = _gradient_rgb((i + 0.5) / levels, args)
        scene[f"c{i}"] = {
            "type": "diffuse",
            "reflectance": {"type": "rgb",
                            "value": _srgb_to_linear(rgb).tolist()},
        }

    if not args.no_ground:
        gs = bs * 12.0
        scene["ground"] = {
            "type": "rectangle",
            "to_world": (T.translate([center[0], center[1], zmin - args.radius])
                         @ T.scale([gs, gs, gs])),
            "bsdf": {"type": "diffuse",
                     "reflectance": {"type": "rgb", "value": args.ground_albedo}},
        }

    # tubes + joint spheres
    k = 0
    r = args.radius
    for p in polys:
        p = _resample_to(np.asarray(p, dtype=np.float64), args.curve_pts)
        vals = p[:, axis]
        for i in range(p.shape[0] - 1):
            if np.linalg.norm(p[i + 1] - p[i]) < 1e-9:
                continue
            mid = 0.5 * (vals[i] + vals[i + 1])
            scene[f"cyl{k}"] = {
                "type": "cylinder",
                "p0": p[i].tolist(), "p1": p[i + 1].tolist(), "radius": r,
                "bsdf": {"type": "ref", "id": color_id(mid)},
            }
            k += 1
        if not args.no_joints:
            for i in range(p.shape[0]):
                scene[f"sph{k}"] = {
                    "type": "sphere",
                    "center": p[i].tolist(), "radius": r,
                    "bsdf": {"type": "ref", "id": color_id(vals[i])},
                }
                k += 1
    return scene


def _mitsuba_render_array(polys: list[np.ndarray], args) -> np.ndarray:
    """Render one shape and return an (H, W, 3) uint8 sRGB image."""
    import mitsuba as mi

    if args.normalize:
        polys = normalize_to_unit_cube(polys)
    scene = mi.load_dict(_mitsuba_scene_dict(polys, args))
    img = mi.render(scene, spp=args.spp)
    bmp = mi.Bitmap(img).convert(
        mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)
    return np.array(bmp)


def _save_png(arr: np.ndarray, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(path, arr)


def render_single_mitsuba(polys: list[np.ndarray], out_path: str, args) -> None:
    _save_png(_mitsuba_render_array(polys, args), out_path)


def render_montage_mitsuba(items: list[tuple[str, list[np.ndarray]]],
                           out_path: str, args) -> None:
    n = len(items)
    cols = args.cols or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    res = args.res
    canvas = np.full((rows * res, cols * res, 3), 255, dtype=np.uint8)
    for i, (name, polys) in enumerate(items):
        arr = _mitsuba_render_array(polys, args)
        rr, cc = (i // cols) * res, (i % cols) * res
        canvas[rr:rr + res, cc:cc + res] = arr[:res, :res]
        print(f"  rendered {name}")
    _save_png(canvas, out_path)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # data location (mirrors configs/data.yaml)
    p.add_argument("--data-root", default="data")
    p.add_argument("--edge-subdir", default="train/sample_edge")
    p.add_argument("--in-dir", default=None,
                   help="override the edge dir directly (else data-root/edge-subdir)")
    # selection
    p.add_argument("--num", type=int, default=6, help="number of shapes")
    p.add_argument("--pick", default="random",
                   choices=["random", "first", "worst"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--files", nargs="*", default=None,
                   help="explicit npz paths (overrides --pick/--num)")
    # backend
    p.add_argument("--backend", default="matplotlib",
                   choices=["matplotlib", "mitsuba"],
                   help="matplotlib = flat gradient strokes (zero deps); "
                        "mitsuba = real tubes + soft shadows on white (CPU)")
    # shared render knobs
    p.add_argument("--cmap", default="warmsoft",
                   help="palette: built-in low-contrast 'soft'/'warmsoft', or any "
                        "matplotlib colormap (Spectral/viridis/plasma/coolwarm/...)")
    p.add_argument("--color-axis", default="z", choices=["x", "y", "z"],
                   help="axis whose value drives the gradient")
    p.add_argument("--sat", type=float, default=0.9,
                   help="color saturation (1=full colormap, lower=calmer/grayer)")
    p.add_argument("--color-lo", type=float, default=0.0,
                   help="use the colormap only from this fraction (skip bright end)")
    p.add_argument("--color-hi", type=float, default=1.0,
                   help="...up to this fraction (lower contrast between extremes)")
    p.add_argument("--res", type=int, default=1024, help="per-shape px (square)")
    p.add_argument("--azimuth", type=float, default=-60.0)
    p.add_argument("--elevation", type=float, default=18.0)
    p.add_argument("--no-normalize", dest="normalize", action="store_false",
                   help="do NOT rescale each shape to the unit cube")
    # matplotlib-only
    p.add_argument("--linewidth", type=float, default=2.5,
                   help="[matplotlib] stroke width (fake tube thickness)")
    p.add_argument("--bg", default="white",
                   help="[matplotlib] background color")
    # mitsuba-only
    p.add_argument("--radius", type=float, default=0.008,
                   help="[mitsuba] tube radius (normalized units)")
    p.add_argument("--spp", type=int, default=128,
                   help="[mitsuba] samples per pixel (higher = less noise)")
    p.add_argument("--fov", type=float, default=33.0, help="[mitsuba] camera fov")
    p.add_argument("--dist-scale", type=float, default=2.8,
                   help="[mitsuba] camera distance (smaller = zoom in)")
    p.add_argument("--ambient", type=float, default=1.0,
                   help="[mitsuba] constant env radiance (1.0 = pure white bg)")
    p.add_argument("--light", type=float, default=2.0,
                   help="[mitsuba] overhead key-light radiance")
    p.add_argument("--ground-albedo", type=float, default=0.85,
                   help="[mitsuba] ground plane brightness (shadow contrast)")
    p.add_argument("--no-ground", action="store_true",
                   help="[mitsuba] drop the ground plane (no contact shadow)")
    p.add_argument("--no-joints", action="store_true",
                   help="[mitsuba] skip joint spheres between tube segments")
    p.add_argument("--color-levels", type=int, default=256,
                   help="[mitsuba] number of quantized gradient colors")
    p.add_argument("--curve-pts", type=int, default=0,
                   help="[mitsuba] resample each edge to <=N pts (0 = keep all)")
    p.add_argument("--mts-variant", default="scalar_rgb",
                   help="[mitsuba] variant (scalar_rgb / llvm_ad_rgb / ...)")
    # output / mode
    p.add_argument("--out-dir", default="logs/render")
    p.add_argument("--montage", action="store_true",
                   help="one combined grid image instead of per-shape PNGs")
    p.add_argument("--cols", type=int, default=0,
                   help="montage columns (0 = auto ~sqrt)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.backend == "mitsuba":
        import mitsuba as mi
        mi.set_variant(args.mts_variant)

    in_dir = args.in_dir or os.path.join(args.data_root, args.edge_subdir)
    if args.files:
        files = list(args.files)
    else:
        all_files = list_npz(in_dir)
        if not all_files:
            raise SystemExit(f"No .npz found in {in_dir!r}")
        files = select_files(all_files, args.num, args.pick, args.seed)

    items: list[tuple[str, list[np.ndarray]]] = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            polys = load_polylines(f)
        except Exception as exc:
            print(f"[warn] failed to read {name}: {exc}")
            continue
        if not polys:
            print(f"[warn] {name}: no drawable edges, skipped")
            continue
        items.append((name, polys))
        print(f"{name:<44} {len(polys):>5} edges")

    if not items:
        raise SystemExit("Nothing to render.")

    montage_fn = (render_montage_mitsuba if args.backend == "mitsuba"
                  else render_montage)
    single_fn = (render_single_mitsuba if args.backend == "mitsuba"
                 else render_single)

    if args.montage:
        out = args.out_dir
        if os.path.isdir(out) or not out.lower().endswith(".png"):
            out = os.path.join(out, "montage.png")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        montage_fn(items, out, args)
        print(f"\nSaved montage -> {out}")
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        for name, polys in items:
            out = os.path.join(args.out_dir, f"{name}.png")
            single_fn(polys, out, args)
            print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
