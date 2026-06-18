"""Wireframe datasets for the Rectified-Flow PC2Wireframe branch.

Two dataset flavours are provided:

``WireframeGraphDataset``
    Used for training / validation. Loads a per-edge GT wireframe NPZ
    (endpoints + resampled curve points) together with its matching surface
    point cloud, then derives the Rectified-Flow **anchor point set**
    ``wf_points (N, 3) = (x, y, z)`` by **global arc-length sampling** over all
    edge polylines (denser / longer curves receive proportionally more points;
    no vertex points, no type channel).

    The fixed-size ``point_cloud`` and ``wf_points`` are stacked by the default
    collate (:func:`collate_rf_batch`); the *native-size* GT graph
    (``vertices`` / ``edge_index`` / ``edge_points``) is carried alongside as a
    Python list under ``gt_wireframes`` so the validation metrics can score
    against the real wireframe.

    Each sample (``__getitem__``) yields::

        shape_id:     str
        point_cloud:  (Ni, 3)              float32   (native, variable size)
        pc_center:    (3,)                 float32
        pc_scale:     ()                   float32
        wf_points:    (wf_num_points, 3)   float32   (fixed size, RF target)
        vertices:     (Vi, 3)              float32   (native GT)
        edge_index:   (Ei, 2)              int64     (LOCAL vertex ids)
        edge_points:  (Ei, U, 3)           float32

``PointCloudDataset``
    Used for prediction / submission. The test split ships only point clouds
    (no ground-truth edges), so this dataset loads ``surface_points`` and
    returns ``{shape_id, point_cloud, pc_center, pc_scale}`` for inference.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

_log = logging.getLogger(__name__)

# Coordinates beyond this magnitude are treated as corrupt and dropped.
_PC_COORD_CLIP = 1e4


def _unit_cube_transform(
    points: np.ndarray, margin: float = 0.95
) -> tuple[np.ndarray, float]:
    """Per-shape normalization transform to a unit cube.

    Returns ``(center (3,), scale)`` so that ``(x - center) / scale`` maps the
    point cloud's bounding box into ``[-margin, margin]`` (the longest axis fills
    it, the others are smaller). The *same* transform is applied to the point
    cloud and the wireframe vertices / curves so the model is trained and
    supervised in one normalized frame -- this is what makes the PTv3 grid
    (``grid_size`` relative to a ~unit extent) well-conditioned. Raw CAD
    coordinates can span thousands of units, which overflows PTv3's
    space-filling-curve depth.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    lo = pts.min(0)
    hi = pts.max(0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    half = float((hi - lo).max()) * 0.5
    scale = max(half / max(margin, 1e-6), 1e-6)
    return center, float(scale)


def _apply_unit_cube(
    points: np.ndarray, center: np.ndarray, scale: float
) -> np.ndarray:
    """Apply a unit-cube transform to a ``(..., 3)`` array."""
    return ((points - center) / scale).astype(np.float32)


# ----------------------------------------------------------------------
# NPZ helpers
# ----------------------------------------------------------------------
def _load_npz_dict(path: str) -> dict[str, Any]:
    """Load an npz file into a plain dict, falling back to pickle if needed."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except ValueError as e:
        msg = str(e).lower()
        if "pickle" not in msg and "object" not in msg:
            raise
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}


def _get_npz_array(data: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray:
    for key in keys:
        if key in data:
            return np.asarray(data[key])
    raise KeyError(f"None of keys {keys!r} found in npz")


def _resample_polyline(points: np.ndarray, num_points: int) -> np.ndarray:
    """Resample an ordered polyline to exactly ``num_points`` 3D points."""
    points = np.asarray(points, dtype=np.float64)
    points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    points = points[:, :3]
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    if points.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    points = np.clip(points, -_PC_COORD_CLIP, _PC_COORD_CLIP)
    if points.shape[0] == num_points:
        return points.astype(np.float32)
    if points.shape[0] == 1:
        return np.repeat(points, num_points, axis=0).astype(np.float32)
    src = np.linspace(0.0, 1.0, points.shape[0], dtype=np.float64)
    dst = np.linspace(0.0, 1.0, num_points, dtype=np.float64)
    out = np.stack(
        [np.interp(dst, src, points[:, c]) for c in range(3)], axis=-1)
    return out.astype(np.float32)


def _clean_point_cloud(points: np.ndarray, max_points: int = 0) -> np.ndarray:
    """Clean a *variable-size* point cloud (no resampling to a fixed count).

    Drops NaN/Inf and out-of-range outlier coordinates and keeps the surface
    points at their native count -- PTv3 consumes a variable-length point cloud
    via the packed ``coord``/``offset`` format, so there is no need to subsample
    to a fixed size (which would throw away geometric detail). When
    ``max_points > 0`` and the cloud is larger, it is randomly subsampled (no
    replacement) only as a memory safety cap. Returns ``(M, 3)`` float32.
    """
    points = np.asarray(points, dtype=np.float64)
    points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    points = points[:, :3]
    finite_mask = (
        np.isfinite(points).all(axis=1)
        & (np.abs(points).max(axis=1) < _PC_COORD_CLIP)
    )
    points = points[finite_mask]
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    points = np.clip(points, -_PC_COORD_CLIP, _PC_COORD_CLIP).astype(np.float32)
    if max_points and points.shape[0] > int(max_points):
        idx = np.random.choice(points.shape[0], size=int(max_points), replace=False)
        points = points[idx]
    return points


def _sample_arclength(edge_points: np.ndarray, num: int) -> np.ndarray:
    """Sample ``num`` points by **global arc length** over all edge polylines.

    ``edge_points`` is ``(E, U, 3)``. All consecutive-point segments across all
    edges are pooled into one polyline parameterised by cumulative arc length,
    then ``num`` points are drawn uniformly in that arc-length space (so denser
    / longer curves receive proportionally more points). Returns ``(num, 3)``.
    """
    eps = 1e-12
    if num <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.asarray(edge_points, dtype=np.float64).reshape(-1, edge_points.shape[1], 3)
    e, u = pts.shape[0], pts.shape[1]
    if e == 0 or u == 0:
        return np.zeros((num, 3), dtype=np.float32)
    if u == 1:
        flat = pts.reshape(-1, 3)
        idx = np.random.choice(
            flat.shape[0], num, replace=flat.shape[0] < num)
        return flat[idx].astype(np.float32)

    seg_start = pts[:, :-1, :].reshape(-1, 3)
    seg_end = pts[:, 1:, :].reshape(-1, 3)
    seg_vec = seg_end - seg_start
    seg_len = np.linalg.norm(seg_vec, axis=1)
    total = float(seg_len.sum())
    if total <= eps:
        flat = pts.reshape(-1, 3)
        idx = np.random.choice(
            flat.shape[0], num, replace=flat.shape[0] < num)
        return flat[idx].astype(np.float32)

    cum = np.cumsum(seg_len)
    cum_prev = cum - seg_len
    u_pos = np.random.uniform(0.0, total, size=num)
    seg_idx = np.clip(
        np.searchsorted(cum, u_pos, side="right"), 0, seg_len.shape[0] - 1)
    local = (u_pos - cum_prev[seg_idx]) / np.maximum(seg_len[seg_idx], eps)
    sampled = seg_start[seg_idx] + local[:, None] * seg_vec[seg_idx]
    return sampled.astype(np.float32)


def _build_wf_target(edge_points: np.ndarray, num_points: int) -> np.ndarray:
    """Build the fixed-size RF anchor point set ``(num_points, 3)``.

    The stage-1 target is a pure xyz anchor cloud sampled by **global
    arc length** over all edge polylines (no vertex points, no type channel).
    """
    return _sample_arclength(edge_points, int(num_points))


# Curve-type fit thresholds (max residual relative to the edge arc length).
# Below ``_CURVE_LINE_THRESH`` the polyline is a straight line; otherwise below
# ``_CURVE_ARC_THRESH`` (after a plane + least-squares-circle fit) it is an arc;
# anything else falls back to a cubic Bezier.
_CURVE_LINE_THRESH = 0.02
_CURVE_ARC_THRESH = 0.02

# Curve-type codes (kept in sync with the grouper's curve_type head + decoder).
_CURVE_LINE = 0
_CURVE_ARC = 1
_CURVE_BEZIER = 2


def _fit_curve_type(
    polyline: np.ndarray,
    *,
    line_thresh: float = _CURVE_LINE_THRESH,
    arc_thresh: float = _CURVE_ARC_THRESH,
) -> int:
    """Classify a GT polyline as ``0=line`` / ``1=arc`` / ``2=bezier``.

    The decision is residual-based and scale-invariant (every residual is
    measured relative to the polyline's arc length):

      * **line**: small max perpendicular residual to the PCA principal axis;
      * **arc**: otherwise, project onto the best-fit plane (the two dominant
        PCA axes), fit a circle by least squares, and accept if the combined
        in-plane radial + out-of-plane residual is small;
      * **bezier**: everything else (the catch-all parameterisation).
    """
    pts = np.asarray(polyline, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 3:
        return _CURVE_LINE
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arclen = float(seg_len.sum())
    if arclen <= 1e-12:
        return _CURVE_LINE

    centered = pts - pts.mean(axis=0)
    # PCA via SVD: rows of vt are principal axes (descending variance).
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    if vt.shape[0] < 3:
        return _CURVE_LINE
    axis0, axis1, normal = vt[0], vt[1], vt[2]

    # Line test: residual orthogonal to the principal axis.
    perp = centered - np.outer(centered @ axis0, axis0)
    line_res = float(np.sqrt((perp ** 2).sum(axis=1)).max())
    if line_res / arclen < line_thresh:
        return _CURVE_LINE

    # Arc test: least-squares circle in the (axis0, axis1) plane.
    x = centered @ axis0
    y = centered @ axis1
    z = centered @ normal
    a_mat = np.stack([x, y, np.ones_like(x)], axis=1)
    b_vec = -(x ** 2 + y ** 2)
    try:
        sol, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    except np.linalg.LinAlgError:
        return _CURVE_BEZIER
    cx, cy = -0.5 * sol[0], -0.5 * sol[1]
    r2 = cx ** 2 + cy ** 2 - sol[2]
    if r2 <= 0.0:
        return _CURVE_BEZIER
    radius = np.sqrt(r2)
    radial = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    arc_res = float(np.sqrt((radial - radius) ** 2 + z ** 2).max())
    if arc_res / arclen < arc_thresh:
        return _CURVE_ARC
    return _CURVE_BEZIER


def _points_on_edges(
    pts: np.ndarray,
    seg_vec: np.ndarray,
    seg_len: np.ndarray,
    edge_len: np.ndarray,
    seg_cum_prev: np.ndarray,
    eid: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Interpolate points at normalised arc-length ``t`` on edges ``eid``.

    ``pts`` is ``(E, U, 3)``; ``eid (M,)`` indexes edges and ``t (M,)`` is the
    per-point arc-length fraction in ``[0, 1]``. Returns ``(M, 3)`` coords.
    """
    eps = 1e-12
    if eid.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    target = t * edge_len[eid]                       # (M,) distance along edge
    cum_end = seg_cum_prev[eid] + seg_len[eid]       # (M, U-1)
    # number of fully-passed segments = index of the hosting segment
    within = np.clip(
        (target[:, None] >= cum_end).sum(axis=1), 0, seg_len.shape[1] - 1)
    seg_l = seg_len[eid, within]
    local = (target - seg_cum_prev[eid, within]) / np.maximum(seg_l, eps)
    local = np.clip(local, 0.0, 1.0)
    return pts[eid, within] + local[:, None] * seg_vec[eid, within]


def _sample_arclength_labeled(
    edge_points: np.ndarray, num: int, min_pts_per_edge: int = 0
) -> dict[str, np.ndarray]:
    """Arc-length sampling with per-point structure labels (stage-2 targets).

    Builds the same global arc-length distribution as :func:`_sample_arclength`,
    but first guarantees every edge receives at least ``min_pts_per_edge``
    uniformly-spaced samples (so short edges are never dropped), then fills the
    rest of the ``num`` budget by global arc length. For every sampled point it
    records the supervision the stage-2 grouper regresses:

      * ``edge_id (num,)``       -- source edge id;
      * ``arclen (num,)``        -- normalised arc-length position ``t``;
      * ``endpoint_a/b (num,3)`` -- the edge's two endpoint coords;
      * ``curve_type (num,)``    -- ``0=line / 1=arc / 2=bezier`` of the edge;
      * ``anchor1/anchor2 (num,3)`` -- the edge's ``t=1/3`` / ``t=2/3`` coords.
    """
    eps = 1e-12
    pts = np.asarray(edge_points, dtype=np.float64).reshape(
        -1, edge_points.shape[1], 3)
    e, u = pts.shape[0], pts.shape[1]
    out = {
        "points": np.zeros((num, 3), dtype=np.float32),
        "edge_id": np.full(num, -1, dtype=np.int64),
        "arclen": np.zeros(num, dtype=np.float32),
        "endpoint_a": np.zeros((num, 3), dtype=np.float32),
        "endpoint_b": np.zeros((num, 3), dtype=np.float32),
        "curve_type": np.zeros(num, dtype=np.int64),
        "anchor1": np.zeros((num, 3), dtype=np.float32),
        "anchor2": np.zeros((num, 3), dtype=np.float32),
    }
    if num <= 0 or e == 0 or u == 0:
        return out

    ea = pts[:, 0, :]   # (E, 3) edge endpoint a
    eb = pts[:, -1, :]  # (E, 3) edge endpoint b

    # Per-edge curve type (computed once on the clean polyline).
    curve_type = np.array(
        [_fit_curve_type(pts[i]) for i in range(e)], dtype=np.int64)

    def _finish(eid: np.ndarray, t: np.ndarray, coords: np.ndarray
                ) -> dict[str, np.ndarray]:
        out["points"] = coords.astype(np.float32)
        out["edge_id"] = eid.astype(np.int64)
        out["arclen"] = t.astype(np.float32)
        out["endpoint_a"] = ea[eid].astype(np.float32)
        out["endpoint_b"] = eb[eid].astype(np.float32)
        out["curve_type"] = curve_type[eid]
        out["anchor1"] = anchor1[eid].astype(np.float32)
        out["anchor2"] = anchor2[eid].astype(np.float32)
        return out

    # Degenerate polylines (single point per edge): fall back to flat sampling
    # with anchors / arclen pinned to the lone point.
    if u == 1:
        anchor1 = ea.copy()
        anchor2 = eb.copy()
        idx = np.random.choice(e, num, replace=e < num)
        return _finish(idx, np.zeros(num), pts.reshape(-1, 3)[idx])

    seg_start = pts[:, :-1, :]            # (E, U-1, 3)
    seg_vec = pts[:, 1:, :] - seg_start   # (E, U-1, 3)
    seg_len = np.linalg.norm(seg_vec, axis=2)        # (E, U-1)
    edge_len = seg_len.sum(axis=1)                   # (E,)
    seg_cum_prev = np.cumsum(seg_len, axis=1) - seg_len  # (E, U-1)

    # Edge anchors at t=1/3 and t=2/3 (arc-length param).
    all_ids = np.arange(e)
    anchor1 = _points_on_edges(
        seg_start, seg_vec, seg_len, edge_len, seg_cum_prev,
        all_ids, np.full(e, 1.0 / 3.0))
    anchor2 = _points_on_edges(
        seg_start, seg_vec, seg_len, edge_len, seg_cum_prev,
        all_ids, np.full(e, 2.0 / 3.0))

    total = float(seg_len.sum())
    if total <= eps:
        idx = np.random.choice(e, num, replace=e < num)
        return _finish(idx, np.zeros(num), ea[idx])

    # 1) Minimum per-edge allocation (uniform in arc length). Clamp the lower
    #    bound so the base never exceeds the total budget.
    per_edge_min = max(0, int(min_pts_per_edge))
    if e > 0 and per_edge_min * e > num:
        per_edge_min = num // e
    base_eid = np.zeros(0, dtype=np.int64)
    base_t = np.zeros(0, dtype=np.float64)
    if per_edge_min > 0:
        base_eid = np.repeat(all_ids, per_edge_min)
        # midpoints of per_edge_min equal sub-intervals -> spread within edge
        ts = (np.arange(per_edge_min) + 0.5) / per_edge_min
        base_t = np.tile(ts, e)

    # 2) Fill the remaining budget by global arc length.
    n_fill = num - base_eid.shape[0]
    if n_fill > 0:
        flat_len = seg_len.reshape(-1)
        cum = np.cumsum(flat_len)
        cum_prev = cum - flat_len
        u_pos = np.random.uniform(0.0, total, size=n_fill)
        seg_idx = np.clip(
            np.searchsorted(cum, u_pos, side="right"), 0, flat_len.shape[0] - 1)
        local = (u_pos - cum_prev[seg_idx]) / np.maximum(flat_len[seg_idx], eps)
        n_seg = u - 1
        fill_eid = seg_idx // n_seg
        within = seg_idx % n_seg
        dist_along = (
            seg_cum_prev[fill_eid, within] + local * seg_len[fill_eid, within])
        fill_t = dist_along / np.maximum(edge_len[fill_eid], eps)
    else:
        fill_eid = np.zeros(0, dtype=np.int64)
        fill_t = np.zeros(0, dtype=np.float64)

    eid = np.concatenate([base_eid, fill_eid]).astype(np.int64)[:num]
    t = np.concatenate([base_t, fill_t]).astype(np.float64)[:num]
    coords = _points_on_edges(
        seg_start, seg_vec, seg_len, edge_len, seg_cum_prev, eid, t)
    return _finish(eid, t, coords)


def _build_wf_target_labeled(
    edge_points: np.ndarray,
    num_points: int,
    min_pts_per_edge: int = 0,
) -> dict[str, np.ndarray]:
    """Build the fixed-size stage-2 anchor point set ``(N, 3)`` *plus* labels.

    Every point is an **edge point** (no vertex points); vertices are recovered
    downstream by endpoint voting. Each point carries the supervision the
    stage-2 grouper regresses:

      * ``edge_id (N,)``       -- source edge id;
      * ``arclen (N,)``        -- arc-length position ``t``;
      * ``endpoint_a/b (N,3)`` -- the edge's two endpoint coords;
      * ``curve_type (N,)``    -- ``0=line / 1=arc / 2=bezier`` of the edge;
      * ``anchor1/anchor2 (N,3)`` -- the edge's ``t=1/3`` / ``t=2/3`` coords.
    """
    s = _sample_arclength_labeled(
        edge_points, int(num_points), min_pts_per_edge=int(min_pts_per_edge))
    return {
        "points": s["points"],
        "edge_id": s["edge_id"],
        "arclen": s["arclen"],
        "endpoint_a": s["endpoint_a"],
        "endpoint_b": s["endpoint_b"],
        "curve_type": s["curve_type"],
        "anchor1": s["anchor1"],
        "anchor2": s["anchor2"],
    }


# ----------------------------------------------------------------------
# Graph format + file resolution
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class GraphFormat:
    """Graph (de)serialization format.

    ``max_vertices`` / ``max_edges`` are *safety caps*: a sample exceeding
    either cap is **skipped**. Set a cap to ``<= 0`` to disable it (unlimited),
    which is the RF-branch default -- we train on the full raw distribution.
    ``wf_num_points`` is the fixed RF target point budget ``N``.
    ``max_pc_points`` caps the (otherwise native-size) input point cloud; ``0``
    keeps all points.
    """

    vertex_merge_tol: float = 1e-4
    max_vertices: int = 0
    max_edges: int = 0
    num_edge_points: int = 32
    max_pc_points: int = 0
    wf_num_points: int = 8192


def list_npz(directory: str, recursive: bool = False) -> list[str]:
    """Return a sorted list of ``*.npz`` files under ``directory``."""
    directory = os.path.expandvars(os.path.expanduser(directory))
    pattern = (
        os.path.join(directory, "**", "*.npz")
        if recursive
        else os.path.join(directory, "*.npz")
    )
    return sorted(glob.glob(pattern, recursive=recursive))


def make_split(
    edge_dir: str,
    *,
    train_ratio: float = 0.9,
    split_seed: int = 42,
    recursive_glob: bool = False,
) -> dict[str, Any]:
    """Deterministically split the edge files in ``edge_dir`` into train/val.

    Returns ``{"train": [...], "val": [...], "meta": {...}}`` with the file
    lists sorted for readability. The shuffle is seeded so the split is fully
    reproducible.
    """
    files = list_npz(edge_dir, recursive=recursive_glob)
    rng = np.random.default_rng(split_seed)
    order = np.arange(len(files))
    rng.shuffle(order)
    cut = int(round(len(files) * train_ratio))
    train_files = sorted(files[int(i)] for i in order[:cut])
    val_files = sorted(files[int(i)] for i in order[cut:])
    return {
        "train": train_files,
        "val": val_files,
        "meta": {
            "edge_dir": edge_dir,
            "num_total": len(files),
            "num_train": len(train_files),
            "num_val": len(val_files),
            "train_ratio": train_ratio,
            "split_seed": split_seed,
            "recursive_glob": recursive_glob,
        },
    }


def save_split(split: dict[str, Any], out_path: str) -> None:
    """Write a split dict to ``out_path`` as JSON (creating parent dirs)."""
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)


def load_split(path: str) -> dict[str, Any]:
    """Load a previously saved split JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_split_files(
    split: str,
    *,
    split_path: str,
    edge_dir: str | None = None,
    train_ratio: float = 0.9,
    split_seed: int = 42,
    recursive_glob: bool = False,
    auto_build: bool = True,
) -> list[str]:
    """Resolve a named split to a list of edge files from a saved split file.

    ``split`` may be ``"train"`` / ``"val"`` (the corresponding saved list) or
    ``"all"`` / ``"trainval"`` (train + val combined). If ``split_path`` does
    not exist and ``auto_build`` is set, the split is built from ``edge_dir``
    and saved first, so the on-disk split stays the single source of truth.
    """
    if split not in ("train", "val", "all", "trainval"):
        raise ValueError(
            f"split must be train/val/all/trainval, got {split!r}")

    if not os.path.isfile(split_path):
        if not (auto_build and edge_dir):
            raise FileNotFoundError(
                f"Split file {split_path!r} not found. Run the split script "
                f"(scripts/make_split.py) first, or pass edge_dir for "
                f"auto-build."
            )
        _log.info("Split %r missing; building from %r", split_path, edge_dir)
        sp = make_split(
            edge_dir,
            train_ratio=train_ratio,
            split_seed=split_seed,
            recursive_glob=recursive_glob,
        )
        save_split(sp, split_path)
    else:
        sp = load_split(split_path)

    if split in ("all", "trainval"):
        return [str(p) for p in sp.get("train", [])] + [
            str(p) for p in sp.get("val", [])
        ]
    return [str(p) for p in sp.get(split, [])]


def _find_pointcloud_path(
    stem: str,
    pointcloud_dirs: list[str],
) -> str | None:
    for d in pointcloud_dirs:
        cand = os.path.join(d, f"{stem}.npz")
        if os.path.isfile(cand):
            return cand
    return None


# ----------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------
class WireframeGraphDataset(Dataset):
    """Loads point clouds + edge NPZ files into RF targets + native GT graphs.

    File membership comes from a *pre-saved* split file (``split_path``).
    ``split`` selects ``"train"`` / ``"val"`` / ``"all"`` (= train + val).
    """

    def __init__(
        self,
        *,
        split: str,
        split_path: str,
        edge_dir: str | None = None,
        pointcloud_dirs: list[str] | None = None,
        files: list[str] | None = None,
        train_ratio: float = 0.9,
        split_seed: int = 42,
        recursive_glob: bool = False,
        auto_build_split: bool = True,
        vertex_merge_tol: float = 1e-4,
        max_vertices: int = 0,
        max_edges: int = 0,
        num_edge_points: int = 32,
        max_pc_points: int = 0,
        wf_num_points: int = 8192,
        min_edges: int = 1,
        max_load_retries: int = 64,
        target_seed: int | None = None,
    ) -> None:
        super().__init__()
        # When set, the (otherwise random) arc-length / vertex sampling of the
        # RF target is seeded deterministically per index, so each shape always
        # yields the SAME wf_points. Needed for a clean single-sample overfit
        # (a target that changes every epoch can never be memorized exactly).
        self.target_seed = target_seed
        self.format = GraphFormat(
            vertex_merge_tol=vertex_merge_tol,
            max_vertices=max_vertices,
            max_edges=max_edges,
            num_edge_points=num_edge_points,
            max_pc_points=max_pc_points,
            wf_num_points=wf_num_points,
        )
        self.split = split
        self.pointcloud_dirs = [
            os.path.expandvars(os.path.expanduser(d))
            for d in (pointcloud_dirs or [])
        ]
        self.min_edges = int(min_edges)
        self.max_load_retries = max(1, int(max_load_retries))
        self._bad_files: set[str] = set()

        # An explicit ``files`` list bypasses split resolution entirely (used to
        # pin the dataset to a fixed set of shapes, e.g. single-sample overfit
        # where train and val are the same file). Otherwise resolve from split.
        if files is not None:
            resolved = [str(p) for p in files]
        else:
            resolved = resolve_split_files(
                split,
                split_path=split_path,
                edge_dir=edge_dir,
                train_ratio=train_ratio,
                split_seed=split_seed,
                recursive_glob=recursive_glob,
                auto_build=auto_build_split,
            )
        self.files = [p for p in resolved if os.path.isfile(p)]
        if not self.files:
            raise RuntimeError(
                f"No edge npz files found for split={split!r} "
                f"(split_path={split_path!r}, files={files!r})")

    def __len__(self) -> int:
        return len(self.files)

    def _load_graph(self, edge_path: str) -> dict[str, np.ndarray | int]:
        fmt = self.format
        data = _load_npz_dict(edge_path)
        start = _get_npz_array(data, ("start_verts", "start_vertices", "sv"))
        end = _get_npz_array(data, ("end_verts", "end_vertices", "ev"))
        raw_edge_points = _get_npz_array(data, ("edge_points", "curve_points"))

        start = np.asarray(start, dtype=np.float32).reshape(-1, 3)
        end = np.asarray(end, dtype=np.float32).reshape(-1, 3)
        n_raw = min(
            start.shape[0],
            end.shape[0],
            int(raw_edge_points.shape[0]),
        )

        vertex_ids: dict[tuple[int, int, int], int] = {}
        vertices: list[np.ndarray] = []
        edge_index: list[tuple[int, int]] = []
        edge_points: list[np.ndarray] = []
        tol = max(float(fmt.vertex_merge_tol), 1e-12)

        def get_vertex_id(v: np.ndarray) -> int | None:
            if v.shape != (3,) or not np.isfinite(v).all():
                return None
            key = tuple(np.rint(v / tol).astype(np.int64, copy=False).tolist())
            found = vertex_ids.get(key)
            if found is not None:
                return found
            vertex_ids[key] = len(vertices)
            vertices.append(v.astype(np.float32))
            return vertex_ids[key]

        for i in range(n_raw):
            if not (np.isfinite(start[i]).all() and np.isfinite(end[i]).all()):
                continue
            u = get_vertex_id(start[i])
            v = get_vertex_id(end[i])
            if u is None or v is None or u == v:
                continue
            edge_index.append((u, v))
            edge_points.append(_resample_polyline(
                raw_edge_points[i], fmt.num_edge_points))

        nv = len(vertices)
        ne = len(edge_index)

        if nv > 0:
            vertices_arr = np.stack(vertices, axis=0).astype(np.float32)
        else:
            vertices_arr = np.zeros((0, 3), dtype=np.float32)
        if ne > 0:
            edge_index_arr = np.asarray(edge_index, dtype=np.int64)
            edge_points_arr = np.stack(edge_points, axis=0).astype(np.float32)
        else:
            edge_index_arr = np.zeros((0, 2), dtype=np.int64)
            edge_points_arr = np.zeros(
                (0, fmt.num_edge_points, 3), dtype=np.float32)

        return {
            "vertices": vertices_arr,
            "edge_index": edge_index_arr,
            "edge_points": edge_points_arr,
            "num_vertices": nv,
            "num_edges": ne,
        }

    def _load_point_cloud(self, edge_path: str) -> np.ndarray:
        stem = os.path.splitext(os.path.basename(edge_path))[0]
        pc_path = _find_pointcloud_path(stem, self.pointcloud_dirs)
        if pc_path is None:
            raise FileNotFoundError(
                f"No matching point cloud for {stem!r}")
        data = _load_npz_dict(pc_path)
        points = _get_npz_array(
            data, ("surface_points", "points", "point_cloud", "pc"))
        return _clean_point_cloud(points, self.format.max_pc_points)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.target_seed is not None:
            np.random.seed((int(self.target_seed) + int(idx)) % (2**32))
        n_files = len(self.files)
        max_tries = min(max(self.max_load_retries, 1), max(n_files * 2, 1))
        start_idx = int(idx) % n_files
        for k in range(max_tries):
            cur_idx = (start_idx + k) % n_files
            edge_path = self.files[cur_idx]
            if edge_path in self._bad_files:
                continue
            try:
                graph = self._load_graph(edge_path)
                nv = int(graph["num_vertices"])
                ne = int(graph["num_edges"])
                mv = self.format.max_vertices
                me = self.format.max_edges
                # Skip to the next sample if too few edges or oversize.
                # A cap <= 0 disables that check (RF-branch default).
                if ne < self.min_edges:
                    continue
                if (mv > 0 and nv > mv) or (me > 0 and ne > me):
                    continue
                # Per-shape normalization: one transform derived from the point
                # cloud, applied to the point cloud AND the wireframe vertices /
                # curves so everything lives in one ~[-1, 1] frame.
                pc = self._load_point_cloud(edge_path)
                if pc.shape[0] < 1:
                    continue
                center, scale = _unit_cube_transform(pc)
                pc = _apply_unit_cube(pc, center, scale)

                vertices = _apply_unit_cube(graph["vertices"], center, scale)
                edge_points = _apply_unit_cube(
                    graph["edge_points"], center, scale)

                return self._make_item(
                    shape_id=os.path.splitext(
                        os.path.basename(edge_path))[0],
                    pc=pc,
                    center=center,
                    scale=scale,
                    vertices=vertices,
                    edge_index=graph["edge_index"],
                    edge_points=edge_points,
                )
            except Exception as exc:
                self._bad_files.add(edge_path)
                if len(self._bad_files) <= 20:
                    _log.warning("Bad sample %s: %s", edge_path, exc)
                continue
        raise RuntimeError(
            f"[WireframeGraphDataset] No valid sample after {max_tries} tries "
            f"(split={self.split!r}, bad_files={len(self._bad_files)})"
        )

    def _make_item(
        self,
        *,
        shape_id: str,
        pc: np.ndarray,
        center: np.ndarray,
        scale: float,
        vertices: np.ndarray,
        edge_index: np.ndarray,
        edge_points: np.ndarray,
    ) -> dict[str, Any]:
        """Build the per-sample dict from a normalized graph + point cloud.

        Factored out of ``__getitem__`` so subclasses (e.g. the stage-2
        ``WireframePointDataset``) can reuse the file loading / normalization
        retry loop and only customise the emitted tensors.
        """
        wf_points = _build_wf_target(edge_points, self.format.wf_num_points)
        return {
            "shape_id": shape_id,
            "point_cloud": torch.from_numpy(np.ascontiguousarray(pc)),
            "pc_center": torch.from_numpy(center),
            "pc_scale": torch.tensor(scale, dtype=torch.float32),
            "wf_points": torch.from_numpy(np.ascontiguousarray(wf_points)),
            "vertices": torch.from_numpy(np.ascontiguousarray(vertices)),
            "edge_index": torch.from_numpy(np.ascontiguousarray(edge_index)),
            "edge_points": torch.from_numpy(np.ascontiguousarray(edge_points)),
        }


class WireframePointDataset(WireframeGraphDataset):
    """Stage-2 dataset: the GT wireframe **anchor set** + per-point structure labels.

    Reuses :class:`WireframeGraphDataset`'s file resolution / normalization /
    retry loop and only changes the emitted sample: instead of the bare RF
    target ``wf_points`` it returns the labelled target from
    :func:`_build_wf_target_labeled` (edge id, arc-length, endpoints, curve type,
    anchors) so the stage-2 grouper network can be trained directly on GT anchor
    point sets. Every point is an edge point (vertices are recovered downstream
    by endpoint voting); the point set is pure xyz ``(N, 3)``.

    To bridge the gap between the clean GT point set (training) and the noisy
    stage-1 RF output (inference), the *input* point set is optionally augmented
    with per-point Gaussian xyz ``jitter_std`` (normalized frame) while the
    *labels stay clean*. Augmentation is label-preserving (point count and
    per-point identity are unchanged), so the fixed-size tensors still stack
    cleanly in the collate.

    ``min_pts_per_edge`` guarantees each GT edge receives at least that many
    sampled points so short edges are never lost under pure arc-length sampling.
    """

    def __init__(
        self,
        *,
        jitter_std: float = 0.0,
        min_pts_per_edge: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.jitter_std = float(jitter_std)
        self.min_pts_per_edge = int(min_pts_per_edge)

    def _make_item(
        self,
        *,
        shape_id: str,
        pc: np.ndarray,
        center: np.ndarray,
        scale: float,
        vertices: np.ndarray,
        edge_index: np.ndarray,
        edge_points: np.ndarray,
    ) -> dict[str, Any]:
        lab = _build_wf_target_labeled(
            edge_points, self.format.wf_num_points,
            min_pts_per_edge=self.min_pts_per_edge)
        wf_points = lab["points"].copy()  # (N, 3) -- the (augmentable) input

        if self.jitter_std > 0.0:
            wf_points += np.random.normal(
                0.0, self.jitter_std, size=wf_points.shape
            ).astype(np.float32)

        def _t(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(a))

        return {
            "shape_id": shape_id,
            "pc_center": _t(center),
            "pc_scale": torch.tensor(scale, dtype=torch.float32),
            "wf_points": _t(wf_points),
            # per-point stage-2 labels (clean)
            "lbl_edge_id": _t(lab["edge_id"]),
            "lbl_arclen": _t(lab["arclen"]),
            "lbl_endpoint_a": _t(lab["endpoint_a"]),
            "lbl_endpoint_b": _t(lab["endpoint_b"]),
            "lbl_curve_type": _t(lab["curve_type"]),
            "lbl_anchor1": _t(lab["anchor1"]),
            "lbl_anchor2": _t(lab["anchor2"]),
            # native GT graph (for decode-time metrics + topo loss)
            "vertices": _t(vertices),
            "edge_index": _t(edge_index),
            "edge_points": _t(edge_points),
        }


class PointCloudDataset(Dataset):
    """Point-cloud-only dataset for inference / submission.

    The competition test split provides ``surface_points`` but no edges, so
    each sample is simply ``{shape_id, point_cloud, pc_center, pc_scale}``.
    """

    def __init__(
        self,
        *,
        pointcloud_dir: str,
        max_pc_points: int = 0,
        recursive_glob: bool = False,
    ) -> None:
        super().__init__()
        self.max_pc_points = int(max_pc_points)
        self.files = list_npz(pointcloud_dir, recursive=recursive_glob)
        if not self.files:
            raise RuntimeError(
                f"No point cloud npz files found in {pointcloud_dir!r}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pc_path = self.files[int(idx) % len(self.files)]
        data = _load_npz_dict(pc_path)
        points = _get_npz_array(
            data, ("surface_points", "points", "point_cloud", "pc"))
        pc = _clean_point_cloud(points, self.max_pc_points)
        center, scale = _unit_cube_transform(pc)
        pc = _apply_unit_cube(pc, center, scale)
        return {
            "shape_id": os.path.splitext(os.path.basename(pc_path))[0],
            "point_cloud": torch.from_numpy(np.ascontiguousarray(pc)),
            "pc_center": torch.from_numpy(center),
            "pc_scale": torch.tensor(scale, dtype=torch.float32),
        }


# ----------------------------------------------------------------------
# Stacking collate (fixed-size tensors stacked; native GT carried as a list)
# ----------------------------------------------------------------------
def collate_rf_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate RF samples into a batch.

    The input point clouds are *variable size* (native surface points), so they
    are **packed** into PTv3's ``coord`` / ``offset`` layout: all points are
    concatenated along dim 0 and a CSR-style cumulative-count ``pc_offset``
    records sample boundaries (``offset2batch`` recovers per-point membership).
    The fixed-size ``wf_points`` (training target) is stacked. The native-size
    GT graph is kept as a list under ``gt_wireframes`` for the val metrics.
    Works for both training samples and prediction samples (which carry neither
    ``wf_points`` nor GT).

    Returned dict (``B`` = batch size, ``P_sum`` = total points in the batch)::

        shape_id:      list[str]                length B
        point_cloud:   (P_sum, 3)               float32  (packed)
        pc_offset:     (B,)                     int64    cumsum of per-sample N
        pc_center:     (B, 3)                   float32
        pc_scale:      (B,)                     float32
        wf_points:     (B, wf_num_points, 3)    float32  (if present)
        gt_wireframes: list[dict]               length B (if present)
        num_graphs:    int                      == B
    """
    point_cloud = torch.cat([s["point_cloud"] for s in samples], dim=0)
    lengths = torch.tensor(
        [int(s["point_cloud"].shape[0]) for s in samples], dtype=torch.long)
    pc_offset = torch.cumsum(lengths, dim=0)
    pc_center = torch.stack([s["pc_center"] for s in samples], dim=0)
    pc_scale = torch.stack([s["pc_scale"] for s in samples], dim=0)

    batch: dict[str, Any] = {
        "shape_id": [str(s["shape_id"]) for s in samples],
        "point_cloud": point_cloud,
        "pc_offset": pc_offset,
        "pc_center": pc_center,
        "pc_scale": pc_scale,
        "num_graphs": len(samples),
    }
    if "wf_points" in samples[0]:
        batch["wf_points"] = torch.stack(
            [s["wf_points"] for s in samples], dim=0)
    if "vertices" in samples[0]:
        batch["gt_wireframes"] = [
            {
                "vertices": s["vertices"],
                "edge_index": s["edge_index"],
                "edge_points": s["edge_points"],
            }
            for s in samples
        ]
    return batch


def collate_grouper_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate grouper samples: stack the fixed-size point set + per-point labels.

    Every tensor here is fixed size ``(N, ...)`` (the labelled RF target), so
    unlike the variable-size point clouds in :func:`collate_rf_batch` they all
    stack directly into ``(B, N, ...)``. The native-size GT graph is kept as a
    Python list under ``gt_wireframes`` for decode-time metrics.
    """
    stack_keys = [
        "wf_points", "lbl_edge_id", "lbl_arclen",
        "lbl_endpoint_a", "lbl_endpoint_b",
        "lbl_curve_type", "lbl_anchor1", "lbl_anchor2",
    ]
    batch: dict[str, Any] = {
        "shape_id": [str(s["shape_id"]) for s in samples],
        "num_graphs": len(samples),
    }
    for k in stack_keys:
        batch[k] = torch.stack([s[k] for s in samples], dim=0)
    if "pc_center" in samples[0]:
        batch["pc_center"] = torch.stack(
            [s["pc_center"] for s in samples], dim=0)
        batch["pc_scale"] = torch.stack(
            [s["pc_scale"] for s in samples], dim=0)
    if "vertices" in samples[0]:
        batch["gt_wireframes"] = [
            {
                "vertices": s["vertices"],
                "edge_index": s["edge_index"],
                "edge_points": s["edge_points"],
            }
            for s in samples
        ]
    return batch


__all__ = [
    "GraphFormat",
    "WireframeGraphDataset",
    "WireframePointDataset",
    "PointCloudDataset",
    "collate_rf_batch",
    "collate_grouper_batch",
    "list_npz",
    "make_split",
    "save_split",
    "load_split",
    "resolve_split_files",
]
