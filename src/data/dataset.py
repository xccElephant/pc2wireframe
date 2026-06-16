"""Wireframe datasets for the Rectified-Flow PC2Wireframe branch.

Two dataset flavours are provided:

``WireframeGraphDataset``
    Used for training / validation. Loads a per-edge GT wireframe NPZ
    (endpoints + resampled curve points) together with its matching surface
    point cloud, then derives the Rectified-Flow **target point set**
    ``wf_points (N, 4) = (x, y, z, type)`` where ``type`` is ``1`` for vertex
    points and ``0`` for edge points:

      * every GT vertex contributes one ``type=1`` point (down-sampled when the
        sample has more than ``N`` vertices);
      * the remaining ``N - V`` points are drawn by **global arc-length
        sampling** over all edge polylines, with ``type=0``.

    The fixed-size ``point_cloud`` and ``wf_points`` are stacked by the default
    collate (:func:`collate_rf_batch`); the *native-size* GT graph
    (``vertices`` / ``edge_index`` / ``edge_points``) is carried alongside as a
    Python list under ``gt_wireframes`` so the validation metrics can score
    against the real wireframe.

    Each sample (``__getitem__``) yields::

        shape_id:     str
        point_cloud:  (pc_num_points, 3)   float32   (fixed size)
        pc_center:    (3,)                 float32
        pc_scale:     ()                   float32
        wf_points:    (wf_num_points, 4)   float32   (fixed size, RF target)
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


def _sample_points(points: np.ndarray, num_points: int) -> np.ndarray:
    """Randomly (re)sample a point cloud to exactly ``num_points`` points."""
    points = np.asarray(points, dtype=np.float64)
    points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    points = points[:, :3]
    finite_mask = (
        np.isfinite(points).all(axis=1)
        & (np.abs(points).max(axis=1) < _PC_COORD_CLIP)
    )
    points = points[finite_mask]
    if points.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    points = np.clip(points, -_PC_COORD_CLIP, _PC_COORD_CLIP).astype(np.float32)
    replace = points.shape[0] < num_points
    idx = np.random.choice(points.shape[0], size=num_points, replace=replace)
    return points[idx]


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


def _build_wf_target(
    vertices: np.ndarray, edge_points: np.ndarray, num_points: int
) -> np.ndarray:
    """Build the fixed-size RF target point set ``(num_points, 4)``.

    The type channel is *not* a fixed budget: all vertices are emitted with
    ``type=1`` (down-sampled if ``V > num_points``) and the rest of the budget
    is filled with arc-length-sampled edge points (``type=0``).
    """
    out = np.zeros((num_points, 4), dtype=np.float32)
    v = int(vertices.shape[0])
    if v >= num_points:
        idx = np.random.choice(v, num_points, replace=False)
        out[:, :3] = vertices[idx]
        out[:, 3] = 1.0
        return out
    if v > 0:
        out[:v, :3] = vertices
        out[:v, 3] = 1.0
    m = num_points - v
    out[v:, :3] = _sample_arclength(edge_points, m)
    out[v:, 3] = 0.0
    return out


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
    """

    vertex_merge_tol: float = 1e-4
    max_vertices: int = 0
    max_edges: int = 0
    num_edge_points: int = 32
    pc_num_points: int = 4096
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
        train_ratio: float = 0.9,
        split_seed: int = 42,
        recursive_glob: bool = False,
        auto_build_split: bool = True,
        vertex_merge_tol: float = 1e-4,
        max_vertices: int = 0,
        max_edges: int = 0,
        num_edge_points: int = 32,
        pc_num_points: int = 4096,
        wf_num_points: int = 8192,
        min_edges: int = 1,
        max_load_retries: int = 64,
    ) -> None:
        super().__init__()
        self.format = GraphFormat(
            vertex_merge_tol=vertex_merge_tol,
            max_vertices=max_vertices,
            max_edges=max_edges,
            num_edge_points=num_edge_points,
            pc_num_points=pc_num_points,
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

        files = resolve_split_files(
            split,
            split_path=split_path,
            edge_dir=edge_dir,
            train_ratio=train_ratio,
            split_seed=split_seed,
            recursive_glob=recursive_glob,
            auto_build=auto_build_split,
        )
        self.files = [p for p in files if os.path.isfile(p)]
        if not self.files:
            raise RuntimeError(
                f"No edge npz files found for split={split!r} "
                f"(split_path={split_path!r})")

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
            return np.zeros((self.format.pc_num_points, 3), dtype=np.float32)
        data = _load_npz_dict(pc_path)
        points = _get_npz_array(
            data, ("surface_points", "points", "point_cloud", "pc"))
        return _sample_points(points, self.format.pc_num_points)

    def __getitem__(self, idx: int) -> dict[str, Any]:
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
                center, scale = _unit_cube_transform(pc)
                pc = _apply_unit_cube(pc, center, scale)

                vertices = _apply_unit_cube(graph["vertices"], center, scale)
                edge_points = _apply_unit_cube(
                    graph["edge_points"], center, scale)
                wf_points = _build_wf_target(
                    vertices, edge_points, self.format.wf_num_points)

                return {
                    "shape_id": os.path.splitext(
                        os.path.basename(edge_path))[0],
                    "point_cloud": torch.from_numpy(
                        np.ascontiguousarray(pc)),
                    "pc_center": torch.from_numpy(center),
                    "pc_scale": torch.tensor(scale, dtype=torch.float32),
                    "wf_points": torch.from_numpy(
                        np.ascontiguousarray(wf_points)),
                    "vertices": torch.from_numpy(
                        np.ascontiguousarray(vertices)),
                    "edge_index": torch.from_numpy(
                        np.ascontiguousarray(graph["edge_index"])),
                    "edge_points": torch.from_numpy(
                        np.ascontiguousarray(edge_points)),
                }
            except Exception as exc:
                self._bad_files.add(edge_path)
                if len(self._bad_files) <= 20:
                    _log.warning("Bad sample %s: %s", edge_path, exc)
                continue
        raise RuntimeError(
            f"[WireframeGraphDataset] No valid sample after {max_tries} tries "
            f"(split={self.split!r}, bad_files={len(self._bad_files)})"
        )


class PointCloudDataset(Dataset):
    """Point-cloud-only dataset for inference / submission.

    The competition test split provides ``surface_points`` but no edges, so
    each sample is simply ``{shape_id, point_cloud, pc_center, pc_scale}``.
    """

    def __init__(
        self,
        *,
        pointcloud_dir: str,
        pc_num_points: int = 4096,
        recursive_glob: bool = False,
    ) -> None:
        super().__init__()
        self.pc_num_points = int(pc_num_points)
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
        pc = _sample_points(points, self.pc_num_points)
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

    ``point_cloud`` / ``wf_points`` / ``pc_center`` / ``pc_scale`` are all
    fixed size and simply stacked. The native-size GT graph of each sample is
    kept as a list of dicts under ``gt_wireframes`` (used only by the
    validation metrics).

    Returned dict (``B`` = batch size)::

        shape_id:      list[str]                length B
        point_cloud:   (B, pc_num_points, 3)    float32
        pc_center:     (B, 3)                   float32
        pc_scale:      (B,)                     float32
        wf_points:     (B, wf_num_points, 4)    float32
        gt_wireframes: list[dict]               length B (vertices/edge_index/
                                                edge_points tensors)
    """
    point_cloud = torch.stack([s["point_cloud"] for s in samples], dim=0)
    pc_center = torch.stack([s["pc_center"] for s in samples], dim=0)
    pc_scale = torch.stack([s["pc_scale"] for s in samples], dim=0)
    wf_points = torch.stack([s["wf_points"] for s in samples], dim=0)
    gt_wireframes = [
        {
            "vertices": s["vertices"],
            "edge_index": s["edge_index"],
            "edge_points": s["edge_points"],
        }
        for s in samples
    ]
    return {
        "shape_id": [str(s["shape_id"]) for s in samples],
        "point_cloud": point_cloud,
        "pc_center": pc_center,
        "pc_scale": pc_scale,
        "wf_points": wf_points,
        "gt_wireframes": gt_wireframes,
        "num_graphs": len(samples),
    }


__all__ = [
    "GraphFormat",
    "WireframeGraphDataset",
    "PointCloudDataset",
    "collate_rf_batch",
    "list_npz",
    "make_split",
    "save_split",
    "load_split",
    "resolve_split_files",
]
