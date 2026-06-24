"""Wireframe datasets for the single-stage WireframeAE branch.

``WireframeGraphDataset``
    Used for training / validation. Loads a per-edge GT wireframe NPZ
    (endpoints + resampled curve points) together with its matching surface
    point cloud, and emits the **native** GT wireframe graph (vertices +
    edge_index) annotated with the per-edge supervision the WireframeAE
    decoder regresses:

      * ``edge_type``   ``(E,)``     -- 0=line / 1=arc / 2=bezier
        (geometric residual fit, :func:`_fit_curve_type`);
      * ``edge_params`` ``(E, 2, 3)``-- the curve coordinates at the arc-length
        fractions ``t = 1/3`` (``q1``, near the start vertex) and ``t = 2/3``
        (``q2``, near the end vertex).

    Coordinates are kept **raw**: the dataset is already normalized to the unit
    cube (``[-1, 1]``), so the branch trains and supervises directly in that
    frame with no extra per-shape normalization. Point clouds with fewer than
    ``min_pc_points`` points are *skipped* (the loader retries the next file),
    as are wireframes with more than ``max_vertices`` vertices.

``PointCloudDataset``
    Used for prediction / submission. The test split ships only point clouds
    (no ground-truth edges), so this dataset loads ``surface_points`` (cleaned,
    not normalized) and returns ``{shape_id, point_cloud}`` for inference.
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
    via the packed ``coord``/``offset`` format. When ``max_points > 0`` and the
    cloud is larger, it is randomly subsampled (no replacement) only as a memory
    safety cap. Returns ``(M, 3)`` float32.
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


# ----------------------------------------------------------------------
# Per-edge curve type + parameter targets
# ----------------------------------------------------------------------
# Curve-type fit thresholds (max residual relative to the edge arc length).
_CURVE_LINE_THRESH = 0.02
_CURVE_ARC_THRESH = 0.02

# Curve-type codes (kept in sync with the WireframeAE edge head + decoder).
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


def _curve_point_at(polyline: np.ndarray, frac: float) -> np.ndarray:
    """Coordinate at normalised arc-length ``frac in [0, 1]`` on a polyline."""
    pts = np.asarray(polyline, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros(3, dtype=np.float64)
    if pts.shape[0] == 1:
        return pts[0]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 1e-12:
        return pts[0]
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    target = float(np.clip(frac, 0.0, 1.0)) * total
    j = int(np.clip(np.searchsorted(cum, target, side="right") - 1,
                    0, seg.shape[0] - 1))
    local = (target - cum[j]) / max(seg[j], 1e-12)
    return pts[j] + local * (pts[j + 1] - pts[j])


def _edge_params(edge_points: np.ndarray) -> np.ndarray:
    """Per-edge ``(E, 2, 3)`` anchors at the t=1/3 (q1) and t=2/3 (q2) points."""
    pts = np.asarray(edge_points, dtype=np.float64)
    e = pts.shape[0]
    out = np.zeros((e, 2, 3), dtype=np.float32)
    for i in range(e):
        out[i, 0] = _curve_point_at(pts[i], 1.0 / 3.0)
        out[i, 1] = _curve_point_at(pts[i], 2.0 / 3.0)
    return out


# ----------------------------------------------------------------------
# Graph format + file resolution
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class GraphFormat:
    """Graph (de)serialization format.

    ``max_vertices`` / ``max_edges`` are *safety caps*: a sample exceeding
    either cap is **skipped**. ``max_vertices`` doubles as the WireframeAE
    vertex-query budget, so it must not exceed the decoder's query count.
    ``num_edge_points`` is the GT curve sampling resolution (= submission
    resolution). ``max_pc_points`` caps the (otherwise native-size) input point
    cloud; ``0`` keeps all points.
    """

    vertex_merge_tol: float = 1e-4
    max_vertices: int = 512
    max_edges: int = 0
    num_edge_points: int = 32
    max_pc_points: int = 0


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
    """Deterministically split the edge files in ``edge_dir`` into train/val."""
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
    """Loads point clouds + edge NPZ files into native GT wireframe graphs.

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
        max_vertices: int = 512,
        max_edges: int = 0,
        num_edge_points: int = 32,
        max_pc_points: int = 0,
        min_edges: int = 1,
        min_pc_points: int = 100,
        max_load_retries: int = 64,
    ) -> None:
        super().__init__()
        self.format = GraphFormat(
            vertex_merge_tol=vertex_merge_tol,
            max_vertices=max_vertices,
            max_edges=max_edges,
            num_edge_points=num_edge_points,
            max_pc_points=max_pc_points,
        )
        self.split = split
        self.pointcloud_dirs = [
            os.path.expandvars(os.path.expanduser(d))
            for d in (pointcloud_dirs or [])
        ]
        self.min_edges = int(min_edges)
        self.min_pc_points = max(0, int(min_pc_points))
        self.max_load_retries = max(1, int(max_load_retries))
        self._bad_files: set[str] = set()

        # An explicit ``files`` list bypasses split resolution entirely.
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
            edge_type_arr = np.array(
                [_fit_curve_type(ep) for ep in edge_points_arr], dtype=np.int64)
            edge_params_arr = _edge_params(edge_points_arr)
        else:
            edge_index_arr = np.zeros((0, 2), dtype=np.int64)
            edge_points_arr = np.zeros(
                (0, fmt.num_edge_points, 3), dtype=np.float32)
            edge_type_arr = np.zeros((0,), dtype=np.int64)
            edge_params_arr = np.zeros((0, 2, 3), dtype=np.float32)

        return {
            "vertices": vertices_arr,
            "edge_index": edge_index_arr,
            "edge_points": edge_points_arr,
            "edge_type": edge_type_arr,
            "edge_params": edge_params_arr,
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
                if ne < self.min_edges:
                    continue
                if (mv > 0 and nv > mv) or (me > 0 and ne > me):
                    continue
                # Raw frame (data is already unit-normalized): skip clouds that
                # are too sparse to encode reliably.
                pc = self._load_point_cloud(edge_path)
                if pc.shape[0] < self.min_pc_points:
                    continue

                return self._make_item(
                    shape_id=os.path.splitext(
                        os.path.basename(edge_path))[0],
                    pc=pc,
                    vertices=graph["vertices"],
                    edge_index=graph["edge_index"],
                    edge_points=graph["edge_points"],
                    edge_type=graph["edge_type"],
                    edge_params=graph["edge_params"],
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
        vertices: np.ndarray,
        edge_index: np.ndarray,
        edge_points: np.ndarray,
        edge_type: np.ndarray,
        edge_params: np.ndarray,
    ) -> dict[str, Any]:
        def _t(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(a))

        return {
            "shape_id": shape_id,
            "point_cloud": _t(pc),
            "vertices": _t(vertices),
            "edge_index": _t(edge_index),
            "edge_points": _t(edge_points),
            "edge_type": _t(edge_type),
            "edge_params": _t(edge_params),
        }


class PointCloudDataset(Dataset):
    """Point-cloud-only dataset for inference / submission.

    The competition test split provides ``surface_points`` but no edges, so
    each sample is simply ``{shape_id, point_cloud}``. Coordinates are cleaned
    but not normalized (the WireframeAE works directly in the raw frame).
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
        return {
            "shape_id": os.path.splitext(os.path.basename(pc_path))[0],
            "point_cloud": torch.from_numpy(np.ascontiguousarray(pc)),
        }


# ----------------------------------------------------------------------
# Collate: pack variable-size clouds (PTv3 coord/offset) + GT graph list
# ----------------------------------------------------------------------
def collate_ae_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate WireframeAE samples into a batch.

    The input point clouds are *variable size* (native surface points), so they
    are **packed** into PTv3's ``coord`` / ``offset`` layout: all points are
    concatenated along dim 0 and a CSR-style cumulative-count ``pc_offset``
    records sample boundaries (``offset2batch`` recovers per-point membership).
    The native-size per-sample GT graphs are carried as a Python list under
    ``gt_wireframes`` for the Hungarian-matched loss + the val metrics. Works
    for both training samples and prediction samples (which carry no GT).

    Returned dict (``B`` = batch size, ``P_sum`` = total points in the batch)::

        shape_id:      list[str]                length B
        point_cloud:   (P_sum, 3)               float32  (packed)
        pc_offset:     (B,)                     int64    cumsum of per-sample N
        gt_wireframes: list[dict]               length B (if present)
        num_graphs:    int                      == B
    """
    point_cloud = torch.cat([s["point_cloud"] for s in samples], dim=0)
    lengths = torch.tensor(
        [int(s["point_cloud"].shape[0]) for s in samples], dtype=torch.long)
    pc_offset = torch.cumsum(lengths, dim=0)

    batch: dict[str, Any] = {
        "shape_id": [str(s["shape_id"]) for s in samples],
        "point_cloud": point_cloud,
        "pc_offset": pc_offset,
        "num_graphs": len(samples),
    }
    if "vertices" in samples[0]:
        batch["gt_wireframes"] = [
            {
                "vertices": s["vertices"],
                "edge_index": s["edge_index"],
                "edge_points": s["edge_points"],
                "edge_type": s["edge_type"],
                "edge_params": s["edge_params"],
            }
            for s in samples
        ]
    return batch


__all__ = [
    "GraphFormat",
    "WireframeGraphDataset",
    "PointCloudDataset",
    "collate_ae_batch",
    "list_npz",
    "make_split",
    "save_split",
    "load_split",
    "resolve_split_files",
]
