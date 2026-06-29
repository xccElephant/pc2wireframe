"""Wireframe datasets for the single-stage WireframeAE branch.

``WireframeGraphDataset``
    Used for training / validation. Loads a per-edge GT wireframe NPZ
    (endpoints + resampled curve points) together with its matching surface
    point cloud, and emits the **native** GT wireframe graph
    (``vertices`` + ``edge_index`` + ordered ``edge_points (E, P, 3)``). The
    edge-centric decoder regresses each edge's ordered sample points directly
    (its first / last points are the endpoints), so no parametric curve-type /
    anchor supervision is produced.

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


def _clean_polyline(points: np.ndarray) -> np.ndarray:
    """Drop NaN/Inf + clip a raw ``(K, 3)`` polyline (no resampling)."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = p[np.isfinite(p).all(axis=1)]
    if p.shape[0]:
        p = np.clip(p, -_PC_COORD_CLIP, _PC_COORD_CLIP)
    return p


def _polyline_length(points: np.ndarray) -> float:
    """Total arc length of an ordered ``(K, 3)`` polyline."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if p.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(p[1:] - p[:-1], axis=1).sum())


def _split_loop(
    pts_full: np.ndarray, num_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Split a closed polyline into two arcs at its point farthest from start.

    Returns ``(far_point, arc1, arc2)`` where ``arc1`` runs start -> far and
    ``arc2`` runs far -> end (~start), each resampled to ``num_points``. The two
    arcs share the inserted ``far`` vertex and the loop's junction vertex, so the
    closed edge becomes two open edges (a faithful 2-cycle). Returns ``None`` if
    the polyline is too short to split.
    """
    k = pts_full.shape[0]
    if k < 3:
        return None
    d = np.linalg.norm(pts_full - pts_full[0], axis=1)
    m = int(np.argmax(d))
    if m <= 0 or m >= k - 1:
        m = k // 2
    if m <= 0 or m >= k - 1:
        return None
    far = pts_full[m].astype(np.float32)
    arc1 = _resample_polyline(pts_full[: m + 1], num_points)
    arc2 = _resample_polyline(pts_full[m:], num_points)
    return far, arc1, arc2


def _max_deviation_index(pts: np.ndarray) -> int:
    """Index of the polyline point farthest from its chord (start -> end).

    Used by the recursive complex-edge splitter to pick a split point. Returns
    an interior index in ``[1, K-2]`` (or ``-1`` if the polyline is too short).
    """
    k = pts.shape[0]
    if k < 3:
        return -1
    a = pts[0]
    b = pts[-1]
    ab = b - a
    ab_len = float(np.linalg.norm(ab))
    if ab_len < 1e-12:
        # Degenerate chord: deviate by raw distance from the start instead.
        dev = np.linalg.norm(pts - a, axis=1)
    else:
        u = ab / ab_len
        rel = pts - a[None, :]
        proj = rel @ u
        perp = rel - proj[:, None] * u[None, :]
        dev = np.linalg.norm(perp, axis=1)
    m = int(np.argmax(dev))
    if m <= 0 or m >= k - 1:
        m = k // 2
    if m <= 0 or m >= k - 1:
        return -1
    return m


def _split_complex_polyline(
    pts: np.ndarray,
    *,
    complex_ratio: float,
    complex_min_arc: float,
    max_depth: int,
    _depth: int = 0,
) -> list[np.ndarray]:
    """Recursively split a "complex" open polyline into simpler sub-polylines.

    An edge is *complex* (hard for a single 12-d curve latent to express) when
    its arc length greatly exceeds its endpoint chord (large bend, big arc,
    spiral / multiple windings). Such a polyline is split at its point of maximum
    deviation from the chord, inserting a new vertex, and each half is processed
    recursively until every segment is simple enough (``arclen/chord <=
    complex_ratio``) or ``max_depth`` is reached. Consecutive returned segments
    share endpoints, so the open edge becomes a chain of simple edges.

    Returns a list of raw sub-polylines (not resampled); a simple edge yields a
    single-element list ``[pts]``.
    """
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if _depth >= int(max_depth) or p.shape[0] < 3:
        return [p]
    chord = float(np.linalg.norm(p[-1] - p[0]))
    arclen = _polyline_length(p)
    if arclen <= float(complex_min_arc):
        return [p]
    # Treat a near-closed sub-arc (tiny chord but real arc) as complex too.
    ratio = arclen / chord if chord > 1e-9 else float("inf")
    if ratio <= float(complex_ratio):
        return [p]
    m = _max_deviation_index(p)
    if m < 0:
        return [p]
    left = _split_complex_polyline(
        p[: m + 1], complex_ratio=complex_ratio,
        complex_min_arc=complex_min_arc, max_depth=max_depth, _depth=_depth + 1)
    right = _split_complex_polyline(
        p[m:], complex_ratio=complex_ratio,
        complex_min_arc=complex_min_arc, max_depth=max_depth, _depth=_depth + 1)
    return left + right


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
        # ----- closed-loop edge splitting -----
        split_loops: bool = True,
        loop_close_tol: float = 1e-2,
        loop_min_arc: float = 5e-2,
        # ----- complex (high arc/chord) edge splitting -----
        split_complex: bool = True,
        complex_ratio: float = 1.6,
        complex_max_depth: int = 4,
        complex_min_arc: float = 5e-2,
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
        self.split_loops = bool(split_loops)
        self.loop_close_tol = float(loop_close_tol)
        self.loop_min_arc = float(loop_min_arc)
        self.split_complex = bool(split_complex)
        self.complex_ratio = float(complex_ratio)
        self.complex_max_depth = int(complex_max_depth)
        self.complex_min_arc = float(complex_min_arc)
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

        def add_open_polyline(poly: np.ndarray) -> None:
            """Add one open polyline as one (or several, if complex) edges.

            A "complex" edge (arc length >> endpoint chord: big arc / spiral /
            multi-winding spline) is recursively split into simpler sub-edges
            with inserted vertices, since a single 12-d curve latent cannot
            faithfully represent it. Each resulting segment is resampled to
            ``num_edge_points`` and added between its (deduplicated) endpoints.
            """
            poly = np.asarray(poly, dtype=np.float64).reshape(-1, 3)
            if poly.shape[0] < 2:
                return
            if self.split_complex:
                segments = _split_complex_polyline(
                    poly,
                    complex_ratio=self.complex_ratio,
                    complex_min_arc=self.complex_min_arc,
                    max_depth=self.complex_max_depth,
                )
            else:
                segments = [poly]
            for seg in segments:
                seg = np.asarray(seg, dtype=np.float64).reshape(-1, 3)
                if seg.shape[0] < 2:
                    continue
                a = get_vertex_id(seg[0].astype(np.float32))
                b = get_vertex_id(seg[-1].astype(np.float32))
                if a is None or b is None or a == b:
                    continue
                edge_index.append((a, b))
                edge_points.append(
                    _resample_polyline(seg, fmt.num_edge_points))

        for i in range(n_raw):
            if not (np.isfinite(start[i]).all() and np.isfinite(end[i]).all()):
                continue
            pts_full = _clean_polyline(raw_edge_points[i])
            if pts_full.shape[0] < 2:
                continue

            # Closed-loop edge: endpoints (near-)coincident but the polyline
            # traverses a real arc (length >> chord). The endpoint-anchored curve
            # frame is undefined for these, so split the loop into two open arcs
            # at the far point -- inserting a midpoint vertex turns it into a
            # faithful 2-cycle. Each arc then goes through the generic complex
            # splitter (a big arc may still need further subdivision).
            chord = float(np.linalg.norm(end[i] - start[i]))
            if self.split_loops and chord < self.loop_close_tol:
                arclen = _polyline_length(pts_full)
                if arclen > self.loop_min_arc and arclen > 4.0 * chord:
                    m = _max_deviation_index(pts_full)
                    if m > 0:
                        add_open_polyline(pts_full[: m + 1])
                        add_open_polyline(pts_full[m:])
                        continue

            add_open_polyline(pts_full)

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
    ) -> dict[str, Any]:
        def _t(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(a))

        return {
            "shape_id": shape_id,
            "point_cloud": _t(pc),
            "vertices": _t(vertices),
            "edge_index": _t(edge_index),
            "edge_points": _t(edge_points),
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
            }
            for s in samples
        ]
    return batch


def unbatch_wireframe_graphs(batch: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a collated batch back into per-shape GT wireframe dicts.

    Slices the packed ``point_cloud`` by ``pc_offset`` and pairs each shape's
    points with its ``gt_wireframes`` entry. Returns a list of length ``B`` with
    ``{shape_id, point_cloud, vertices, edge_index, edge_points}`` tensors (the
    point cloud is empty for prediction batches that carry no GT). Used by the
    evaluation / visualisation scripts.
    """
    b = int(batch.get("num_graphs", len(batch.get("shape_id", []))))
    shape_ids = batch.get("shape_id", [None] * b)
    pc = batch.get("point_cloud")
    offset = batch.get("pc_offset")
    graphs = batch.get("gt_wireframes")
    out: list[dict[str, Any]] = []
    for s in range(b):
        if pc is not None and offset is not None:
            lo = int(offset[s - 1]) if s > 0 else 0
            hi = int(offset[s])
            pts = pc[lo:hi]
        else:
            pts = pc.new_zeros((0, 3)) if pc is not None else None
        entry: dict[str, Any] = {"shape_id": shape_ids[s], "point_cloud": pts}
        if graphs is not None:
            entry.update(
                vertices=graphs[s]["vertices"],
                edge_index=graphs[s]["edge_index"],
                edge_points=graphs[s]["edge_points"],
            )
        out.append(entry)
    return out


__all__ = [
    "GraphFormat",
    "WireframeGraphDataset",
    "PointCloudDataset",
    "collate_ae_batch",
    "unbatch_wireframe_graphs",
    "list_npz",
    "make_split",
    "save_split",
    "load_split",
    "resolve_split_files",
]
