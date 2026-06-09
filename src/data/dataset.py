"""Wireframe datasets for the PC2Wireframe task.

Two dataset flavours are provided:

``WireframeGraphDataset``
    Used for training / validation. Loads a pre-processed CAD wireframe NPZ
    (per-edge endpoints + resampled curve points) together with its matching
    surface point cloud. Endpoints are deduplicated into vertices by rounding
    coordinates to ``vertex_merge_tol``.

    Unlike the padded baseline, each sample is returned at its *native* size
    (no padding to ``max_*``), and a custom collate
    (:func:`collate_wireframe_graphs`) packs a list of samples into a single
    PyG-style "big graph" with batch-assignment vectors instead of padding.

    Each sample (``__getitem__``) yields *variable-length* tensors::

        shape_id:       str
        point_cloud:    (pc_num_points, 3)   float32   (fixed size)
        vertices:       (Vi, 3)              float32
        edge_index:     (Ei, 2)              int64     (LOCAL vertex ids)
        edge_points:    (Ei, U, 3)           float32
        edge_endpoints: (Ei, 6)              float32
        num_vertices, num_edges:             int64 scalars

    See :func:`collate_wireframe_graphs` for the packed batch layout.

``PointCloudDataset``
    Used for prediction / submission. The test split ships only point clouds
    (no ground-truth edges), so this dataset loads ``surface_points`` and
    returns ``{shape_id, point_cloud}`` for inference. Point clouds are fixed
    size, so this split uses the default (stacking) collate.
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


# ----------------------------------------------------------------------
# Graph format + file resolution
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class GraphFormat:
    """Graph (de)serialization format.

    ``max_vertices`` / ``max_edges`` are no longer padding targets; they act
    only as *safety caps*. A sample exceeding either cap is **skipped** (the
    loader simply moves on to the next file), not truncated, to avoid feeding
    the model a corrupted partial wireframe. Set a cap to ``<= 0`` to disable
    it (unlimited). Real samples keep their native size.
    """

    vertex_merge_tol: float = 1e-4
    max_vertices: int = 384
    max_edges: int = 1024
    num_edge_points: int = 32
    pc_num_points: int = 4096


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
    reproducible. Paths are kept exactly as globbed (relative to ``edge_dir``'s
    leading component, e.g. ``data/train/sample_edge/x.npz``).
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

    ``split`` may be:
        ``"train"`` / ``"val"`` -> the corresponding saved list;
        ``"all"`` / ``"trainval"`` -> train + val combined (use *everything*
        for training, i.e. no held-out validation).

    If ``split_path`` does not exist and ``auto_build`` is set, the split is
    built from ``edge_dir`` and saved to ``split_path`` first, so the on-disk
    split stays the single source of truth across runs / DDP ranks.
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
    """Loads point clouds + edge NPZ files into native-size wireframe graphs.

    File membership comes from a *pre-saved* split file (``split_path``); the
    on-the-fly shuffle has been removed so the split is fixed across runs.
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
        max_vertices: int = 384,
        max_edges: int = 1024,
        num_edge_points: int = 32,
        pc_num_points: int = 4096,
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

        # Native-size arrays (no padding). ``edge_index`` holds LOCAL vertex
        # ids in ``[0, nv)``; the collate fn offsets them into a packed graph.
        if nv > 0:
            vertices_arr = np.stack(vertices, axis=0).astype(np.float32)
        else:
            vertices_arr = np.zeros((0, 3), dtype=np.float32)
        if ne > 0:
            edge_index_arr = np.asarray(edge_index, dtype=np.int64)
            edge_points_arr = np.stack(edge_points, axis=0).astype(np.float32)
            edge_endpoints = np.concatenate(
                [
                    vertices_arr[edge_index_arr[:, 0]],
                    vertices_arr[edge_index_arr[:, 1]],
                ],
                axis=1,
            ).astype(np.float32)
        else:
            edge_index_arr = np.zeros((0, 2), dtype=np.int64)
            edge_points_arr = np.zeros(
                (0, fmt.num_edge_points, 3), dtype=np.float32)
            edge_endpoints = np.zeros((0, 6), dtype=np.float32)

        return {
            "vertices": vertices_arr,
            "edge_index": edge_index_arr,
            "edge_points": edge_points_arr,
            "edge_endpoints": edge_endpoints,
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
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
                # A cap <= 0 disables that check.
                if ne < self.min_edges:
                    continue
                if (mv > 0 and nv > mv) or (me > 0 and ne > me):
                    continue
                sample: dict[str, torch.Tensor | str] = {
                    "shape_id": os.path.splitext(
                        os.path.basename(edge_path))[0],
                    "point_cloud": torch.from_numpy(
                        self._load_point_cloud(edge_path)),
                }
                for key, value in graph.items():
                    if isinstance(value, np.ndarray):
                        sample[key] = torch.from_numpy(value)
                    else:
                        sample[key] = torch.tensor(value, dtype=torch.long)
                return sample
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
    each sample is simply ``{shape_id, point_cloud}``.
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        pc_path = self.files[int(idx) % len(self.files)]
        data = _load_npz_dict(pc_path)
        points = _get_npz_array(
            data, ("surface_points", "points", "point_cloud", "pc"))
        pc = _sample_points(points, self.pc_num_points)
        return {
            "shape_id": os.path.splitext(os.path.basename(pc_path))[0],
            "point_cloud": torch.from_numpy(pc),
        }


# ----------------------------------------------------------------------
# PyG-style packed collate (no padding)
# ----------------------------------------------------------------------
def collate_wireframe_graphs(
    samples: list[dict[str, torch.Tensor | str]],
) -> dict[str, torch.Tensor | list[str] | int]:
    """Pack a list of native-size graphs into one big batched graph.

    Instead of padding every sample up to ``max_*`` and stacking, the
    variable-length vertex/edge tensors are *concatenated* along their first
    dimension (exactly like ``torch_geometric.data.Batch``). Membership is
    recovered through batch-assignment vectors (``vertex_batch`` /
    ``edge_batch``) and CSR-style pointers (``vertex_ptr`` / ``edge_ptr``).
    Point clouds are fixed size so they are simply stacked.

    Returned dict (``B`` = batch size, ``Vsum`` = sum of per-sample vertices,
    ``Esum`` = sum of per-sample edges, ``U`` = points per curve)::

        point_cloud:    (B, pc_num_points, 3)  float32   (stacked)
        vertices:       (Vsum, 3)              float32   (concatenated)
        vertex_batch:   (Vsum,)                int64  sample id per vertex
        vertex_ptr:     (B + 1,)               int64  CSR offsets, cumsum(nv)
        edge_index:     (2, Esum)              int64  GLOBAL vertex ids
        edge_points:    (Esum, U, 3)           float32
        edge_endpoints: (Esum, 6)              float32
        edge_batch:     (Esum,)                int64  sample id per edge
        edge_ptr:       (B + 1,)               int64  CSR offsets, cumsum(ne)
        num_vertices:   (B,)                   int64  per-sample vertex count
        num_edges:      (B,)                   int64  per-sample edge count
        shape_id:       list[str]              length B
        num_graphs:     int                    == B

    ``edge_index`` follows PyG's ``(2, E)`` convention with columns
    ``[src, dst]`` already offset into the packed vertex space, so
    ``vertices[edge_index]`` indexes the correct global vertices directly.
    """
    batch_size = len(samples)
    num_edge_points = int(samples[0]["edge_points"].shape[1])

    point_cloud = torch.stack(
        [s["point_cloud"] for s in samples], dim=0)
    nv = torch.tensor(
        [int(s["num_vertices"]) for s in samples], dtype=torch.long)
    ne = torch.tensor(
        [int(s["num_edges"]) for s in samples], dtype=torch.long)

    vertices = torch.cat([s["vertices"] for s in samples], dim=0)
    edge_points = torch.cat([s["edge_points"] for s in samples], dim=0)
    edge_endpoints = torch.cat([s["edge_endpoints"] for s in samples], dim=0)

    arange = torch.arange(batch_size, dtype=torch.long)
    vertex_batch = torch.repeat_interleave(arange, nv)
    edge_batch = torch.repeat_interleave(arange, ne)
    vertex_ptr = torch.cat(
        [torch.zeros(1, dtype=torch.long), torch.cumsum(nv, dim=0)])
    edge_ptr = torch.cat(
        [torch.zeros(1, dtype=torch.long), torch.cumsum(ne, dim=0)])

    # Offset each sample's LOCAL edge_index into the packed vertex space and
    # transpose to PyG's (2, E) layout.
    offsets = vertex_ptr[:-1]
    eidx_parts = [
        s["edge_index"] + offsets[b]
        for b, s in enumerate(samples)
        if int(s["num_edges"]) > 0
    ]
    if eidx_parts:
        edge_index = torch.cat(eidx_parts, dim=0).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return {
        "point_cloud": point_cloud,
        "vertices": vertices,
        "vertex_batch": vertex_batch,
        "vertex_ptr": vertex_ptr,
        "edge_index": edge_index,
        "edge_points": edge_points,
        "edge_endpoints": edge_endpoints,
        "edge_batch": edge_batch,
        "edge_ptr": edge_ptr,
        "num_vertices": nv,
        "num_edges": ne,
        "shape_id": [str(s["shape_id"]) for s in samples],
        "num_graphs": batch_size,
    }


def unbatch_wireframe_graphs(
    batch: dict[str, torch.Tensor | list[str] | int],
) -> list[dict[str, torch.Tensor | str]]:
    """Split a packed batch back into per-sample graphs.

    Inverse of :func:`collate_wireframe_graphs`. Edge indices are rebased to
    LOCAL vertex ids so each returned graph is self-contained (handy for
    eval / submission export). ``point_cloud`` is sliced per sample.
    """
    vertex_ptr = batch["vertex_ptr"].tolist()
    edge_ptr = batch["edge_ptr"].tolist()
    edge_index = batch["edge_index"]
    out: list[dict[str, torch.Tensor | str]] = []
    for b in range(int(batch["num_graphs"])):
        v0, v1 = vertex_ptr[b], vertex_ptr[b + 1]
        e0, e1 = edge_ptr[b], edge_ptr[b + 1]
        local_edge_index = (edge_index[:, e0:e1] - v0).t().contiguous()
        out.append({
            "shape_id": batch["shape_id"][b],
            "point_cloud": batch["point_cloud"][b],
            "vertices": batch["vertices"][v0:v1],
            "edge_index": local_edge_index,
            "edge_points": batch["edge_points"][e0:e1],
            "edge_endpoints": batch["edge_endpoints"][e0:e1],
            "num_vertices": int(batch["num_vertices"][b]),
            "num_edges": int(batch["num_edges"][b]),
        })
    return out


__all__ = [
    "GraphFormat",
    "WireframeGraphDataset",
    "PointCloudDataset",
    "collate_wireframe_graphs",
    "unbatch_wireframe_graphs",
    "list_npz",
    "make_split",
    "save_split",
    "load_split",
    "resolve_split_files",
]
