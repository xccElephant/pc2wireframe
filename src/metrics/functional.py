"""Wireframe evaluation metrics, backed by PyTorch3D GPU kernels.

The official AICAD server-side scorer is **not** published, so these are
principled, configurable proxies that follow the competition's stated emphasis
(accurate curve geometry + strictly-preserved topology) and standard wireframe
reconstruction literature. Tune the score-mapping ``tau`` / ``match_thresh`` to
match the official leaderboard once a few public scores are known.

Nearest-neighbour queries (chamfer, vertex matching) use
``pytorch3d.ops.knn_points`` so everything runs batched on the GPU instead of
per-point scipy KD-trees.

Conventions (all wireframes are assumed normalised to roughly ``[-1, 1]^3``):
  * ``vertices``    ``(V, 3)``
  * ``edge_index``  ``(E, 2)`` integer endpoints into ``vertices``
  * ``edge_points`` ``(E, P, 3)`` optional dense samples along each curve; when
                    absent, edges are sampled as straight segments.

Metrics:
  * ``CCD`` (Curve Chamfer Distance) -- symmetric chamfer between dense point
    samples of predicted vs GT curves. Lower is better.
  * ``VPE`` (Vertex Position Error) -- symmetric vertex chamfer between
    predicted and GT vertices. Lower is better.
  * ``TA``  (Topology Accuracy) -- edge-level F1 after nearest-neighbour vertex
    matching (within ``match_thresh``). Higher is better, in ``[0, 1]``.
"""
from __future__ import annotations

import numpy as np
import torch


def _as_points(x, device: torch.device | str = "cpu") -> torch.Tensor:
    """Coerce an array / tensor to a ``(N, 3)`` float32 tensor on ``device``."""
    if isinstance(x, torch.Tensor):
        t = x.to(device=device, dtype=torch.float32)
    else:
        t = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    return t.reshape(-1, 3)


def nn_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Euclidean distance from each point in ``a`` to its nearest in ``b``."""
    from pytorch3d.ops import knn_points

    # knn_points returns *squared* distances; (1, N, 1) -> (N,)
    sq = knn_points(a[None], b[None], K=1).dists[0, :, 0]
    return sq.clamp_min(0.0).sqrt()


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------
def chamfer_distance(a, b, device: torch.device | str = "cpu") -> float:
    """Symmetric chamfer distance between two point sets via PyTorch3D.

    Thin wrapper over ``pytorch3d.loss.chamfer_distance``: the mean per-point
    *squared* L2 nearest-neighbour distance, summed over both directions.
    Returns ``inf`` if either set is empty.

    NOTE: this is the *squared* chamfer (PyTorch3D's convention), not the
    Euclidean mean-of-means used previously -- the ``*_tau`` score-mapping knobs
    must be recalibrated to this smaller scale.
    """
    from pytorch3d.loss import chamfer_distance as _p3d_chamfer

    a = _as_points(a, device)
    b = _as_points(b, device)
    if a.numel() == 0 or b.numel() == 0:
        return float("inf")
    loss, _ = _p3d_chamfer(a[None], b[None])
    return float(loss)


def sample_wireframe_points(
    vertices,
    edge_index,
    edge_points=None,
    num_per_edge: int = 32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return a dense ``(N, 3)`` point sampling of all curves in a wireframe."""
    if edge_points is not None and len(edge_points) > 0:
        return _as_points(edge_points, device)
    edge_index = torch.as_tensor(
        np.asarray(edge_index), dtype=torch.long, device=device
    ).reshape(-1, 2)
    if edge_index.numel() == 0:
        return torch.zeros((0, 3), dtype=torch.float32, device=device)
    verts = _as_points(vertices, device)
    t = torch.linspace(0.0, 1.0, num_per_edge, device=device)[None, :, None]
    a = verts[edge_index[:, 0]][:, None, :]  # (E, 1, 3)
    b = verts[edge_index[:, 1]][:, None, :]
    pts = a * (1.0 - t) + b * t              # (E, P, 3)
    return pts.reshape(-1, 3)


# ----------------------------------------------------------------------
# the three metrics
# ----------------------------------------------------------------------
def curve_chamfer_distance(
    pred: dict, gt: dict, num_per_edge: int = 32, device: torch.device | str = "cpu"
) -> float:
    """CCD: chamfer between dense curve samples of ``pred`` and ``gt``."""
    p = sample_wireframe_points(
        pred.get("vertices"), pred.get("edge_index"),
        pred.get("edge_points"), num_per_edge, device,
    )
    g = sample_wireframe_points(
        gt.get("vertices"), gt.get("edge_index"),
        gt.get("edge_points"), num_per_edge, device,
    )
    return chamfer_distance(p, g, device)


def vertex_position_error(
    pred: dict, gt: dict, device: torch.device | str = "cpu"
) -> float:
    """VPE: symmetric vertex chamfer between predicted and GT vertices."""
    return chamfer_distance(pred.get("vertices"), gt.get("vertices"), device)


def topology_accuracy(
    pred: dict,
    gt: dict,
    match_thresh: float = 0.1,
    device: torch.device | str = "cpu",
) -> float:
    """TA: edge-level F1 after nearest-neighbour vertex matching.

    Each predicted vertex is matched to its nearest GT vertex if within
    ``match_thresh`` (else unmatched). A predicted edge is *correct* iff both
    endpoints matched and the resulting GT-id pair is a GT edge. Precision is
    over predicted edges, recall over GT edges; ``TA`` is their F1.
    """
    gt_e = np.asarray(gt.get("edge_index"), dtype=np.int64).reshape(-1, 2)
    pred_e = np.asarray(pred.get("edge_index"), dtype=np.int64).reshape(-1, 2)
    pred_v = _as_points(pred.get("vertices"), device)
    gt_v = _as_points(gt.get("vertices"), device)

    if len(gt_e) == 0:
        return 1.0 if len(pred_e) == 0 else 0.0
    if len(pred_e) == 0 or pred_v.numel() == 0 or gt_v.numel() == 0:
        return 0.0

    from pytorch3d.ops import knn_points

    nn = knn_points(pred_v[None], gt_v[None], K=1)
    dist = nn.dists[0, :, 0].clamp_min(0.0).sqrt().cpu().numpy()
    idx = nn.idx[0, :, 0].cpu().numpy()
    mapped = np.where(dist <= match_thresh, idx, -1)  # pred vid -> gt vid / -1

    gt_set = {frozenset((int(i), int(j))) for i, j in gt_e if i != j}

    correct_pred = 0
    pred_pairs: set[frozenset] = set()
    for i, j in pred_e:
        gi, gj = int(mapped[i]), int(mapped[j])
        if gi < 0 or gj < 0 or gi == gj:
            continue
        pair = frozenset((gi, gj))
        if pair in gt_set:
            correct_pred += 1
            pred_pairs.add(pair)

    precision = correct_pred / max(1, len(pred_e))
    recall = len(pred_pairs & gt_set) / max(1, len(gt_set))
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ----------------------------------------------------------------------
# error -> [0, 1] score mapping (higher = better)
# ----------------------------------------------------------------------
def distance_to_score(distance: float, tau: float) -> float:
    """Map a non-negative error to a ``(0, 1]`` score via ``exp(-d / tau)``."""
    if not np.isfinite(distance):
        return 0.0
    return float(np.exp(-max(0.0, distance) / max(1e-9, tau)))


__all__ = [
    "nn_distances",
    "chamfer_distance",
    "sample_wireframe_points",
    "curve_chamfer_distance",
    "vertex_position_error",
    "topology_accuracy",
    "distance_to_score",
]
