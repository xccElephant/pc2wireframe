"""Evaluation metrics for the PC2Wireframe task.

The official server-side scorer (CCD / TA / VPE and the weighted final score)
is not published. This package provides principled, configurable proxies:

  * ``CCD`` -- curve chamfer distance (geometry of the curves);
  * ``TA``  -- topology accuracy (edge F1 after vertex matching);
  * ``VPE`` -- vertex position error (vertex chamfer);
  * final score = ``0.3*ccd_score + 0.4*TA + 0.3*vpe_score`` (higher = better),
    where geometric errors are mapped to ``(0, 1]`` via ``exp(-d / tau)``.

See :mod:`src.metrics.functional` for the numpy core and
:class:`src.metrics.wireframe_metrics.WireframeScore` for the epoch aggregator.
Calibrate ``tau`` / ``match_thresh`` against the official leaderboard.
"""
from .functional import (
    chamfer_distance,
    curve_chamfer_distance,
    distance_to_score,
    sample_wireframe_points,
    topology_accuracy,
    vertex_position_error,
)
from .wireframe_metrics import WireframeScore

__all__ = [
    "WireframeScore",
    "chamfer_distance",
    "sample_wireframe_points",
    "curve_chamfer_distance",
    "vertex_position_error",
    "topology_accuracy",
    "distance_to_score",
]
