"""Torchmetrics aggregator for the wireframe metrics + final weighted score.

Accumulates per-sample CCD / VPE / TA over a validation epoch (DDP-safe via
summed states) and computes the competition's weighted final score::

    final = w_ccd * ccd_score + w_ta * TA + w_vpe * vpe_score

where the geometric errors are mapped to ``(0, 1]`` scores with
``exp(-d / tau)`` so that the final score is in ``[0, 1]`` and *higher is
better* (suitable for ``ModelCheckpoint(mode="max")``).

Default weights follow the brief: ``(CCD, TA, VPE) = (0.3, 0.4, 0.3)`` -- TA is
weighted highest because the competition stresses topological correctness. The
``tau`` and ``match_thresh`` knobs are exposed so the proxy can be calibrated
to the (unpublished) official scorer.
"""
from __future__ import annotations

import torch
from torchmetrics import Metric

from .functional import (
    curve_chamfer_distance,
    distance_to_score,
    topology_accuracy,
    vertex_position_error,
)


class WireframeScore(Metric):
    """Aggregate CCD / TA / VPE and the weighted final score over an epoch."""

    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        *,
        w_ccd: float = 0.3,
        w_ta: float = 0.4,
        w_vpe: float = 0.3,
        ccd_tau: float = 0.1,
        vpe_tau: float = 0.1,
        match_thresh: float = 0.1,
        num_per_edge: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.w_ccd = float(w_ccd)
        self.w_ta = float(w_ta)
        self.w_vpe = float(w_vpe)
        self.ccd_tau = float(ccd_tau)
        self.vpe_tau = float(vpe_tau)
        self.match_thresh = float(match_thresh)
        self.num_per_edge = int(num_per_edge)

        self.add_state("ccd_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("vpe_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("ta_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("ccd_score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("vpe_score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: list[dict], targets: list[dict]) -> None:
        """Accumulate one batch.

        Args:
            preds:   list of predicted wireframes (``reconstruct`` output).
            targets: list of GT wireframes with the same keys
                     (``vertices``/``edge_index``/optional ``edge_points``).
        """
        import math

        device = self.device
        for pred, gt in zip(preds, targets):
            ccd = curve_chamfer_distance(pred, gt, self.num_per_edge, device)
            vpe = vertex_position_error(pred, gt, device)
            ta = topology_accuracy(pred, gt, self.match_thresh, device)

            self.ccd_sum += ccd if math.isfinite(ccd) else 0.0
            self.vpe_sum += vpe if math.isfinite(vpe) else 0.0
            self.ta_sum += float(ta)
            self.ccd_score_sum += distance_to_score(ccd, self.ccd_tau)
            self.vpe_score_sum += distance_to_score(vpe, self.vpe_tau)
            self.count += 1.0

    def compute(self) -> dict[str, torch.Tensor]:
        n = torch.clamp(self.count, min=1.0)
        ccd = self.ccd_sum / n
        vpe = self.vpe_sum / n
        ta = self.ta_sum / n
        ccd_score = self.ccd_score_sum / n
        vpe_score = self.vpe_score_sum / n
        score = self.w_ccd * ccd_score + self.w_ta * ta + self.w_vpe * vpe_score
        return {
            "score": score,
            "ccd": ccd,
            "ta": ta,
            "vpe": vpe,
            "ccd_score": ccd_score,
            "vpe_score": vpe_score,
        }


__all__ = ["WireframeScore"]
