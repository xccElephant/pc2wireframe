"""Graph (un)packing helpers shared by the edge-query wireframe pipeline.

The dataloader (:func:`~src.data.dataset.collate_ae_batch`) hands us a packed
point cloud (``point_cloud (P_sum, 3)`` + ``pc_offset (B,)``) plus a Python list
of native-size GT wireframes under ``gt_wireframes`` (each a dict with
``vertices`` / ``edge_index`` / ordered ``edge_points (E, P, 3)``). Two consumers
turn that into supervision:

  * **stage 1 (curve VAE)** -- :func:`normalized_curves_from_batch` returns *all*
    of the batch's per-curve canonical polylines concatenated into a single
    ``(Esum, U, 3)`` tensor (endpoints pinned to ``[-1,0,0]`` / ``[1,0,0]``) to
    autoencode.
  * **stage 2 (point cloud -> wireframe)** -- :func:`build_targets` slices the
    batch into per-sample GT edge targets (oriented endpoint pairs + canonical
    per-edge curves), the supervision targets for the Hungarian edge-set
    matching in :mod:`src.models.edge_set_criterion`.

:func:`decode_curve_latent` turns a predicted per-edge curve latent back into a
canonical polyline through the (frozen) curve VAE decoder; reconstruction code
then denormalises it onto the predicted endpoints with
:func:`~src.models.vae.recon_utils.denorm_curves`.
"""
from __future__ import annotations

from typing import Any

import torch

from .vae.curve_packing import (
    canonical_curves,
    decode_curve_latent,
    orient_curves,
)


def _edge_points_list(batch: dict[str, Any]) -> list[torch.Tensor]:
    """Per-shape ``edge_points`` tensors ``(E_i, P, 3)`` from the batch."""
    graphs = batch.get("gt_wireframes")
    if graphs is None:
        return []
    out: list[torch.Tensor] = []
    for g in graphs:
        ep = g["edge_points"]
        if not torch.is_tensor(ep):
            ep = torch.as_tensor(ep)
        out.append(ep.float())
    return out


def normalized_curves_from_batch(
    batch: dict[str, Any]
) -> torch.Tensor | None:
    """All per-curve canonical curves of a batch -> ``(Esum, U, 3)`` (or None).

    Concatenates every shape's oriented canonical curves (endpoints pinned to
    ``[-1,0,0]`` / ``[1,0,0]`` by :func:`canonical_curves`). Returns ``None`` if
    the batch contains no edges. This is the stage-1 curve-VAE training target.
    """
    eps = _edge_points_list(batch)
    parts: list[torch.Tensor] = []
    for ep in eps:
        if ep.numel() == 0 or ep.shape[0] == 0:
            continue
        parts.append(canonical_curves(ep.reshape(ep.shape[0], -1, 3)))
    if not parts:
        return None
    return torch.cat(parts, dim=0)


def build_targets(
    batch: dict[str, Any], device: torch.device | None = None
) -> list[dict[str, torch.Tensor]]:
    """Slice a batch into per-sample edge-set supervision targets.

    Returns a list of length ``B``; each entry has (``ne`` GT edges, ``U`` points
    per curve):

        ``endpoints``   ``(ne, 2, 3)``   oriented endpoint pair (start = lex-min)
        ``edge_curve``  ``(ne, U, 3)``   canonical GT curve (start -> end order)

    The endpoint pair and the canonical curve are oriented by the *same*
    deterministic lexicographic rule (lexicographically-smaller endpoint first),
    so endpoint targets and curve orientation are mutually consistent.
    """
    eps = _edge_points_list(batch)
    out: list[dict[str, torch.Tensor]] = []
    for ep in eps:
        if ep.numel() == 0 or ep.shape[0] == 0:
            u = ep.shape[1] if ep.ndim == 3 and ep.shape[1] > 0 else 1
            empty = ep.new_zeros((0, u, 3))
            entry = {
                "endpoints": ep.new_zeros((0, 2, 3)),
                "edge_curve": empty,
            }
        else:
            ep = ep.reshape(ep.shape[0], -1, 3)
            oriented = orient_curves(ep)
            endpoints = torch.stack(
                [oriented[:, 0], oriented[:, -1]], dim=1)      # (ne, 2, 3)
            entry = {
                "endpoints": endpoints,
                "edge_curve": canonical_curves(ep),
            }
        if device is not None:
            entry = {k: v.to(device) for k, v in entry.items()}
        out.append(entry)
    return out


__all__ = [
    "normalized_curves_from_batch",
    "build_targets",
    "decode_curve_latent",
]
