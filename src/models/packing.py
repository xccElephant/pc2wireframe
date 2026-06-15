"""Graph (un)packing helpers shared by the wireframe pipeline.

The dataloader hands us a *packed* batch (PyG-style: variable-size graphs
concatenated along the first dim with CSR pointers). Two consumers need to turn
that into something else:

  * **stage 1 (curve VAE)** -- :func:`normalized_curves_from_batch` returns the
    per-curve canonical polylines ``(Esum, U, 3)`` (endpoints pinned to
    ``[-1,0,0]`` / ``[1,0,0]``) to autoencode.
  * **stage 2 (point cloud -> wireframe)** -- :func:`build_targets` slices the
    packed batch into per-sample GT node coords, oriented edge endpoint refs
    and canonical per-edge curves, the supervision targets for the Hungarian
    node / edge matching in ``criterion.py``.

:func:`decode_curve_latent` turns a predicted per-edge curve latent back into a
canonical polyline through the (frozen) curve VAE decoder; reconstruction code
then denormalises it onto the predicted endpoints with
:func:`~src.models.vae.recon_utils.denorm_curves`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from einops import rearrange


def normalized_curves_from_batch(batch: dict[str, Any]) -> torch.Tensor | None:
    """Per-curve canonical curves ``(Esum, U, 3)`` (endpoints -> [-1,0,0]/[1,0,0]).

    The collate fn precomputes ``edge_points_norm`` in the dataloader workers,
    so the common path is a zero-copy read of that tensor (already on device).
    Falls back to normalising ``edge_points`` on the fly for batches that lack
    the precomputed key. Returns ``None`` if the batch contains no edges.
    """
    ep_norm = batch.get("edge_points_norm")
    if ep_norm is not None:
        return ep_norm if ep_norm.shape[0] > 0 else None

    from .vae.geometry import normalize_curves

    ep = batch.get("edge_points")
    if ep is None or ep.shape[0] == 0:
        return None
    device = ep.device
    arr = ep.detach().cpu().numpy().astype(np.float64)
    norm = normalize_curves(arr).astype(np.float32)
    return torch.from_numpy(norm).to(device)


def build_targets(batch: dict[str, Any]) -> list[dict[str, torch.Tensor]]:
    """Slice a packed batch into per-sample wireframe supervision targets.

    Returns a list of length ``B``; each entry has (``nv`` GT vertices, ``ne``
    GT edges, ``U`` points per curve):

        ``node_coords``  ``(nv, 3)``     GT vertex coordinates
        ``edge_a``       ``(ne,)``       endpoint-A node id (edge "start")
        ``edge_b``       ``(ne,)``       endpoint-B node id (edge "end")
        ``edge_curve``   ``(ne, U, 3)``  canonical GT curve (start -> end order)

    The directed ``(start, end)`` convention comes straight from the dataset
    (curve points run start -> end, ``edge_points_norm`` is canonicalised in
    that order), so endpoint-A / curve orientation are mutually consistent and
    no re-normalisation is needed here.
    """
    device = batch["point_cloud"].device
    b = int(batch["num_graphs"])
    vptr = batch["vertex_ptr"].tolist()
    eptr = batch["edge_ptr"].tolist()
    vertices = batch["vertices"]
    edge_index = batch["edge_index"]           # (2, Esum) global vertex ids
    edge_curve = normalized_curves_from_batch(batch)

    out: list[dict[str, torch.Tensor]] = []
    for s in range(b):
        v0, v1 = vptr[s], vptr[s + 1]
        e0, e1 = eptr[s], eptr[s + 1]
        node_coords = vertices[v0:v1]
        if e1 > e0:
            edge_a = (edge_index[0, e0:e1] - v0).long()
            edge_b = (edge_index[1, e0:e1] - v0).long()
            curves = (
                edge_curve[e0:e1]
                if edge_curve is not None
                else torch.zeros(0, device=device)
            )
        else:
            edge_a = torch.zeros(0, dtype=torch.long, device=device)
            edge_b = torch.zeros(0, dtype=torch.long, device=device)
            curves = torch.zeros(0, device=device)
        out.append({
            "node_coords": node_coords,
            "edge_a": edge_a,
            "edge_b": edge_b,
            "edge_curve": curves,
        })
    return out


def decode_curve_latent(
    curve_vae: torch.nn.Module,
    curve_latent: torch.Tensor,
    num_points: int = 32,
    pin_endpoints: bool = False,
) -> torch.Tensor:
    """Per-edge curve latent ``(M, D)`` -> canonical polyline ``(M, P, 3)``.

    Reshapes ``D = latent_channels * latent_len`` to the curve-VAE latent layout
    and decodes at a uniform parametric grid ``t in [0, 1]``. The output lives in
    the canonical frame (endpoints near ``[-1,0,0]`` / ``[1,0,0]``); set
    ``pin_endpoints`` to hard-pin them (used at reconstruction time before
    denormalising onto the predicted vertices).
    """
    if curve_latent.shape[0] == 0:
        return curve_latent.new_zeros(0, num_points, 3)
    ch = int(curve_vae.config.latent_channels)
    z = rearrange(curve_latent, "m (c l) -> m c l", c=ch)
    t = torch.linspace(0.0, 1.0, num_points, device=curve_latent.device)
    t = t.unsqueeze(0).expand(z.shape[0], -1)
    dec = curve_vae.decode(z, t)               # (M, 3, P)
    if pin_endpoints:
        dec[:, :, 0] = torch.tensor([-1.0, 0.0, 0.0], device=dec.device)
        dec[:, :, -1] = torch.tensor([1.0, 0.0, 0.0], device=dec.device)
    return rearrange(dec, "m c p -> m p c")


__all__ = [
    "normalized_curves_from_batch",
    "build_targets",
    "decode_curve_latent",
]
