"""Curve canonicalisation + latent (de)coding helpers for the joint branch.

The joint vertex+edge decoder represents each edge's *shape* as a 12-d curve VAE
latent (``latent_channels * latent_len``). Training and reconstruction both need
to move between three representations of a curve:

  * **world curve** ``(P, 3)`` -- the raw ordered polyline (its first / last
    points are the two endpoints), as stored in the GT wireframe;
  * **canonical curve** ``(P, 3)`` -- endpoints pinned to ``[-1,0,0]`` /
    ``[1,0,0]`` by :func:`~src.models.vae.geometry.normalize_curves`, so the VAE
    only models the intrinsic shape;
  * **curve latent** ``(D,)`` -- the VAE token latent of the canonical curve.

To make the canonical frame's start / end well defined (and identical at train
and inference time) every curve is first **oriented** by a deterministic
lexicographic rule on its two endpoints: the lexicographically-smaller endpoint
becomes the curve start. At reconstruction time the predicted endpoints are
sorted by the same rule before :func:`~src.models.vae.recon_utils.denorm_curves`
maps the decoded canonical curve back onto them, so orientation is consistent.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from einops import rearrange

from .geometry import normalize_curves


# ----------------------------------------------------------------------
# orientation (deterministic lexicographic endpoint rule)
# ----------------------------------------------------------------------
def lexicographic_flip_mask(endpoints_a: torch.Tensor,
                            endpoints_b: torch.Tensor) -> torch.Tensor:
    """Bool mask ``(E,)``: ``True`` where ``a`` > ``b`` lexicographically.

    Compares two batches of 3-D endpoints coordinate-by-coordinate; ``True``
    means the curve should be flipped so its lexicographically-smaller endpoint
    becomes the start.
    """
    a = endpoints_a
    b = endpoints_b
    gt = a[:, 0] > b[:, 0]
    eq0 = a[:, 0] == b[:, 0]
    gt = gt | (eq0 & (a[:, 1] > b[:, 1]))
    eq1 = eq0 & (a[:, 1] == b[:, 1])
    gt = gt | (eq1 & (a[:, 2] > b[:, 2]))
    return gt


def orient_curves(edge_points: torch.Tensor) -> torch.Tensor:
    """Orient curves ``(E, P, 3)`` so the start endpoint is lexicographic-min."""
    if edge_points.shape[0] == 0:
        return edge_points
    flip = lexicographic_flip_mask(edge_points[:, 0], edge_points[:, -1])
    if flip.any():
        edge_points = edge_points.clone()
        edge_points[flip] = torch.flip(edge_points[flip], dims=[1])
    return edge_points


# ----------------------------------------------------------------------
# canonicalisation
# ----------------------------------------------------------------------
# Minimum endpoint chord length below which a curve is treated as degenerate
# (closed / near-closed loop). The endpoint-anchored canonical frame
# (``normalize_curves`` scales by ``target_chord / original_chord``) blows up to
# ~1e6 as the chord -> 0, which then explodes the curve-VAE inputs / KL / latent
# losses and ultimately NaNs the whole network. Such curves cannot be expressed
# in this two-endpoint frame anyway, so we substitute a benign straight segment.
_MIN_CHORD = 1e-3
# Hard clip on canonical coordinates: legitimate curves live around [-1, 1] with
# some bulge; this caps any residual blow-up feeding the VAE.
_CANON_CLIP = 4.0


def _straight_canonical(p: int) -> np.ndarray:
    """Safe canonical polyline ``(P, 3)``: a straight line ``[-1,0,0]->[1,0,0]``."""
    line = np.zeros((p, 3), dtype=np.float32)
    line[:, 0] = np.linspace(-1.0, 1.0, p, dtype=np.float32)
    return line


def canonical_curves(edge_points: torch.Tensor) -> torch.Tensor:
    """World curves ``(E, P, 3)`` -> oriented canonical curves ``(E, P, 3)``.

    Orients each curve by the lexicographic endpoint rule, then maps it into the
    canonical frame (endpoints -> ``[-1,0,0]`` / ``[1,0,0]``). Runs the numpy
    ``normalize_curves`` under the hood and returns a tensor on the input device.

    Degenerate (closed / near-closed) curves -- endpoint chord ``< _MIN_CHORD``
    -- are replaced by a straight canonical segment (their endpoint-anchored
    frame is ill-defined and would otherwise explode), and the output is clipped
    + sanitised so a single bad curve can never inject NaN/Inf into the VAE.
    """
    if edge_points.shape[0] == 0:
        return edge_points.new_zeros((0, edge_points.shape[1], 3))
    oriented = orient_curves(edge_points)
    device = oriented.device
    p = oriented.shape[1]
    arr = oriented.detach().cpu().numpy().astype(np.float64)

    chord = np.linalg.norm(arr[:, -1] - arr[:, 0], axis=-1)      # (E,)
    degen = chord < _MIN_CHORD

    norm = np.empty((arr.shape[0], p, 3), dtype=np.float32)
    good = ~degen
    if good.any():
        norm[good] = normalize_curves(arr[good]).astype(np.float32)
    if degen.any():
        norm[degen] = _straight_canonical(p)

    norm = np.nan_to_num(norm, nan=0.0, posinf=_CANON_CLIP, neginf=-_CANON_CLIP)
    np.clip(norm, -_CANON_CLIP, _CANON_CLIP, out=norm)
    return torch.from_numpy(norm).to(device)


def normalized_curves_from_batch(
    gt_wireframes: list[dict[str, Any]],
    device: torch.device | None = None,
) -> list[torch.Tensor]:
    """Per-shape oriented canonical curves from a list of GT wireframes.

    Returns a list of length ``B``; entry ``i`` is the canonical curve tensor
    ``(E_i, P, 3)`` (empty ``(0, P, 3)`` for edge-less shapes). The orientation /
    canonicalisation matches the reconstruction path so the curve VAE sees the
    same frame at train and inference time.
    """
    out: list[torch.Tensor] = []
    for g in gt_wireframes:
        ep = g["edge_points"]
        if not torch.is_tensor(ep):
            ep = torch.as_tensor(ep)
        ep = ep.float()
        if device is not None:
            ep = ep.to(device)
        if ep.numel() == 0:
            p = ep.shape[1] if ep.ndim == 3 and ep.shape[1] > 0 else 1
            out.append(ep.new_zeros((0, p, 3)))
            continue
        out.append(canonical_curves(ep.reshape(ep.shape[0], -1, 3)))
    return out


# ----------------------------------------------------------------------
# latent <-> curve
# ----------------------------------------------------------------------
def encode_curve_mu(curve_vae: torch.nn.Module,
                    canonical: torch.Tensor) -> torch.Tensor:
    """Posterior mean latent ``(E, D)`` of canonical curves ``(E, P, 3)``.

    ``D = latent_channels * latent_len`` matches the edge head's curve-latent
    width. Returns an empty ``(0, D)`` when there are no curves.
    """
    ch = int(curve_vae.config.latent_channels)
    ll = int(curve_vae.latent_len)
    if canonical.shape[0] == 0:
        return canonical.new_zeros((0, ch * ll))
    x = rearrange(canonical, "e p c -> e c p")          # (E, 3, P)
    posterior = curve_vae.encode(x)                      # moments (E, 2C, L)
    return rearrange(posterior.mode(), "e c l -> e (c l)")


def decode_curve_latent(
    curve_vae: torch.nn.Module,
    curve_latent: torch.Tensor,
    num_points: int = 32,
    pin_endpoints: bool = False,
) -> torch.Tensor:
    """Per-edge curve latent ``(M, D)`` -> canonical polyline ``(M, P, 3)``.

    Reshapes ``D = latent_channels * latent_len`` to the curve-VAE latent layout
    and decodes at a uniform parametric grid ``t in [0, 1]``. Output endpoints
    sit near ``[-1,0,0]`` / ``[1,0,0]``; ``pin_endpoints`` hard-pins them (used
    at reconstruction before denormalising onto the predicted vertices).
    """
    if curve_latent.shape[0] == 0:
        return curve_latent.new_zeros((0, num_points, 3))
    ch = int(curve_vae.config.latent_channels)
    z = rearrange(curve_latent, "m (c l) -> m c l", c=ch)
    t = torch.linspace(0.0, 1.0, num_points, device=curve_latent.device,
                       dtype=curve_latent.dtype)
    t = t.unsqueeze(0).expand(z.shape[0], -1)
    dec = curve_vae.decode(z, t)                         # (M, 3, P)
    if pin_endpoints:
        dec[:, :, 0] = torch.tensor([-1.0, 0.0, 0.0], device=dec.device,
                                    dtype=dec.dtype)
        dec[:, :, -1] = torch.tensor([1.0, 0.0, 0.0], device=dec.device,
                                     dtype=dec.dtype)
    return rearrange(dec, "m c p -> m p c")


__all__ = [
    "lexicographic_flip_mask",
    "orient_curves",
    "canonical_curves",
    "normalized_curves_from_batch",
    "encode_curve_mu",
    "decode_curve_latent",
]
