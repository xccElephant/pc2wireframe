"""Shared differentiable curve samplers (line / arc / cubic).

A wireframe edge is parameterised by **four on-curve points**: its two
endpoints ``a`` / ``b`` and two interior anchors ``q1`` / ``q2`` at the
arc-length fractions ``t = 1/3`` and ``t = 2/3``. Given a per-edge curve type
(``0=line`` / ``1=arc`` / ``2=bezier``) these four points define the curve:

  * **line**   -- straight segment ``a -> b`` (anchors ignored);
  * **arc**    -- circular arc through ``a -> q1 -> b`` (q1 = the midpoint
                  control; falls back to the line when (near-)collinear);
  * **bezier** -- the unique cubic Bezier passing through ``a, q1, q2, b``.

These torch samplers are shared by the training loss
(:mod:`src.module`) and the decode step (:mod:`src.recon.wireframe`) so the
geometry is consistent end to end. All take a leading batch of edges
(``(M, 3)`` control points) and return ``(M, num, 3)`` polylines.
"""
from __future__ import annotations

import math

import torch


def sample_line(a: torch.Tensor, b: torch.Tensor, num: int) -> torch.Tensor:
    """Sample ``num`` points on the straight segment ``a -> b``."""
    t = torch.linspace(0.0, 1.0, num, device=a.device, dtype=a.dtype).view(1, num, 1)
    return a[:, None, :] * (1.0 - t) + b[:, None, :] * t


def sample_bezier(
    a: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor, b: torch.Tensor, num: int
) -> torch.Tensor:
    """Cubic curve interpolating the four on-curve points ``a, q1, q2, b``.

    ``q1`` / ``q2`` are the t=1/3 / t=2/3 coordinates, so the Bezier control
    points are solved so the curve passes through all four (not the usual
    control-polygon interpretation).
    """
    big_a = 27.0 * q1 - 8.0 * a - b
    big_b = 27.0 * q2 - a - 8.0 * b
    p1 = (2.0 * big_a - big_b) / 18.0
    p2 = (2.0 * big_b - big_a) / 18.0
    t = torch.linspace(0.0, 1.0, num, device=a.device, dtype=a.dtype).view(1, num, 1)
    mt = 1.0 - t
    return (
        mt ** 3 * a[:, None, :]
        + 3.0 * mt ** 2 * t * p1[:, None, :]
        + 3.0 * mt * t ** 2 * p2[:, None, :]
        + t ** 3 * b[:, None, :]
    )


def sample_arc(
    a: torch.Tensor, m: torch.Tensor, b: torch.Tensor, num: int
) -> torch.Tensor:
    """Sample the circular arc through three points ``a -> m -> b``.

    Falls back to the straight segment ``a -> b`` when the three points are
    (near-)collinear. Every divisor uses the "safe-denominator" trick (replace
    the degenerate value by ``1`` *before* dividing) so the arc branch stays
    finite even for collinear inputs -- otherwise ``inf``/``nan`` from the
    unselected branch would still poison the gradient through ``torch.where``.
    """
    eps = 1e-6
    aa = a - m
    bb = b - m
    cr = torch.cross(aa, bb, dim=-1)                       # (M, 3)
    cr_n2 = (cr * cr).sum(-1, keepdim=True)                # (M, 1)
    collinear = cr_n2 < eps                                # (M, 1) bool
    safe_cr_n2 = torch.where(collinear, torch.ones_like(cr_n2), cr_n2)
    alpha = (aa * aa).sum(-1, keepdim=True)
    beta = (bb * bb).sum(-1, keepdim=True)
    center = m + torch.cross(alpha * bb - beta * aa, cr, dim=-1) / (
        2.0 * safe_cr_n2)
    ua = a - center
    r = ua.norm(dim=-1, keepdim=True)                     # (M, 1)
    u = ua / r.clamp_min(eps)
    cr_norm = cr.norm(dim=-1, keepdim=True)
    safe_cr_norm = torch.where(
        collinear, torch.ones_like(cr_norm), cr_norm.clamp_min(eps))
    nrm = cr / safe_cr_norm
    v = torch.cross(nrm, u, dim=-1)

    def _ang(p: torch.Tensor) -> torch.Tensor:
        d = p - center
        return torch.atan2((d * v).sum(-1), (d * u).sum(-1))  # (M,)

    two_pi = 2.0 * math.pi
    m_ang = _ang(m) % two_pi
    b_ang = _ang(b) % two_pi
    sweep = torch.where(m_ang <= b_ang, b_ang, b_ang - two_pi)   # (M,)
    t = torch.linspace(0.0, 1.0, num, device=a.device, dtype=a.dtype)  # (num,)
    theta = sweep[:, None] * t[None, :]                  # (M, num)
    pts = center[:, None, :] + r[:, None, :] * (
        torch.cos(theta)[..., None] * u[:, None, :]
        + torch.sin(theta)[..., None] * v[:, None, :]
    )
    return torch.where(
        collinear[:, :, None], sample_line(a, b, num), pts)


def sample_curve_by_type(
    a: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    b: torch.Tensor,
    ctype: torch.Tensor,
    num: int,
) -> torch.Tensor:
    """Sample each edge's curve by its (integer) ``ctype`` 0=line/1=arc/2=bezier.

    Each parameterisation is evaluated **only** on the edges that select it
    (via boolean indexing) and scattered back into the output. This is both
    cheaper and -- crucially -- isolates the autograd graphs: an arc/Bezier
    blow-up on its own edges can no longer leak ``inf``/``nan`` gradients into
    edges of a different type (the failure mode of the previous all-branches +
    ``torch.where`` implementation). ``a, q1, q2, b`` are ``(M, 3)``;
    ``ctype`` is ``(M,)``.
    """
    m = a.shape[0]
    if m == 0:
        return a.new_zeros((0, num, 3))
    ctype = ctype.reshape(-1)
    out = a.new_zeros((m, num, 3))

    arc_mask = ctype == 1
    bez_mask = ctype == 2
    line_mask = ~(arc_mask | bez_mask)

    if line_mask.any():
        idx = line_mask.nonzero(as_tuple=False).reshape(-1)
        out = out.index_copy(0, idx, sample_line(a[idx], b[idx], num))
    if arc_mask.any():
        idx = arc_mask.nonzero(as_tuple=False).reshape(-1)
        out = out.index_copy(0, idx, sample_arc(a[idx], q1[idx], b[idx], num))
    if bez_mask.any():
        idx = bez_mask.nonzero(as_tuple=False).reshape(-1)
        out = out.index_copy(
            0, idx, sample_bezier(a[idx], q1[idx], q2[idx], b[idx], num))
    return out


__all__ = [
    "sample_line",
    "sample_arc",
    "sample_bezier",
    "sample_curve_by_type",
]
