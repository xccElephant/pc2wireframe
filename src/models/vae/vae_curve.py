"""Per-curve attention/token VAE (pure PyTorch, no diffusers).

The curve is assumed to already live in the canonical frame (the pipeline pins
endpoints to ``[-1,0,0]`` / ``[1,0,0]`` via ``geometry.normalize_curves`` and
inverts it with ``recon_utils.denorm_curves``), so this module only models the
intrinsic shape.

  * **Encoder** -- ``L`` learnable latent queries attend (self + cross) to the
    Fourier-embedded (+ tangent) curve points via a ``nn.TransformerDecoder``;
    a head emits ``2 * latent_channels`` (mean/logvar) per token, reshaped to
    ``(B, latent_channels, L)`` to keep the ``(b c l)`` latent layout and the
    12-d wireframe-VAE contract (``latent_channels * L``).
  * **Decoder** -- the ``L`` latent tokens self-attend (``nn.TransformerEncoder``);
    parametric ``t`` queries then cross-attend to them (``nn.TransformerDecoder``)
    and an MLP predicts a residual on top of the linear baseline ``[-1..1,0,0]``.

Absorbed CLR-Wire perks: tangent input features, continuous-``t`` decoding
(arbitrary query points, random ``t`` during training) and the length-weighted
reconstruction loss. Kept baseline strengths: token latent, line-residual prior
and an explicit endpoint anchor.

The public surface mirrors the old CLR-Wire ``AutoencoderKL1D`` so it stays
drop-in for ``packing.py`` / ``module.py``: ``encode(x).latent_dist`` (with
``.mode() / .std / .sample()``), ``decode(z, t).sample``,
``forward(..., return_loss=True) -> (loss, parts)``, ``.config.latent_channels``
and ``.sample_points_num``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from .torch_tools import interpolate_1d, calculate_polyline_lengths
from .geometry import point_seq_tangent


# ----------------------------------------------------------------------
# small helpers / lightweight output containers (replace diffusers)
# ----------------------------------------------------------------------
def _num_heads(d_model: int, preferred: int = 8) -> int:
    """Largest head count <= ``preferred`` that divides ``d_model``."""
    for h in (preferred, 4, 2, 1):
        if d_model % h == 0:
            return h
    return 1


def _fourier_embed(x: torch.Tensor, num_bands: int) -> torch.Tensor:
    """Concatenate ``[x, sin(2^k pi x), cos(2^k pi x)]`` along the last dim."""
    bands = 2.0 ** torch.arange(num_bands, device=x.device, dtype=x.dtype)
    out = x.unsqueeze(-1) * bands * math.pi
    return torch.cat([x, out.sin().flatten(-2), out.cos().flatten(-2)], dim=-1)


class GaussianLatent:
    """Diagonal Gaussian posterior over moments ``(B, 2C, L)``."""

    def __init__(self, moments: torch.Tensor):
        self.mean, self.logvar = moments.chunk(2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        noise = torch.randn(
            self.mean.shape, generator=generator,
            device=self.mean.device, dtype=self.mean.dtype)
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        """Per-sample KL to ``N(0, I)``, summed over latent dims ``(B,)``."""
        return 0.5 * torch.sum(
            self.mean.pow(2) + self.var - 1.0 - self.logvar, dim=[1, 2])


@dataclass
class EncoderOutput:
    latent_dist: GaussianLatent


@dataclass
class DecoderOutput:
    sample: torch.Tensor


def _resolve_dims(
    block_out_channels: Tuple[int, ...],
    down_block_types: Tuple[str, ...],
    sample_points_num: int,
) -> Tuple[int, int]:
    """``(d_model, latent_len)``; latent_len mirrors the old conv downsampling."""
    d_model = int(block_out_channels[-1])
    latent_len = max(1, sample_points_num // (2 ** len(down_block_types)))
    return d_model, latent_len


# ----------------------------------------------------------------------
# encoder / decoder modules
# ----------------------------------------------------------------------
class CurveEncoder(nn.Module):
    """Curve points ``(B, 3, U)`` -> moments ``(B, 2*latent_channels, L)``."""

    def __init__(
        self,
        d_model: int,
        latent_channels: int,
        latent_tokens: int,
        sample_points_num: int,
        num_layers: int = 2,
        num_fourier_bands: int = 6,
        use_tangent: bool = True,
        nhead: Optional[int] = None,
    ):
        super().__init__()
        self.U = sample_points_num
        self.use_tangent = use_tangent
        self.num_fourier_bands = num_fourier_bands
        nhead = nhead or _num_heads(d_model)

        in_feat = 3 + 2 * 3 * num_fourier_bands + (3 if use_tangent else 0)
        self.point_in = nn.Linear(in_feat, d_model)
        self.point_pos = nn.Parameter(torch.randn(1, sample_points_num, d_model) * 0.02)
        self.query = nn.Parameter(torch.randn(1, latent_tokens, d_model) * 0.02)
        # self-attn(query) + cross-attn(query -> points) + FFN per layer.
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward=d_model * 2, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True),
            num_layers)
        self.to_moments = nn.Linear(d_model, 2 * latent_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.U:  # (B, 3, U_in) -> fixed encoder length
            x = F.interpolate(x, size=self.U, mode="linear", align_corners=True)
        pts = rearrange(x, "b c n -> b n c")  # (B, U, 3)

        feat = _fourier_embed(pts, self.num_fourier_bands)
        if self.use_tangent:
            feat = torch.cat([feat, point_seq_tangent(pts)], dim=-1)
        memory = self.point_in(feat) + self.point_pos

        q = self.query.expand(memory.shape[0], -1, -1)
        q = self.transformer(tgt=q, memory=memory)  # (B, L, d)
        moments = self.to_moments(q)                # (B, L, 2C)
        return rearrange(moments, "b l c -> b c l")  # (B, 2C, L)


class CurveDecoder(nn.Module):
    """Latent ``(B, latent_channels, L)`` + ``t`` ``(B, P)`` -> curve ``(B, 3, P)``."""

    def __init__(
        self,
        d_model: int,
        latent_channels: int,
        latent_tokens: int,
        num_layers: int = 2,
        num_fourier_bands: int = 6,
        nhead: Optional[int] = None,
    ):
        super().__init__()
        self.num_fourier_bands = num_fourier_bands
        nhead = nhead or _num_heads(d_model)

        self.lat_proj = nn.Linear(latent_channels, d_model)
        self.lat_pos = nn.Parameter(torch.randn(1, latent_tokens, d_model) * 0.02)
        self.lat_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=d_model * 2, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True),
            num_layers)
        self.t_in = nn.Linear(1 + 2 * num_fourier_bands, d_model)
        # self-attn(t queries) + cross-attn(t -> latent) + FFN per layer.
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward=d_model * 2, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True),
            num_layers)
        self.point_out = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3))

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = rearrange(z, "b c l -> b l c")
        h = self.lat_proj(h) + self.lat_pos
        h = self.lat_self(h)  # (B, L, d)

        t_feat = _fourier_embed(t.unsqueeze(-1), self.num_fourier_bands)  # (B, P, 1+2F)
        t_tok = self.t_in(t_feat)
        decoded = self.transformer(tgt=t_tok, memory=h)  # (B, P, d)
        residual = self.point_out(decoded)               # (B, P, 3)

        baseline = torch.zeros_like(residual)
        baseline[..., 0] = 2.0 * t - 1.0                 # line (-1,0,0)->(1,0,0)
        out = baseline + 0.5 * residual
        return rearrange(out, "b p c -> b c p")           # (B, 3, P)


# ----------------------------------------------------------------------
# full VAE + split encode/decode variants
# ----------------------------------------------------------------------
class AutoencoderKL1D(nn.Module):
    """Per-curve attention/token VAE (pure PyTorch, CLR-Wire-compatible API)."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownBlock1D",),
        up_block_types: Tuple[str, ...] = ("UpBlock1D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_points_num: int = 64,
        kl_weight: float = 1e-6,
        num_fourier_bands: int = 6,
        use_tangent: bool = True,
    ):
        super().__init__()
        d_model, latent_len = _resolve_dims(
            block_out_channels, down_block_types, sample_points_num)
        self.latent_len = latent_len
        self.sample_points_num = sample_points_num
        self.kl_weight = kl_weight
        # ``packing.py`` reads ``curve_vae.config.latent_channels``.
        self.config = SimpleNamespace(
            in_channels=in_channels, out_channels=out_channels,
            latent_channels=latent_channels, sample_points_num=sample_points_num,
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
            block_out_channels=tuple(block_out_channels),
            layers_per_block=layers_per_block, kl_weight=kl_weight,
            num_fourier_bands=num_fourier_bands, use_tangent=use_tangent,
        )

        self.encoder = CurveEncoder(
            d_model=d_model, latent_channels=latent_channels,
            latent_tokens=latent_len, sample_points_num=sample_points_num,
            num_layers=layers_per_block, num_fourier_bands=num_fourier_bands,
            use_tangent=use_tangent)
        self.decoder = CurveDecoder(
            d_model=d_model, latent_channels=latent_channels,
            latent_tokens=latent_len, num_layers=layers_per_block,
            num_fourier_bands=num_fourier_bands)

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor, return_dict: bool = True):
        posterior = GaussianLatent(self.encoder(x))
        if not return_dict:
            return (posterior,)
        return EncoderOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor, t: torch.Tensor, return_dict: bool = True):
        dec = self.decoder(z, t)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def _sample_t(self, bs: int, device: torch.device) -> torch.Tensor:
        """Random sorted ``t`` with the endpoints pinned to ``{0, 1}``."""
        t = torch.rand(bs, self.sample_points_num, device=device)
        t, _ = torch.sort(t, dim=-1)
        t[:, 0] = 0.0
        t[:, -1] = 1.0
        return t

    def forward(
        self,
        data: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        return_loss: bool = False,
        **kwargs,
    ):
        data = rearrange(data, "b n c -> b c n")  # (B, 3, U) for the encoder
        bs = data.shape[0]

        posterior = self.encode(data).latent_dist
        z = posterior.sample(generator) if sample_posterior else posterior.mode()

        if t is None:
            t = self._sample_t(bs, data.device)
        else:
            assert t.shape[1] == self.sample_points_num, (
                "t must have sample_points_num columns")

        dec = self.decode(z, t).sample  # (B, 3, P)

        if not return_dict:
            return (dec,)

        if return_loss:
            kl_loss = posterior.kl().mean()

            gt_samples = interpolate_1d(t, data)  # (B, 3, P) from full-res curve

            curves = rearrange(data, "b c n -> b n c")
            lengths = calculate_polyline_lengths(curves).clamp(min=2.0, max=math.pi * 10)
            weights = torch.log(lengths + 0.2)  # down-weight long polylines

            per_curve = F.mse_loss(dec, gt_samples, reduction="none").mean(dim=[1, 2])
            recon_loss = (per_curve * weights).mean()
            endpoint_loss = F.l1_loss(dec[:, :, [0, -1]], gt_samples[:, :, [0, -1]])
            recon_loss = recon_loss + 0.5 * endpoint_loss

            loss = recon_loss + self.kl_weight * kl_loss
            return loss, dict(recon_loss=recon_loss, kl_loss=kl_loss)

        return DecoderOutput(sample=dec)


class AutoencoderKL1DFastEncode(nn.Module):
    """Encoder-only variant (kept for the wireframe VAE / inference)."""

    def __init__(
        self,
        in_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownBlock1D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_points_num: int = 16,
        num_fourier_bands: int = 6,
        use_tangent: bool = True,
        **kwargs,
    ):
        super().__init__()
        d_model, latent_len = _resolve_dims(
            block_out_channels, down_block_types, sample_points_num)
        self.config = SimpleNamespace(
            latent_channels=latent_channels, sample_points_num=sample_points_num)
        self.encoder = CurveEncoder(
            d_model=d_model, latent_channels=latent_channels,
            latent_tokens=latent_len, sample_points_num=sample_points_num,
            num_layers=layers_per_block, num_fourier_bands=num_fourier_bands,
            use_tangent=use_tangent)

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        posterior = GaussianLatent(self.encoder(x))
        if not return_dict:
            return (posterior,)
        return EncoderOutput(latent_dist=posterior)

    def forward(self, data: torch.Tensor, return_std: bool = False, **kwargs):
        data = rearrange(data, "b n c -> b c n")
        posterior = self.encode(data).latent_dist
        mu = posterior.mode()
        if return_std:
            return mu, posterior.std
        return mu


class AutoencoderKL1DFastDecode(nn.Module):
    """Decoder-only variant (kept for inference)."""

    def __init__(
        self,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpBlock1D",),
        down_block_types: Tuple[str, ...] = ("DownBlock1D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_points_num: int = 16,
        num_fourier_bands: int = 6,
        **kwargs,
    ):
        super().__init__()
        d_model, latent_len = _resolve_dims(
            block_out_channels, down_block_types, sample_points_num)
        self.sample_points_num = sample_points_num
        self.decoder = CurveDecoder(
            d_model=d_model, latent_channels=latent_channels,
            latent_tokens=latent_len, num_layers=layers_per_block,
            num_fourier_bands=num_fourier_bands)

    def forward(self, z: torch.Tensor, t: torch.Tensor = None, return_dict: bool = True):
        if t is None:
            bs = z.shape[0]
            t = torch.linspace(0, 1, self.sample_points_num, device=z.device).repeat(bs, 1)
        dec = self.decoder(z, t)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)
