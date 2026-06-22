"""Permutation-equivariant point-set velocity network (DiT) for Rectified Flow.

Predicts the flow-matching velocity ``v (B, N, 3)`` of a noised point set
``xt (B, N, 3)`` at time ``t (B,)``, conditioned on the point-cloud latent
``z (B, K, D)`` (here ``K=64`` tokens of ``D=64``).

Design (a point-set DiT, in the spirit of DiT / Point-E):

  * a per-point input projection ``3 -> d_model`` (no positional encoding -- the
    set is permutation-equivariant, the network must be too);
  * ``depth`` transformer blocks, each = global **self-attention** over the
    ``N`` points + **cross-attention** to the ``K`` latent tokens + an MLP, all
    modulated by **AdaLN-Zero** from a per-sample conditioning vector built from
    the sinusoidal time embedding plus the mean-pooled latent;
  * a final AdaLN + zero-initialised linear head projecting back to ``3``.

Attention uses ``torch.nn.MultiheadAttention`` with ``need_weights=False``,
which dispatches internally to ``scaled_dot_product_attention`` (Flash /
memory-efficient kernels). Self-attention over the ``N`` points therefore
costs ``O(N)`` memory rather than materialising the ``O(N^2)`` score matrix.
Optional ``grad_checkpoint`` trades compute for activation memory.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 1.0e4) -> torch.Tensor:
    """Sinusoidal embedding of a (continuous) timestep ``t (B,)`` -> ``(B, dim)``.

    Standard transformer / DiT sinusoidal embedding. ``t`` is the flow time in
    ``[0, 1]``; it is scaled by 1000 so the usual ``max_period`` frequency band
    covers the interval with good resolution.
    """
    t = t.float().reshape(-1) * 1000.0
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half, 1)
    )
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: ``x * (1 + scale) + shift`` (cond broadcast over N)."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """One DiT block: AdaLN-Zero self-attn + cross-attn + MLP.

    The conditioning vector ``c (B, d)`` produces 9 modulation tensors via a
    zero-initialised projection (AdaLN-Zero): (shift, scale, gate) for each of
    self-attention, cross-attention, and the MLP. Zero init makes every block
    an identity map at the start of training (gates = 0), which stabilises deep
    DiTs.
    """

    def __init__(self, dim: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            dim, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            dim, nhead, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        (
            sa_shift, sa_scale, sa_gate,
            ca_shift, ca_scale, ca_gate,
            mlp_shift, mlp_scale, mlp_gate,
        ) = self.adaln(c).chunk(9, dim=-1)

        h = _modulate(self.norm1(x), sa_shift, sa_scale)
        x = x + sa_gate.unsqueeze(1) * self.self_attn(
            h, h, h, need_weights=False)[0]
        q = _modulate(self.norm2(x), ca_shift, ca_scale)
        x = x + ca_gate.unsqueeze(1) * self.cross_attn(
            q, cond_tokens, cond_tokens, need_weights=False)[0]
        x = x + mlp_gate.unsqueeze(1) * self.mlp(
            _modulate(self.norm3(x), mlp_shift, mlp_scale)
        )
        return x


class _FinalLayer(nn.Module):
    """AdaLN + zero-initialised linear projection back to the point dim."""

    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(dim, out_dim)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln(c).chunk(2, dim=-1)
        return self.proj(_modulate(self.norm(x), shift, scale))


class RFPointSetVelocity(nn.Module):
    """Point-set DiT velocity field ``v(t, xt, z)``.

    Args:
        point_dim: per-point channels (``3`` = xyz).
        cond_dim: per-token latent channels ``D`` (default 256).
        d_model: transformer width.
        depth: number of DiT blocks.
        nhead: attention heads.
        mlp_ratio: MLP hidden expansion.
        dropout: dropout in attention / MLP.
        grad_checkpoint: gradient-checkpoint each block (saves activation
            memory at the cost of a recompute in the backward pass).
    """

    def __init__(
        self,
        point_dim: int = 3,
        cond_dim: int = 256,
        d_model: int = 384,
        depth: int = 8,
        nhead: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.point_dim = int(point_dim)
        self.grad_checkpoint = bool(grad_checkpoint)

        self.in_proj = nn.Linear(point_dim, d_model)
        self.cond_proj = nn.Linear(cond_dim, d_model)
        # Time embedding -> conditioning vector (DiT timestep MLP).
        self.t_embed_dim = d_model
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        # Global latent summary folded into the conditioning vector.
        self.cond_global = nn.Linear(d_model, d_model)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(d_model, nhead, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.final = _FinalLayer(d_model, point_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        def basic(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(basic)
        # AdaLN-Zero: zero the last layer of every modulation MLP so blocks
        # start as identity and the output head starts at zero.
        for block in self.blocks:
            nn.init.zeros_(block.adaln[-1].weight)
            nn.init.zeros_(block.adaln[-1].bias)
        nn.init.zeros_(self.final.adaln[-1].weight)
        nn.init.zeros_(self.final.adaln[-1].bias)
        nn.init.zeros_(self.final.proj.weight)
        nn.init.zeros_(self.final.proj.bias)

    def forward(
        self, t: torch.Tensor, xt: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Compute the velocity.

        Args:
            t:  ``(B,)`` flow time in ``[0, 1]`` (a scalar is broadcast).
            xt: ``(B, N, point_dim)`` noised point set.
            z:  ``(B, K, cond_dim)`` conditioning latent tokens.

        Returns:
            ``(B, N, point_dim)`` velocity.
        """
        b, n, _ = xt.shape
        if t.ndim == 0:
            t = t.expand(b)
        t = t.to(xt.dtype)

        x = self.in_proj(xt)
        cond_tokens = self.cond_proj(z)
        t_emb = self.t_mlp(timestep_embedding(t, self.t_embed_dim).to(xt.dtype))
        c = t_emb + self.cond_global(cond_tokens.mean(dim=1))

        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                x = checkpoint(block, x, cond_tokens, c, use_reentrant=False)
            else:
                x = block(x, cond_tokens, c)
        return self.final(x, c)


__all__ = ["RFPointSetVelocity"]
