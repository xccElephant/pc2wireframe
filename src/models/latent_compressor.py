"""Latent compressor: cross-attention pooling of encoder tokens into the
fixed-length, budget-constrained wireframe latent ``Z_W``.

This is the analogue of the baseline's ``LatentCompressor`` and of the wireframe
VAE's Perceiver-style graph encoder, but it pools *point-cloud* features instead
of ground-truth wireframe features. ``K = num_tokens`` learnable queries
cross-attend over the (padded) encoder token set and project to a
``latent_dim``-channel distribution, giving a latent of ``num_tokens *
latent_dim`` floats. The competition hard cap is 4096 float32 values.

Output layout is ``(B, num_tokens, latent_dim)`` -- callers that feed the
wireframe VAE decoder should rearrange to ``(B, latent_dim, num_tokens)``
(the decoder's ``decode(z=...)`` expects ``'b d n'``).
"""
from __future__ import annotations

import torch
from torch import nn


class LatentCompressor(nn.Module):
    """Pool variable-length encoder tokens into ``(num_tokens, latent_dim)``.

    Args:
        in_dim: feature dim of the input tokens (e.g. PTv3 output channels).
        num_tokens: number of latent tokens ``K`` (= ``wireframe_latent_num``).
        latent_dim: per-token latent channels (= ``latent_channels``).
        nhead: attention heads for the cross-attention pooling.
        variational: if True, also predict ``logvar`` for a VAE-style latent
            (KL + reparameterisation). For a deterministic point-cloud -> Z_W
            regressor this can be False (predict the mean only).
        dropout: attention dropout.
        latent_budget_max: hard cap on ``num_tokens * latent_dim``.
    """

    LATENT_BUDGET_MAX_DEFAULT = 4096

    def __init__(
        self,
        in_dim: int,
        num_tokens: int = 64,
        latent_dim: int = 64,
        nhead: int = 8,
        variational: bool = True,
        dropout: float = 0.0,
        latent_budget_max: int | None = None,
    ) -> None:
        super().__init__()
        budget = int(num_tokens) * int(latent_dim)
        cap = (
            self.LATENT_BUDGET_MAX_DEFAULT
            if latent_budget_max is None
            else int(latent_budget_max)
        )
        if budget > cap:
            raise ValueError(
                f"latent budget {num_tokens}x{latent_dim}={budget} > {cap} "
                f"(competition cap is {self.LATENT_BUDGET_MAX_DEFAULT} floats)"
            )
        self.in_dim = int(in_dim)
        self.num_tokens = int(num_tokens)
        self.latent_dim = int(latent_dim)
        self.variational = bool(variational)

        self.queries = nn.Parameter(torch.randn(1, num_tokens, in_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            in_dim, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(in_dim)
        self.to_mu = nn.Linear(in_dim, latent_dim)
        self.to_logvar = (
            nn.Linear(in_dim, latent_dim) if self.variational else None
        )

    @property
    def latent_budget(self) -> int:
        return self.num_tokens * self.latent_dim

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Pool ``tokens (B, L, in_dim)`` into the latent distribution.

        Args:
            tokens: padded encoder tokens.
            key_padding_mask: bool ``(B, L)`` with ``True`` at padded
                positions (ignored by attention), matching
                ``nn.MultiheadAttention``.

        Returns:
            ``(mu, logvar)`` each ``(B, num_tokens, latent_dim)``; ``logvar``
            is ``None`` when ``variational=False``.
        """
        b = tokens.shape[0]
        q = self.queries.expand(b, -1, -1)
        attn_out, _ = self.cross_attn(
            q, tokens, tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        h = self.norm(q + attn_out)
        mu = self.to_mu(h)
        logvar = (
            self.to_logvar(h).clamp(-20.0, 10.0)
            if self.to_logvar is not None
            else None
        )
        return mu, logvar

    @staticmethod
    def reparameterize(
        mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)


__all__ = ["LatentCompressor"]
