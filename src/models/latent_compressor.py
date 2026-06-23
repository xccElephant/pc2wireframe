"""Latent compressor: cross-attention pooling of encoder tokens into the
fixed-length, budget-constrained tokenized latent ``Z``.

``K = num_tokens`` learnable queries are pooled over the (padded) encoder token
set by a standard ``nn.TransformerDecoder`` (Perceiver / DETR style: each layer
= query self-attention + cross-attention to the encoder tokens + FFN) and then
projected to a ``latent_dim``-channel distribution, giving a latent of
``num_tokens * latent_dim`` floats. The competition hard cap is 4096 float32
values. A single decoder layer (``num_layers=1``) already performs the pooling;
more layers just add mixing capacity.

Output layout is ``(B, num_tokens, latent_dim)``, consumed directly as the
compressed latent decoded by the WireframeAE decoder
(:class:`~src.models.wireframe_ae.WireframeAE`).
"""
from __future__ import annotations

import torch
from torch import nn


class LatentCompressor(nn.Module):
    """Pool variable-length encoder tokens into ``(num_tokens, latent_dim)``.

    Args:
        in_dim: feature dim of the input tokens (e.g. PTv3 output channels).
        num_tokens: number of latent tokens ``K`` (default 16).
        latent_dim: per-token latent channels (default 256; 16*256=4096 floats).
        nhead: attention heads for the cross-attention pooling.
        num_layers: number of ``nn.TransformerDecoder`` layers pooling the
            latent tokens (default 1; one layer already pools, more add mixing).
        dropout: attention dropout.
        latent_budget_max: hard cap on ``num_tokens * latent_dim``.
    """

    LATENT_BUDGET_MAX_DEFAULT = 4096

    def __init__(
        self,
        in_dim: int,
        num_tokens: int = 16,
        latent_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 1,
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
        self.num_layers = max(1, int(num_layers))

        # K learnable query tokens pooled by a standard pre-norm Transformer
        # decoder (self-attn over queries + cross-attn to the encoder tokens +
        # FFN). One layer already pools; the decoder *is* the pooler, so there
        # is no separate cross-attention step.
        self.queries = nn.Parameter(torch.randn(1, num_tokens, in_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=in_dim,
            nhead=nhead,
            dim_feedforward=4 * in_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=self.num_layers, norm=nn.LayerNorm(in_dim)
        )
        self.to_latent = nn.Linear(in_dim, latent_dim)

    @property
    def latent_budget(self) -> int:
        return self.num_tokens * self.latent_dim

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool ``tokens (B, L, in_dim)`` into the latent ``(B, num_tokens, latent_dim)``.

        Args:
            tokens: padded encoder tokens.
            key_padding_mask: bool ``(B, L)`` with ``True`` at padded
                positions (ignored by attention), matching
                ``nn.MultiheadAttention``.
        """
        b = tokens.shape[0]
        q = self.queries.expand(b, -1, -1)
        h = self.decoder(
            tgt=q, memory=tokens, memory_key_padding_mask=key_padding_mask
        )
        return self.to_latent(h)


__all__ = ["LatentCompressor"]
