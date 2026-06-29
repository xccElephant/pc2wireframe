"""Edge-set transformer decoder: tokenized latent -> a set of edge queries.

This is the trainable head of the (single-stage) point-cloud -> wireframe model.
It consumes the fixed-length latent ``Z`` produced by the PTv3 encoder + latent
compressor (``(B, K, D)``, default ``16 x 256``) and decodes a **set of edges**
directly, in the WireframeDETR / PBWR spirit (an edge-set regression, not a node
set + pairwise relation head):

  * the latent tokens are projected to ``d_model`` and act as cross-attention
    **memory** (plus a learned per-token positional embedding);
  * ``num_edge_queries`` learnable **edge queries** self-attend, then
    cross-attend to the latent memory through ``N`` pre-norm decoder layers;
  * three heads read off each refined edge query:

      - **existence** -- a confidence logit ``(Ne,)``;
      - **endpoints** -- two 3-D endpoints ``(Ne, 2, 3)`` squashed by ``tanh``
        into the unit cube ``[-1, 1]`` (the data's normalized frame);
      - **curve latent** -- a ``curve_latent_dim``-d per-edge curve VAE latent
        ``(Ne, D_c)`` (``D_c = latent_channels * latent_len`` of the curve VAE),
        decoded into the edge's intrinsic shape by the frozen curve VAE.

Deep supervision is optional: every intermediate decoder layer emits a full set
of edge predictions through the shared heads; ``forward`` returns them under
``aux`` so the criterion can supervise each layer. Training matches edge queries
to the GT edges with the Hungarian algorithm (see ``edge_set_criterion.py``);
this module is pure ``forward`` and owns no loss logic.
"""
from __future__ import annotations

import torch
from torch import nn


class _DecoderBlock(nn.Module):
    """Pre-norm self-attention -> cross-attention -> FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sa_norm = nn.LayerNorm(d_model)
        self.sa = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_norm_q = nn.LayerNorm(d_model)
        self.cross_norm_kv = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout))

    def forward(self, q: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        x = self.sa_norm(q)
        q = q + self.sa(x, x, x, need_weights=False)[0]
        xq = self.cross_norm_q(q)
        kv = self.cross_norm_kv(mem)
        q = q + self.cross(xq, kv, kv, need_weights=False)[0]
        q = q + self.ff(self.ff_norm(q))
        return q


class EdgeSetDecoder(nn.Module):
    """Latent tokens -> a set of edges (existence, endpoints, curve latent).

    Args:
        latent_token_dim: per-token latent channels ``D`` of ``Z`` (= 256).
        num_latent_tokens: number of latent tokens ``K`` of ``Z`` (= 16).
        num_edge_queries: learnable edge queries ``Ne`` (= 512).
        curve_latent_dim: per-edge curve-latent width ``D_c`` (= curve VAE
            ``latent_channels * latent_len``).
        d_model / nhead / dim_ff: transformer width / heads / FFN size.
        num_layers: decoder depth for the edge stack.
        dropout: dropout in attention / FFN.
        deep_supervision: emit per-layer predictions for deep supervision.
    """

    def __init__(
        self,
        *,
        latent_token_dim: int = 256,
        num_latent_tokens: int = 16,
        num_edge_queries: int = 512,
        curve_latent_dim: int = 12,
        d_model: int = 512,
        nhead: int = 8,
        dim_ff: int = 2048,
        num_layers: int = 6,
        dropout: float = 0.1,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        self.num_edge_queries = int(num_edge_queries)
        self.curve_latent_dim = int(curve_latent_dim)
        self.d_model = int(d_model)
        self.deep_supervision = bool(deep_supervision)

        # ----- latent memory -----
        self.latent_proj = nn.Linear(latent_token_dim, d_model)
        self.latent_norm = nn.LayerNorm(d_model)
        self.latent_pos = nn.Parameter(
            torch.randn(1, num_latent_tokens, d_model) * 0.02)

        # ----- edge queries -----
        self.edge_queries = nn.Parameter(
            torch.randn(1, self.num_edge_queries, d_model) * 0.02)
        self.layers = nn.ModuleList([
            _DecoderBlock(d_model, nhead, dim_ff, dropout)
            for _ in range(max(1, int(num_layers)))
        ])
        self.out_norm = nn.LayerNorm(d_model)

        # ----- heads -----
        self.exist_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1))
        self.endpoint_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 6))
        self.curve_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, self.curve_latent_dim))

    # ------------------------------------------------------------------
    def encode_memory(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        """Latent ``(B, K, D)`` -> cross-attention memory ``(B, K, d_model)``."""
        h = self.latent_proj(latent_tokens)
        return self.latent_norm(h) + self.latent_pos

    def _heads(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply the three heads to refined edge tokens ``(B, Ne, d_model)``."""
        b, ne, _ = h.shape
        exist = self.exist_head(h).squeeze(-1)                 # (B, Ne)
        endpoints = torch.tanh(self.endpoint_head(h)).view(b, ne, 2, 3)
        curve_latent = self.curve_head(h)                      # (B, Ne, D_c)
        return {
            "edge_exist_logit": exist,
            "endpoints": endpoints,
            "curve_latent": curve_latent,
        }

    def forward(self, latent_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode the latent into the edge-set predictions.

        Returns a dict with the **final-layer** predictions (``B`` batch, ``Ne``
        edge queries, ``D_c`` curve latent)::

            edge_exist_logit  (B, Ne)        existence confidence logit
            endpoints         (B, Ne, 2, 3)  two endpoints in [-1, 1]
            curve_latent      (B, Ne, D_c)   per-edge curve VAE latent

        plus, when ``deep_supervision`` is on, a list ``aux`` of the same dicts
        for the intermediate decoder layers (finest last).
        """
        b = latent_tokens.shape[0]
        mem = self.encode_memory(latent_tokens)
        q = self.edge_queries.expand(b, -1, -1)

        aux: list[dict[str, torch.Tensor]] = []
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            q = layer(q, mem)
            if self.deep_supervision and i < n - 1:
                aux.append(self._heads(self.out_norm(q)))

        out = self._heads(self.out_norm(q))
        out["aux"] = aux
        return out


__all__ = ["EdgeSetDecoder"]
