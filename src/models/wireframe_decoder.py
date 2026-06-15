"""Transformer wireframe decoder: tokenized latent -> node set + edge set.

This is the trainable head of the (single-stage) point-cloud -> wireframe model.
It consumes the fixed-length latent ``Z`` produced by the PTv3 encoder + latent
compressor and decodes a wireframe **directly** -- there is no separate graph
VAE any more:

  * the latent tokens are projected to ``d_model`` and act as cross-attention
    **memory**;
  * ``num_node_queries`` learnable node queries cross-attend to the memory and a
    pair of heads predict, in parallel, each node's **coordinate** and
    **existence** confidence (a count-free node set);
  * ``num_edge_queries`` learnable edge queries self-attend, cross-attend to the
    latent memory and then cross-attend to the decoded **node features**; heads
    predict each edge's **existence**, two **endpoint distributions** (a
    pointer-style softmax over the node queries for endpoint A / B) and a
    per-edge **curve latent** (consumed by the frozen curve VAE decoder, which
    predicts a residual polyline on top of the endpoint-interpolation baseline).

Training matches node / edge queries to the GT graph with the Hungarian
algorithm (see ``criterion.py``); this module is pure ``forward`` and owns no
loss logic.
"""
from __future__ import annotations

import torch
from torch import nn


class _DecoderBlock(nn.Module):
    """Pre-norm self-attn -> one or two cross-attns -> FFN.

    ``num_cross`` cross-attention sub-layers are applied in sequence, one per
    memory tensor passed to ``forward`` (node queries attend to the latent
    only; edge queries attend to the latent and then to the node features).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        dropout: float = 0.1,
        num_cross: int = 1,
    ) -> None:
        super().__init__()
        self.sa_norm = nn.LayerNorm(d_model)
        self.sa = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_norms_q = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_cross)])
        self.cross_norms_kv = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_cross)])
        self.cross = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                  batch_first=True)
            for _ in range(num_cross)
        ])
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout))

    def forward(self, q: torch.Tensor, mems: list[torch.Tensor]) -> torch.Tensor:
        x = self.sa_norm(q)
        q = q + self.sa(x, x, x, need_weights=False)[0]
        for norm_q, norm_kv, attn, mem in zip(
            self.cross_norms_q, self.cross_norms_kv, self.cross, mems
        ):
            xq = norm_q(q)
            kv = norm_kv(mem)
            q = q + attn(xq, kv, kv, need_weights=False)[0]
        q = q + self.ff(self.ff_norm(q))
        return q


class WireframeDecoder(nn.Module):
    """Latent tokens -> node set + edge set (existence, endpoints, curve latent).

    Args:
        latent_token_dim: per-token latent channels ``D`` of ``Z`` (= 256).
        num_latent_tokens: number of latent tokens ``K`` of ``Z`` (= 16).
        curve_latent_dim: per-edge curve-VAE latent size (= ``K_c * D_c``, 12).
        num_node_queries: learnable node queries (>= dataset ``max_vertices``).
        num_edge_queries: learnable edge queries (>= dataset ``max_edges``).
        d_model / nhead / dim_ff: transformer width / heads / FFN size.
        node_layers / edge_layers: decoder depth for the node / edge stacks.
        endpoint_dim: projection dim for the pointer-style endpoint scoring.
        dropout: dropout in attention / FFN.
    """

    def __init__(
        self,
        *,
        latent_token_dim: int = 256,
        num_latent_tokens: int = 16,
        curve_latent_dim: int = 12,
        num_node_queries: int = 768,
        num_edge_queries: int = 1024,
        d_model: int = 512,
        nhead: int = 8,
        dim_ff: int = 2048,
        node_layers: int = 6,
        edge_layers: int = 4,
        endpoint_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_node_queries = int(num_node_queries)
        self.num_edge_queries = int(num_edge_queries)
        self.curve_latent_dim = int(curve_latent_dim)
        self.d_model = int(d_model)
        self.endpoint_dim = int(endpoint_dim)

        # ----- latent memory -----
        self.latent_proj = nn.Linear(latent_token_dim, d_model)
        self.latent_norm = nn.LayerNorm(d_model)
        self.latent_pos = nn.Parameter(
            torch.randn(1, num_latent_tokens, d_model) * 0.02)

        # ----- node decoder -----
        self.node_queries = nn.Parameter(
            torch.randn(num_node_queries, d_model) * 0.02)
        self.node_layers = nn.ModuleList([
            _DecoderBlock(d_model, nhead, dim_ff, dropout, num_cross=1)
            for _ in range(node_layers)
        ])
        self.node_norm = nn.LayerNorm(d_model)
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 3))
        self.node_exist_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1))

        # ----- edge decoder (cross-attends to latent then to node features) -----
        self.edge_queries = nn.Parameter(
            torch.randn(num_edge_queries, d_model) * 0.02)
        self.edge_layers = nn.ModuleList([
            _DecoderBlock(d_model, nhead, dim_ff, dropout, num_cross=2)
            for _ in range(edge_layers)
        ])
        self.edge_norm = nn.LayerNorm(d_model)
        self.edge_exist_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1))
        # pointer-style endpoint scoring: edge query . node key -> per-node logit.
        self.ep_a_proj = nn.Linear(d_model, endpoint_dim)
        self.ep_b_proj = nn.Linear(d_model, endpoint_dim)
        self.node_key_proj = nn.Linear(d_model, endpoint_dim)
        self.curve_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, curve_latent_dim))

    # ------------------------------------------------------------------
    def encode_memory(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        """Latent ``(B, K, D)`` -> cross-attention memory ``(B, K, d_model)``."""
        h = self.latent_proj(latent_tokens)
        return self.latent_norm(h) + self.latent_pos

    def forward(self, latent_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode the latent into the node / edge predictions.

        Returns a dict with (``B`` batch, ``Nq`` node queries, ``Ne`` edge
        queries):
            ``node_tokens``       ``(B, Nq, d_model)``
            ``coord``             ``(B, Nq, 3)``   in ``(-1, 1)``
            ``node_exist_logit``  ``(B, Nq)``
            ``edge_tokens``       ``(B, Ne, d_model)``
            ``edge_exist_logit``  ``(B, Ne)``
            ``ep_a_logits``       ``(B, Ne, Nq)``  endpoint-A node distribution
            ``ep_b_logits``       ``(B, Ne, Nq)``  endpoint-B node distribution
            ``curve_latent``      ``(B, Ne, curve_latent_dim)``
        """
        b = latent_tokens.shape[0]
        mem = self.encode_memory(latent_tokens)

        # ----- nodes -----
        q = self.node_queries.unsqueeze(0).expand(b, -1, -1)
        for layer in self.node_layers:
            q = layer(q, [mem])
        node_tokens = self.node_norm(q)
        coord = torch.tanh(self.coord_head(node_tokens))
        node_exist_logit = self.node_exist_head(node_tokens).squeeze(-1)

        # ----- edges (attend to latent memory + node features) -----
        e = self.edge_queries.unsqueeze(0).expand(b, -1, -1)
        for layer in self.edge_layers:
            e = layer(e, [mem, node_tokens])
        edge_tokens = self.edge_norm(e)
        edge_exist_logit = self.edge_exist_head(edge_tokens).squeeze(-1)

        node_keys = self.node_key_proj(node_tokens)             # (B, Nq, h)
        scale = self.endpoint_dim ** -0.5
        ep_a_logits = torch.matmul(
            self.ep_a_proj(edge_tokens), node_keys.transpose(1, 2)) * scale
        ep_b_logits = torch.matmul(
            self.ep_b_proj(edge_tokens), node_keys.transpose(1, 2)) * scale
        curve_latent = self.curve_head(edge_tokens)

        return {
            "node_tokens": node_tokens,
            "coord": coord,
            "node_exist_logit": node_exist_logit,
            "edge_tokens": edge_tokens,
            "edge_exist_logit": edge_exist_logit,
            "ep_a_logits": ep_a_logits,
            "ep_b_logits": ep_b_logits,
            "curve_latent": curve_latent,
        }


__all__ = ["WireframeDecoder"]
