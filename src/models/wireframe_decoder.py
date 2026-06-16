"""Transformer wireframe decoder: tokenized latent -> node set (+ rln token).

This is the trainable vertex head of the (single-stage) point-cloud ->
wireframe model. It consumes the fixed-length latent ``Z`` produced by the
PTv3 encoder + latent compressor and decodes a **node set** directly:

  * the latent tokens are projected to ``d_model`` and act as cross-attention
    **memory**;
  * ``num_node_queries`` learnable node queries -- together with a small number
    of learnable **``[rln]`` (relation) tokens** -- self-attend, then
    cross-attend to the memory. A pair of heads predict, per node query, its
    **coordinate** and **existence** confidence (a count-free node set); the
    ``[rln]`` tokens carry no coordinate and instead accumulate global
    relation / topology context for the pairwise edge head (see
    ``edge_decoder.RelationEdgeHead``).

Edges are **not** decoded here any more. There is no edge query set, no pointer
endpoints and no per-edge curve head in this module; instead the model scores
candidate *vertex pairs* downstream with the Relationformer-style relation head,
using the node tokens ``o`` and the global ``[rln]`` token ``r`` produced here.

Two refinements borrowed from the DETR family:

  * **Iterative coordinate refinement** (Deformable-DETR style): each node query
    carries a reference point; every layer predicts a *delta* that updates the
    reference in inverse-sigmoid space, refining coordinates layer by layer.
  * **Deep supervision**: every intermediate node layer emits a full set of
    node predictions through the shared heads; ``forward`` returns them under
    ``aux_node`` so ``criterion.py`` can supervise each layer.

Training matches node queries to the GT vertices with the Hungarian algorithm
(see ``criterion.py``); this module is pure ``forward`` and owns no loss logic.
"""
from __future__ import annotations

import torch
from torch import nn


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Numerically-stable logit (inverse of ``sigmoid``) on ``[0, 1]`` inputs."""
    x = x.clamp(0.0, 1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class _DecoderBlock(nn.Module):
    """Pre-norm self-attn -> one or two cross-attns -> FFN.

    ``num_cross`` cross-attention sub-layers are applied in sequence, one per
    memory tensor passed to ``forward``.
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
    """Latent tokens -> node set (coord, existence) + global ``[rln]`` token(s).

    Args:
        latent_token_dim: per-token latent channels ``D`` of ``Z`` (= 256).
        num_latent_tokens: number of latent tokens ``K`` of ``Z`` (= 16).
        num_node_queries: learnable node queries (>= dataset ``max_vertices``).
        num_rln_tokens: learnable Relationformer relation tokens (1-4); they
            self-attend with the node queries and feed the pairwise edge head.
        d_model / nhead / dim_ff: transformer width / heads / FFN size.
        node_layers: decoder depth for the node stack.
        dropout: dropout in attention / FFN.
    """

    def __init__(
        self,
        *,
        latent_token_dim: int = 256,
        num_latent_tokens: int = 16,
        num_node_queries: int = 768,
        num_rln_tokens: int = 1,
        d_model: int = 512,
        nhead: int = 8,
        dim_ff: int = 2048,
        node_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_node_queries = int(num_node_queries)
        self.num_rln_tokens = max(1, int(num_rln_tokens))
        self.d_model = int(d_model)

        # ----- latent memory -----
        self.latent_proj = nn.Linear(latent_token_dim, d_model)
        self.latent_norm = nn.LayerNorm(d_model)
        self.latent_pos = nn.Parameter(
            torch.randn(1, num_latent_tokens, d_model) * 0.02)

        # ----- node + relation tokens -----
        self.node_queries = nn.Parameter(
            torch.randn(self.num_node_queries, d_model) * 0.02)
        self.rln_tokens = nn.Parameter(
            torch.randn(self.num_rln_tokens, d_model) * 0.02)
        self.node_layers = nn.ModuleList([
            _DecoderBlock(d_model, nhead, dim_ff, dropout, num_cross=1)
            for _ in range(node_layers)
        ])
        self.node_norm = nn.LayerNorm(d_model)
        # Per-query initial reference point (in (0, 1)) for iterative coordinate
        # refinement; the coord head predicts a delta that updates the reference
        # in inverse-sigmoid space at every layer.
        self.ref_head = nn.Linear(d_model, 3)
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 3))
        self.node_exist_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1))

    # ------------------------------------------------------------------
    def encode_memory(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        """Latent ``(B, K, D)`` -> cross-attention memory ``(B, K, d_model)``."""
        h = self.latent_proj(latent_tokens)
        return self.latent_norm(h) + self.latent_pos

    def forward(self, latent_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode the latent into the node predictions + relation token(s).

        Returns a dict with the **final-layer** predictions (``B`` batch, ``Nq``
        node queries, ``R`` relation tokens):
            ``node_tokens``       ``(B, Nq, d_model)``
            ``coord``             ``(B, Nq, 3)``   in ``(-1, 1)``
            ``node_exist_logit``  ``(B, Nq)``
            ``rln_token``         ``(B, R, d_model)``  global relation token(s)
        plus deep-supervision side outputs (list over the *intermediate* node
        layers, finest last):
            ``aux_node``  list of ``{coord, node_exist_logit}``
        """
        b = latent_tokens.shape[0]
        nq = self.num_node_queries
        mem = self.encode_memory(latent_tokens)

        # node queries + relation tokens self-attend together, then cross-attend
        # to the latent memory; coordinates are iteratively refined.
        seed = torch.cat([self.node_queries, self.rln_tokens], dim=0)
        q = seed.unsqueeze(0).expand(b, -1, -1)
        reference = self.ref_head(self.node_queries).sigmoid()
        reference = reference.unsqueeze(0).expand(b, -1, -1)

        node_coords: list[torch.Tensor] = []
        node_exists: list[torch.Tensor] = []
        node_tokens = q[:, :nq]
        rln_token = q[:, nq:]
        for layer in self.node_layers:
            q = layer(q, [mem])
            h = self.node_norm(q)
            node_tokens = h[:, :nq]
            rln_token = h[:, nq:]
            # refine the reference in inverse-sigmoid space, then map to (-1, 1).
            delta = self.coord_head(node_tokens)
            new_ref = (delta + inverse_sigmoid(reference)).sigmoid()
            node_coords.append(new_ref * 2.0 - 1.0)
            node_exists.append(self.node_exist_head(node_tokens).squeeze(-1))
            reference = new_ref.detach()

        out: dict[str, torch.Tensor] = {
            "node_tokens": node_tokens,
            "coord": node_coords[-1],
            "node_exist_logit": node_exists[-1],
            "rln_token": rln_token,
            "aux_node": [
                {"coord": c, "node_exist_logit": x}
                for c, x in zip(node_coords[:-1], node_exists[:-1])
            ],
        }
        return out


__all__ = ["WireframeDecoder"]
