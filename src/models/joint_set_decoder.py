"""Joint vertex + edge set decoder: multi-scale ``z_q`` -> vertices + edge curves.

The ComplexGen-style middle road between the two failing extremes (edge-only +
union-find merge -> bad vertices/topology; vertex-only + pairwise edge
prediction -> bad edge geometry). It keeps **two** disjoint query sets and an
explicit edge-vertex **association matrix** so geometry and topology are
decoded by dedicated heads instead of being entangled:

  * the per-scale (quantized) latent tokens are projected to ``d_model``, tagged
    with a learned **scale embedding** and concatenated into one memory;
  * ``num_vertex_queries`` **vertex queries** and ``num_edge_queries`` **edge
    queries** are refined together by ``N`` :class:`JointDecoderLayer` s. Each
    layer does, with ``norm_first`` pre-norm residuals:

      1. **self-attention** within each set (vertex<->vertex, edge<->edge);
      2. **cross-attention** of each set to the ``z_q`` memory;
      3. **mutual cross-attention** -- vertices attend the edges and edges
         attend the vertices (computed from a snapshot so neither ordering
         wins), the coupling a plain ``nn.TransformerDecoderLayer`` cannot
         express;
      4. a per-set **FFN**.

  * heads read off the refined states:

      - **vertex** -- existence logit ``(Nv,)`` + coordinate ``(Nv, 3)``;
      - **edge**   -- existence logit ``(Ne,)`` + a ``curve_latent (Ne, D)``
        (``D = latent_channels * latent_len`` of the curve VAE);
      - **association** -- low-rank ``A_logit (Ne, Nv) = (He W_e)(Hv W_v)^T /
        sqrt(k)``; ``sigmoid(A_logit)`` is the soft edge->vertex incidence used
        (top-2 per edge) to pick endpoints at reconstruction time.

Exposes ``vertex_exist_logit / vertex_coord / edge_exist_logit / curve_latent /
assoc_logit``; the joint criterion (Hungarian-matched) and the joint
reconstruction consume these directly.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _mlp(d_in: int, d_hidden: int, d_out: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = d_in
    for _ in range(max(1, depth) - 1):
        layers += [nn.Linear(d, d_hidden), nn.GELU()]
        d = d_hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class JointDecoderLayer(nn.Module):
    """One joint decoder layer over vertex + edge tokens (pre-norm residuals).

    Sub-layers (each a pre-norm residual): per-set self-attention, per-set
    cross-attention to the latent memory, mutual vertex<->edge cross-attention
    (from a pre-mutual snapshot), per-set FFN.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        def _mha() -> nn.MultiheadAttention:
            return nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True)

        def _ff() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_model, dim_feedforward), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))

        # vertex-token sublayers
        self.v_self_attn = _mha()
        self.v_mem_attn = _mha()
        self.v_edge_attn = _mha()
        self.v_ffn = _ff()
        self.v_norm_self = nn.LayerNorm(d_model)
        self.v_norm_mem = nn.LayerNorm(d_model)
        self.v_norm_edge = nn.LayerNorm(d_model)
        self.v_norm_ffn = nn.LayerNorm(d_model)

        # edge-token sublayers
        self.e_self_attn = _mha()
        self.e_mem_attn = _mha()
        self.e_vert_attn = _mha()
        self.e_ffn = _ff()
        self.e_norm_self = nn.LayerNorm(d_model)
        self.e_norm_mem = nn.LayerNorm(d_model)
        self.e_norm_vert = nn.LayerNorm(d_model)
        self.e_norm_ffn = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _attn(attn: nn.MultiheadAttention, q: torch.Tensor,
              kv: torch.Tensor) -> torch.Tensor:
        out, _ = attn(q, kv, kv, need_weights=False)
        return out

    def forward(
        self,
        hv: torch.Tensor,
        he: torch.Tensor,
        mem: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. self-attention within each set.
        hv = hv + self.dropout(self._attn(
            self.v_self_attn, self.v_norm_self(hv), self.v_norm_self(hv)))
        he = he + self.dropout(self._attn(
            self.e_self_attn, self.e_norm_self(he), self.e_norm_self(he)))

        # 2. cross-attention to the latent memory.
        hv = hv + self.dropout(self._attn(
            self.v_mem_attn, self.v_norm_mem(hv), mem))
        he = he + self.dropout(self._attn(
            self.e_mem_attn, self.e_norm_mem(he), mem))

        # 3. mutual cross-attention (snapshot so neither ordering wins).
        hv_n = self.v_norm_edge(hv)
        he_n = self.e_norm_vert(he)
        hv = hv + self.dropout(self._attn(self.v_edge_attn, hv_n, he_n))
        he = he + self.dropout(self._attn(self.e_vert_attn, he_n, hv_n))

        # 4. per-set FFN.
        hv = hv + self.dropout(self.v_ffn(self.v_norm_ffn(hv)))
        he = he + self.dropout(self.e_ffn(self.e_norm_ffn(he)))
        return hv, he


class JointSetDecoder(nn.Module):
    """Vertex + edge query set decoder with an edge-vertex association head.

    Args:
        latent_dim: per-token channels of the (quantized) latent tokens.
        num_vertex_queries: number of vertex queries ``Nv``.
        num_edge_queries: number of edge queries ``Ne``.
        num_scales: number of latent scales (for the scale embedding table).
        curve_latent_dim: edge curve-latent width ``D`` (= curve VAE
            ``latent_channels * latent_len``).
        assoc_dim: rank ``k`` of the low-rank association projection.
        d_model: transformer width.
        nhead: attention heads.
        num_layers: joint decoder layers.
        mlp_ratio: feed-forward expansion.
        dropout: dropout in attention / heads.
        coord_tanh: squash predicted vertex coords through ``tanh`` (the data is
            unit-cube normalised to ``[-1, 1]``).
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_vertex_queries: int = 512,
        num_edge_queries: int = 512,
        num_scales: int = 5,
        curve_latent_dim: int = 12,
        assoc_dim: int = 64,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        coord_tanh: bool = True,
    ) -> None:
        super().__init__()
        self.num_vertex_queries = int(num_vertex_queries)
        self.num_edge_queries = int(num_edge_queries)
        self.num_scales = int(num_scales)
        self.curve_latent_dim = int(curve_latent_dim)
        self.assoc_dim = int(assoc_dim)
        self.d_model = int(d_model)
        self.coord_tanh = bool(coord_tanh)

        self.latent_proj = (
            nn.Linear(latent_dim, d_model)
            if latent_dim != d_model else nn.Identity()
        )
        self.scale_emb = nn.Parameter(
            torch.randn(self.num_scales, d_model) * 0.02)
        self.vertex_queries = nn.Parameter(
            torch.randn(1, self.num_vertex_queries, d_model) * 0.02)
        self.edge_queries = nn.Parameter(
            torch.randn(1, self.num_edge_queries, d_model) * 0.02)

        self.layers = nn.ModuleList([
            JointDecoderLayer(
                d_model, nhead, int(d_model * mlp_ratio), dropout)
            for _ in range(max(1, int(num_layers)))
        ])
        self.vertex_norm = nn.LayerNorm(d_model)
        self.edge_norm = nn.LayerNorm(d_model)

        # vertex heads
        self.vertex_exist_head = _mlp(d_model, d_model, 1)
        self.vertex_coord_head = _mlp(d_model, d_model, 3)
        # edge heads
        self.edge_exist_head = _mlp(d_model, d_model, 1)
        self.curve_latent_head = _mlp(d_model, d_model, self.curve_latent_dim)
        # low-rank association projections
        self.vertex_assoc_proj = nn.Linear(d_model, self.assoc_dim)
        self.edge_assoc_proj = nn.Linear(d_model, self.assoc_dim)

    # ------------------------------------------------------------------
    def _build_memory(self, z_q_list: list[torch.Tensor]) -> torch.Tensor:
        """Concat per-scale ``z_q`` into one memory with scale embeddings."""
        if len(z_q_list) > self.num_scales:
            raise ValueError(
                f"got {len(z_q_list)} scales > num_scales={self.num_scales}")
        parts = []
        for s, z in enumerate(z_q_list):
            parts.append(self.latent_proj(z) + self.scale_emb[s][None, None, :])
        return torch.cat(parts, dim=1)                   # (B, sum N_s, d_model)

    def forward(self, z_q_list: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Decode the multi-scale latent into vertices + edge curves + assoc.

        Returns dict with::

            vertex_exist_logit (B, Nv)      vertex existence logit
            vertex_coord       (B, Nv, 3)   vertex coordinate
            edge_exist_logit   (B, Ne)      edge existence logit
            curve_latent       (B, Ne, D)   per-edge curve VAE latent
            assoc_logit        (B, Ne, Nv)  edge->vertex association logit
        """
        if isinstance(z_q_list, torch.Tensor):
            z_q_list = [z_q_list]
        b = z_q_list[0].shape[0]
        mem = self._build_memory(z_q_list)
        hv = self.vertex_queries.expand(b, -1, -1)
        he = self.edge_queries.expand(b, -1, -1)
        for layer in self.layers:
            hv, he = layer(hv, he, mem)
        hv = self.vertex_norm(hv)
        he = self.edge_norm(he)

        vertex_exist_logit = self.vertex_exist_head(hv).squeeze(-1)   # (B, Nv)
        vertex_coord = self.vertex_coord_head(hv)                     # (B, Nv, 3)
        if self.coord_tanh:
            vertex_coord = torch.tanh(vertex_coord)
        edge_exist_logit = self.edge_exist_head(he).squeeze(-1)       # (B, Ne)
        curve_latent = self.curve_latent_head(he)                     # (B, Ne, D)

        v_proj = self.vertex_assoc_proj(hv)                           # (B, Nv, k)
        e_proj = self.edge_assoc_proj(he)                             # (B, Ne, k)
        assoc_logit = torch.matmul(
            e_proj, v_proj.transpose(1, 2)) / math.sqrt(self.assoc_dim)

        return {
            "vertex_exist_logit": vertex_exist_logit,
            "vertex_coord": vertex_coord,
            "edge_exist_logit": edge_exist_logit,
            "curve_latent": curve_latent,
            "assoc_logit": assoc_logit,
        }


__all__ = ["JointDecoderLayer", "JointSetDecoder"]
