"""WireframeAE decoder: latent ``(B, K, D)`` -> vertex queries + pairwise edges.

The decoder is the *decompressor* half of the WireframeAE autoencoder. It works
**only** from the 16x256 latent (the competition submission), reconstructing an
explicit wireframe with a DETR-style set predictor:

  * ``num_queries`` learnable **vertex queries** cross-attend to the latent via
    an ``nn.TransformerDecoder``; each query emits an ``alive`` logit (is this a
    real vertex?) plus its ``xyz`` coordinate.
  * the **edge head** scores *pairs* of vertex queries. For a pair ``(i, j)`` it
    consumes a symmetric, global-aware feature
    ``[h_i, h_j, h_i * h_j, |h_i - h_j|, global]`` (``global`` = the mean of the
    decoded query tokens, injecting shape-level context) and emits an
    ``exist`` logit, a 3-way curve ``type`` (line / arc / bezier) and the two
    interior curve anchors ``(q1, q2)`` (6 params). The edge endpoints ``a, b``
    are the two queries' predicted ``xyz``.

The edge head is exposed as a standalone callable (:meth:`edge_logits`) so the
loss can score only the matched-query pairs and the decoder can score only the
alive-vertex pairs, instead of materialising all ``Q*(Q-1)/2`` pairs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _mlp(d_in: int, d_hidden: int, d_out: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = d_in
    for _ in range(max(1, depth) - 1):
        layers += [nn.Linear(d, d_hidden), nn.GELU()]
        d = d_hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class WireframeAE(nn.Module):
    """Vertex-query set decoder + pairwise edge head, conditioned on the latent.

    Args:
        latent_dim: per-token latent channels ``D`` (= encoder ``latent_dim``).
        num_queries: number of vertex queries ``Q`` (the max vertex count).
        d_model: decoder transformer width.
        nhead: attention heads.
        num_layers: ``nn.TransformerDecoder`` layers.
        mlp_ratio: feed-forward expansion.
        dropout: dropout in attention / heads.
        edge_hidden: hidden width of the pairwise edge MLP.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_queries: int = 512,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        edge_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.d_model = int(d_model)

        self.latent_proj = (
            nn.Linear(latent_dim, d_model)
            if latent_dim != d_model else nn.Identity()
        )
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=max(1, int(num_layers)), norm=nn.LayerNorm(d_model)
        )

        # Per-query vertex heads.
        self.alive_head = _mlp(d_model, d_model, 1)
        self.xyz_head = _mlp(d_model, d_model, 3)

        # Pairwise edge head: [h_i, h_j, h_i*h_j, |h_i-h_j|, global] -> ...
        edge_in = 5 * d_model
        self.edge_exist_head = _mlp(edge_in, edge_hidden, 1)
        self.edge_type_head = _mlp(edge_in, edge_hidden, 3)
        self.edge_param_head = _mlp(edge_in, edge_hidden, 6)

    # ------------------------------------------------------------------
    def vertex_heads(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply the per-query vertex heads to a hidden state ``(B, Q, d)``.

        Factored out so the same heads can score the final decoder layer *and*
        each intermediate layer (DETR-style auxiliary losses).
        """
        return {
            "vertex_logit": self.alive_head(h).squeeze(-1),
            "vertex_xyz": self.xyz_head(h),
            "hidden": h,
            "global": h.mean(dim=1),
        }

    def forward(
        self, latent: torch.Tensor, return_intermediate: bool = False
    ) -> dict[str, torch.Tensor]:
        """Decode the latent into per-query vertex fields + hidden states.

        Args:
            latent: ``(B, K, D)`` compressed latent tokens.
            return_intermediate: also return every non-final decoder layer's
                (normed) hidden state under ``aux_hidden`` for auxiliary losses.

        Returns:
            dict with::

                vertex_logit (B, Q)          alive logit per query
                vertex_xyz   (B, Q, 3)       predicted vertex coordinate
                hidden       (B, Q, d_model) query tokens (for the edge head)
                global       (B, d_model)    mean query token (shape context)
                aux_hidden   list[(B, Q, d_model)]  (only if return_intermediate)
        """
        b = latent.shape[0]
        mem = self.latent_proj(latent)                  # (B, K, d_model)
        q = self.queries.expand(b, -1, -1)

        if not return_intermediate:
            h = self.decoder(tgt=q, memory=mem)         # (B, Q, d_model)
            return self.vertex_heads(h)

        # Manually unroll the layers to expose intermediate hidden states; the
        # final-layer output is numerically identical to ``self.decoder(...)``.
        hs: list[torch.Tensor] = []
        output = q
        for layer in self.decoder.layers:
            output = layer(output, mem)
            hs.append(output)
        if self.decoder.norm is not None:
            hs = [self.decoder.norm(o) for o in hs]
        out = self.vertex_heads(hs[-1])
        out["aux_hidden"] = hs[:-1]
        return out

    # ------------------------------------------------------------------
    def edge_logits(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        global_vec: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score a set of query pairs from their hidden states.

        Args:
            h_i: ``(M, d_model)`` first-endpoint query hidden states.
            h_j: ``(M, d_model)`` second-endpoint query hidden states.
            global_vec: ``(M, d_model)`` (or ``(d_model,)`` broadcastable)
                shape-context feature.

        Returns:
            dict with ``exist (M,)``, ``type (M, 3)``, ``params (M, 2, 3)``.
        """
        if h_i.shape[0] == 0:
            return {
                "exist": h_i.new_zeros(0),
                "type": h_i.new_zeros(0, 3),
                "params": h_i.new_zeros(0, 2, 3),
            }
        if global_vec.dim() == 1:
            global_vec = global_vec[None, :].expand(h_i.shape[0], -1)
        feat = torch.cat(
            [h_i, h_j, h_i * h_j, (h_i - h_j).abs(), global_vec], dim=-1)
        return {
            "exist": self.edge_exist_head(feat).squeeze(-1),
            "type": self.edge_type_head(feat),
            "params": self.edge_param_head(feat).reshape(-1, 2, 3),
        }


__all__ = ["WireframeAE"]
