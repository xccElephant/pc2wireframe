"""Edge-centric wireframe decoder: multi-scale ``z_q`` -> a set of edge curves.

The decompressor half of the VQVAE branch, reworked as an **edge set predictor**
(DETR-style, but over *edges* rather than vertices). It works only from the
quantized multi-scale latent (the discrete competition submission, rebuilt from
indices):

  * the per-scale ``z_q`` token sets are projected to ``d_model``, tagged with a
    learned **scale embedding** and concatenated into one memory;
  * ``num_edge_queries`` learnable **edge queries** are refined by an
    ``nn.TransformerDecoder`` whose every layer does (a) **self-attention over
    the edge queries** (edges talk to each other, so edges that should share a
    vertex can coordinate their endpoints) + (b) **cross-attention over the
    ``z_q`` memory** + (c) an FFN;
  * each refined edge query emits two things directly (no curve VAE, no
    normalization):

      - an **existence logit** (is this query a real edge?);
      - ``sample_points_num`` **ordered world-space sample points**
        ``(P, 3)`` regressed by an MLP. By convention ``pts[0]`` is endpoint
        ``v1`` and ``pts[-1]`` is endpoint ``v2`` -- the endpoints are just the
        first/last regressed points, so they can be supervised directly and two
        edges that share a GT vertex can each regress to the same coordinate
        (the basis for the union-find vertex merge at inference time).

The decoder exposes ``edge_exist_logit (B, Q)`` and ``edge_points (B, Q, P, 3)``;
the LightningModule's edge-set criterion (Hungarian-matched) and the endpoint
aggregation reconstruction consume these directly.
"""
from __future__ import annotations

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


class EdgeSetDecoder(nn.Module):
    """Edge-query set decoder: ``z_q`` -> per-edge existence + ordered points.

    Args:
        latent_dim: per-token channels of the (quantized) latent tokens.
        num_edge_queries: number of edge queries ``Q`` (the max edge count).
        num_scales: number of latent scales (for the scale embedding table).
        sample_points_num: ordered samples per edge curve ``P`` (``pts[0]`` /
            ``pts[-1]`` are the two endpoints).
        d_model: transformer width.
        nhead: attention heads.
        num_layers: transformer decoder layers (self-attn edges + cross-attn z_q).
        mlp_ratio: feed-forward expansion.
        dropout: dropout in attention / heads.
        points_hidden: hidden width of the per-edge points MLP.
        learn_exist_temperature: add a learnable scalar **temperature** that
            divides the existence logits (initialised to 1, i.e. a no-op). It
            lets the model calibrate the existence head's sharpness jointly with
            the BCE so ``sigmoid(logit)`` lands near ``0.5`` at the boundary;
            applied consistently at train / val / export. ``False`` -> identity.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_edge_queries: int = 512,
        num_scales: int = 3,
        sample_points_num: int = 32,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        points_hidden: int = 256,
        chord_residual: bool = True,
        learn_exist_temperature: bool = False,
    ) -> None:
        super().__init__()
        self.num_edge_queries = int(num_edge_queries)
        self.sample_points_num = int(sample_points_num)
        self.d_model = int(d_model)
        self.num_scales = int(num_scales)
        self.chord_residual = bool(chord_residual)
        self.learn_exist_temperature = bool(learn_exist_temperature)

        self.latent_proj = (
            nn.Linear(latent_dim, d_model)
            if latent_dim != d_model else nn.Identity()
        )
        self.scale_emb = nn.Parameter(
            torch.randn(self.num_scales, d_model) * 0.02)
        self.queries = nn.Parameter(
            torch.randn(1, self.num_edge_queries, d_model) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=int(d_model * mlp_ratio), dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=max(1, int(num_layers)), norm=nn.LayerNorm(d_model)
        )

        # Per-edge heads (from the refined edge-query states).
        self.exist_head = _mlp(d_model, d_model, 1)
        self.points_head = _mlp(
            d_model, points_hidden, self.sample_points_num * 3)
        # Optional post-hoc existence-logit temperature (log-param -> T = exp,
        # init T = 1). Divides the logits, learned jointly with the BCE.
        if self.learn_exist_temperature:
            self.exist_log_temp = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    def _build_memory(self, z_q_list: list[torch.Tensor]) -> torch.Tensor:
        """Concat per-scale ``z_q`` into one memory with scale embeddings."""
        if len(z_q_list) > self.num_scales:
            raise ValueError(
                f"got {len(z_q_list)} scales > num_scales={self.num_scales}")
        parts = []
        for s, z in enumerate(z_q_list):
            mem = self.latent_proj(z) + self.scale_emb[s][None, None, :]
            parts.append(mem)
        return torch.cat(parts, dim=1)                   # (B, sum N_s, d_model)

    def forward(self, z_q_list: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Decode the multi-scale latent into a set of edge curves.

        Returns dict with::

            edge_exist_logit (B, Q)        existence logit per edge query
            edge_points      (B, Q, P, 3)  ordered world-space curve samples
                                           (pts[0] = v1, pts[-1] = v2)
        """
        if isinstance(z_q_list, torch.Tensor):
            z_q_list = [z_q_list]
        b = z_q_list[0].shape[0]
        mem = self._build_memory(z_q_list)
        q = self.queries.expand(b, -1, -1)
        h = self.decoder(tgt=q, memory=mem)              # (B, Q, d_model)

        edge_exist_logit = self.exist_head(h).squeeze(-1)            # (B, Q)
        if self.learn_exist_temperature:
            edge_exist_logit = edge_exist_logit / self.exist_log_temp.exp()
        raw = self.points_head(h).reshape(
            b, self.num_edge_queries, self.sample_points_num, 3)     # (B, Q, P, 3)

        if self.chord_residual:
            # Decouple position from shape: slots 0 / -1 are the *absolute*
            # endpoints v1 / v2 (directly regressed -> precise, mergeable), the
            # interior is the straight chord v1->v2 plus a learned offset (so a
            # straight edge only needs offset≈0; arcs learn a small residual).
            edge_points = self._chord_residual(raw)
        else:
            edge_points = raw
        return {
            "edge_exist_logit": edge_exist_logit,
            "edge_points": edge_points,
        }

    def _chord_residual(self, raw: torch.Tensor) -> torch.Tensor:
        """``raw (B,Q,P,3)`` -> points = chord(v1,v2) + interior offset.

        ``v1 = raw[...,0,:]``, ``v2 = raw[...,-1,:]`` are the absolute endpoints;
        interior point ``i`` is ``lerp(v1, v2, t_i) + raw[...,i,:]`` (the offset
        is masked to 0 at the two ends, so ``pts[0]=v1`` and ``pts[-1]=v2``
        exactly).
        """
        p = self.sample_points_num
        v1 = raw[..., 0, :]                                  # (B, Q, 3)
        v2 = raw[..., -1, :]
        t = torch.linspace(0.0, 1.0, p, device=raw.device, dtype=raw.dtype)
        chord = (v1[..., None, :] * (1.0 - t)[..., :, None]
                 + v2[..., None, :] * t[..., :, None])       # (B, Q, P, 3)
        mask = torch.ones(p, device=raw.device, dtype=raw.dtype)
        mask[0] = 0.0
        mask[-1] = 0.0
        return chord + raw * mask[..., :, None]


__all__ = ["EdgeSetDecoder"]
