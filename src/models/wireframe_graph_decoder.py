"""Graph wireframe decoder: multi-scale ``z_q`` -> vertices + kNN-graph edges.

The decompressor half of the VQVAE branch. It works **only** from the quantized
multi-scale latent (the discrete competition submission, rebuilt from indices),
reconstructing an explicit wireframe with a vertex-query set predictor and a
kNN-restricted, GNN-refined pairwise edge head:

  * the per-scale ``z_q`` token sets are projected to ``d_model``, tagged with a
    learned **scale embedding** and concatenated into one memory; ``num_queries``
    learnable **vertex queries** cross-attend that memory
    (``nn.TransformerDecoder``). Each query emits an ``alive`` logit + ``xyz``.
  * a few rounds of **graph self-attention** over the query nodes (a geometry-
    aware graph transformer: node features + an xyz embedding) refine the nodes
    for relation reasoning.
  * edges are scored only over **candidate pairs** -- at inference the kNN of the
    predicted ``xyz`` (``O(V*k)`` instead of ``O(V^2)``), at training the GT
    positive pairs unioned with the kNN of the alive set (built by the loss /
    decode code). The pairwise :meth:`edge_logits` consumes a symmetric,
    global-aware feature ``[h_i, h_j, h_i*h_j, |h_i-h_j|, global]`` of the
    *refined* node states and emits ``exist`` / ``type`` (line/arc/bezier) /
    interior anchors ``(q1, q2)``.

Output field names (``vertex_logit``, ``vertex_xyz``, ``hidden``, ``global``)
and the :meth:`edge_logits` signature mirror the original ``WireframeAE`` so the
LightningModule loss / match / decode and the export script reuse them directly.
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


@torch.no_grad()
def knn_candidate_pairs(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """Undirected candidate edge pairs from the kNN of ``xyz``.

    Args:
        xyz: ``(V, 3)`` vertex coordinates.
        k: neighbours per vertex (clamped to ``V - 1``).

    Returns:
        ``(P, 2)`` long tensor of unique ``(i, j)`` pairs with ``i < j``
        (``O(V*k)`` candidates instead of the ``O(V^2)`` full upper triangle).
    """
    v = int(xyz.shape[0])
    device = xyz.device
    if v < 2:
        return torch.zeros((0, 2), dtype=torch.long, device=device)
    k_eff = min(int(k), v - 1)
    if k_eff <= 0:
        return torch.zeros((0, 2), dtype=torch.long, device=device)
    dist = torch.cdist(xyz, xyz)                          # (V, V)
    dist.fill_diagonal_(float("inf"))
    nn_idx = dist.topk(k_eff, dim=-1, largest=False).indices  # (V, k_eff)
    src = torch.arange(v, device=device).repeat_interleave(k_eff)
    dst = nn_idx.reshape(-1)
    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    key = lo * v + hi                                    # canonical undirected id
    uniq = torch.unique(key)
    return torch.stack([uniq // v, uniq % v], dim=1).long()


class WireframeGraphDecoder(nn.Module):
    """Vertex-query set decoder + GNN relation reasoning + kNN pairwise edges.

    Args:
        latent_dim: per-token channels of the (quantized) latent tokens.
        num_queries: number of vertex queries ``Q`` (the max vertex count).
        num_scales: number of latent scales (for the scale embedding table).
        d_model: transformer width.
        nhead: attention heads.
        num_layers: vertex cross-attention decoder layers.
        gnn_rounds: graph self-attention rounds over the query nodes.
        knn_k: neighbours per vertex for the kNN edge candidates (used by the
            loss / decode; stored here as the single source of truth).
        mlp_ratio: feed-forward expansion.
        dropout: dropout in attention / heads.
        edge_hidden: hidden width of the pairwise edge MLP.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_queries: int = 512,
        num_scales: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        gnn_rounds: int = 3,
        knn_k: int = 24,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        edge_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.d_model = int(d_model)
        self.num_scales = int(num_scales)
        self.knn_k = int(knn_k)

        self.latent_proj = (
            nn.Linear(latent_dim, d_model)
            if latent_dim != d_model else nn.Identity()
        )
        self.scale_emb = nn.Parameter(
            torch.randn(self.num_scales, d_model) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=int(d_model * mlp_ratio), dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=max(1, int(num_layers)), norm=nn.LayerNorm(d_model)
        )

        # Per-query vertex heads (from the cross-attended nodes).
        self.alive_head = _mlp(d_model, d_model, 1)
        self.xyz_head = _mlp(d_model, d_model, 3)

        # Geometry-aware GNN relation reasoning over the Q nodes.
        self.xyz_emb = nn.Linear(3, d_model)
        self.gnn_rounds = max(0, int(gnn_rounds))
        if self.gnn_rounds > 0:
            gnn_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=int(d_model * mlp_ratio), dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.gnn = nn.TransformerEncoder(gnn_layer, num_layers=self.gnn_rounds)
        else:
            self.gnn = None

        # Pairwise edge head: [h_i, h_j, h_i*h_j, |h_i-h_j|, global] -> ...
        edge_in = 5 * d_model
        self.edge_exist_head = _mlp(edge_in, edge_hidden, 1)
        self.edge_type_head = _mlp(edge_in, edge_hidden, 3)
        self.edge_param_head = _mlp(edge_in, edge_hidden, 6)

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
        """Decode the multi-scale latent into vertex fields + refined nodes.

        Returns dict with::

            vertex_logit (B, Q)          alive logit per query
            vertex_xyz   (B, Q, 3)       predicted vertex coordinate
            hidden       (B, Q, d_model) GNN-refined node states (edge head input)
            global       (B, d_model)    mean refined node (shape context)
        """
        if isinstance(z_q_list, torch.Tensor):
            z_q_list = [z_q_list]
        b = z_q_list[0].shape[0]
        mem = self._build_memory(z_q_list)
        q = self.queries.expand(b, -1, -1)
        h = self.decoder(tgt=q, memory=mem)              # (B, Q, d_model)

        vertex_logit = self.alive_head(h).squeeze(-1)
        vertex_xyz = self.xyz_head(h)

        # Geometry-aware relation reasoning on the query nodes.
        h_ref = h + self.xyz_emb(vertex_xyz)
        if self.gnn is not None:
            h_ref = self.gnn(h_ref)

        return {
            "vertex_logit": vertex_logit,
            "vertex_xyz": vertex_xyz,
            "hidden": h_ref,
            "global": h_ref.mean(dim=1),
        }

    # ------------------------------------------------------------------
    def edge_logits(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        global_vec: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score query pairs from their (refined) hidden states.

        Args:
            h_i: ``(M, d_model)`` first-endpoint node states.
            h_j: ``(M, d_model)`` second-endpoint node states.
            global_vec: ``(M, d_model)`` or ``(d_model,)`` shape-context feature.

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


__all__ = ["WireframeGraphDecoder", "knn_candidate_pairs"]
