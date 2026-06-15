"""Graph wireframe VAE (pure PyTorch).

A wireframe is modelled as an **attributed graph** ``G = (V, E)``:

  * **nodes** ``V`` -- the curve endpoints, each carrying a 3D coordinate;
  * **edges** ``E`` -- the curves, each carrying a frozen curve-VAE latent
    (intrinsic shape) plus the two endpoint references.

Unlike the previous CLR-Wire formulation (which serialised the graph into a
fragile ``(col_diff, row_diff)`` differential-adjacency sequence and regressed
each edge's 6-d endpoints independently), this VAE decodes the graph natively:

  * **Encoder** -- node tokens (Fourier-embedded coords) and edge tokens
    (curve latent + injected endpoint embeddings) self-attend; ``M`` learnable
    latent queries cross-attend to them to produce the Gaussian latent
    ``(B, latent_channels, M)`` (kept at ``64 x 64`` for the stage-3 contract).
  * **Decoder** -- the latent tokens self-attend; ``max_nodes`` learnable node
    queries cross-attend to them. Three heads then predict:
      - per-node ``(coord, exist)``      (the vertex set, count-free),
      - a symmetric inner-product adjacency over node tokens (link prediction),
      - per-edge curve latent from the two endpoint node tokens.

Training uses Hungarian matching between predicted and GT nodes (permutation
invariance), then supervises node coords / existence / adjacency / curve latent
in the matched index space. Endpoints are thus decoded **once** and shared
exactly; topology is a pairwise classification rather than a cumsum chain.

Public surface (used by ``packing.py`` / ``pc2wireframe.py`` / ``module.py``):
``encode(...) -> GaussianLatent`` (``.mode()`` is ``(B, C, M)``),
``decode(z) -> dict(node_tokens, coord, exist_logit)``,
``predict_adjacency`` / ``predict_curve`` heads, ``compute_losses(...)`` and
``forward(..., return_loss=True) -> (loss, parts)``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from scipy.optimize import linear_sum_assignment

from .modules import MLP, PointEmbed
from .vae_curve import GaussianLatent


# ----------------------------------------------------------------------
# encoder / decoder
# ----------------------------------------------------------------------
class GraphEncoder(nn.Module):
    """Node + edge tokens -> ``M`` latent moments via self- then cross-attn."""

    def __init__(
        self,
        *,
        out_channels: int,
        curve_latent_dim: int,
        attn_dim: int,
        num_heads: int,
        depth: int,
        latent_num: int,
        node_embed_hidden: int = 48,
        double_z: bool = True,
    ):
        super().__init__()
        self.node_embed = PointEmbed(hidden_dim=node_embed_hidden, dim=attn_dim)
        self.edge_proj = nn.Linear(curve_latent_dim, attn_dim)
        # token-type embeddings keep node / edge tokens distinguishable.
        self.node_type = nn.Parameter(torch.randn(attn_dim) * 0.02)
        self.edge_type = nn.Parameter(torch.randn(attn_dim) * 0.02)

        self.self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                attn_dim, num_heads, dim_feedforward=attn_dim * 2, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True),
            depth)

        self.latent_queries = nn.Parameter(torch.randn(latent_num, attn_dim) * 0.02)
        self.cross_attn = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                attn_dim, num_heads, dim_feedforward=attn_dim * 2, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True),
            depth)

        oc = 2 * out_channels if double_z else out_channels
        self.project_out = nn.Linear(attn_dim, oc)

    def forward(self, *, node_coords, node_mask, edge_pairs, edge_mask, edge_feat):
        b, n, _ = node_coords.shape

        flat = rearrange(node_coords, "b n c -> (b n) c")
        node_tok = self.node_embed(flat)
        node_tok = rearrange(node_tok, "(b n) d -> b n d", b=b)
        node_tok = node_tok + self.node_type

        edge_tok = self.edge_proj(edge_feat)
        # inject the two endpoint embeddings so each edge "knows" its endpoints.
        idx_a = edge_pairs[..., 0].clamp(min=0)
        idx_b = edge_pairs[..., 1].clamp(min=0)
        gather_a = torch.gather(node_tok, 1, idx_a.unsqueeze(-1).expand(-1, -1, node_tok.shape[-1]))
        gather_b = torch.gather(node_tok, 1, idx_b.unsqueeze(-1).expand(-1, -1, node_tok.shape[-1]))
        edge_tok = edge_tok + gather_a + gather_b + self.edge_type

        tokens = torch.cat([node_tok, edge_tok], dim=1)
        pad = torch.cat([~node_mask, ~edge_mask], dim=1)  # True == ignore

        tokens = self.self_attn(tokens, src_key_padding_mask=pad)

        q = repeat(self.latent_queries, "m d -> b m d", b=b)
        h = self.cross_attn(tgt=q, memory=tokens, memory_key_padding_mask=pad)
        return self.project_out(h)  # (b, M, 2C)


class GraphDecoder(nn.Module):
    """Latent tokens -> node tokens (self- then cross-attn) + heads."""

    def __init__(
        self,
        *,
        in_channels: int,
        attn_dim: int,
        num_heads: int,
        self_depth: int,
        cross_depth: int,
        max_nodes: int,
        adj_dim: int = 128,
        curve_latent_dim: int = 12,
        use_latent_pos_emb: bool = False,
        latent_num: int = 64,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.use_latent_pos_emb = use_latent_pos_emb

        self.proj_in = nn.Linear(in_channels, attn_dim)
        if use_latent_pos_emb:
            self.latent_pos = nn.Parameter(torch.randn(latent_num, attn_dim) * 0.02)
        self.self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                attn_dim, num_heads, dim_feedforward=attn_dim * 2, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True),
            self_depth)

        self.node_queries = nn.Parameter(torch.randn(max_nodes, attn_dim) * 0.02)
        self.cross_attn = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                attn_dim, num_heads, dim_feedforward=attn_dim * 2, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True),
            cross_depth)

        self.coord_head = MLP(in_dim=attn_dim, out_dim=3, expansion_factor=1.0)
        self.exist_head = nn.Linear(attn_dim, 1)
        # symmetric inner-product link-prediction decoder (VGAE-style).
        self.adj_proj = nn.Linear(attn_dim, adj_dim)
        self.adj_scale = nn.Parameter(torch.tensor(1.0 / (adj_dim ** 0.5)))
        self.adj_bias = nn.Parameter(torch.zeros(()))
        # per-edge curve latent from the two endpoint node tokens.
        self.curve_head = MLP(
            in_dim=2 * attn_dim, out_dim=curve_latent_dim, expansion_factor=2.0)

    def forward(self, z):
        b = z.shape[0]
        h = self.proj_in(z)
        if self.use_latent_pos_emb:
            h = h + self.latent_pos
        h = self.self_attn(h)

        q = repeat(self.node_queries, "n d -> b n d", b=b)
        node_tokens = self.cross_attn(tgt=q, memory=h)  # (b, Nq, d)

        coord = self.coord_head(node_tokens)             # (b, Nq, 3)
        exist_logit = self.exist_head(node_tokens).squeeze(-1)  # (b, Nq)
        return {"node_tokens": node_tokens, "coord": coord, "exist_logit": exist_logit}

    def predict_adjacency(self, node_tokens):
        a = self.adj_proj(node_tokens)               # (b, Nq, adj_dim)
        logits = torch.matmul(a, a.transpose(-1, -2)) * self.adj_scale + self.adj_bias
        return logits                                # (b, Nq, Nq) symmetric

    def predict_curve(self, node_tokens, pairs):
        """``pairs`` ``(b, P, 2)`` node-query indices -> curve latent ``(b, P, D)``."""
        d = node_tokens.shape[-1]
        a = torch.gather(node_tokens, 1, pairs[..., 0:1].expand(-1, -1, d))
        bb = torch.gather(node_tokens, 1, pairs[..., 1:2].expand(-1, -1, d))
        return self.curve_head(torch.cat([a, bb], dim=-1))


# ----------------------------------------------------------------------
# full VAE
# ----------------------------------------------------------------------
class AutoencoderKLWireframe(nn.Module):
    """Graph wireframe VAE (node set + adjacency + per-edge curve latent)."""

    def __init__(
        self,
        latent_channels: int = 64,
        wireframe_latent_num: int = 64,
        max_nodes: int = 768,
        max_curves_num: int = 1024,
        curve_latent_dim: int = 12,
        attn_dim: int = 768,
        num_heads: int = 12,
        attn_encoder_depth: int = 4,
        attn_decoder_self_depth: int = 12,
        attn_decoder_cross_depth: int = 2,
        adj_dim: int = 128,
        node_embed_hidden: int = 48,
        # loss weights
        coord_loss_weight: float = 1.0,
        exist_loss_weight: float = 1.0,
        adj_loss_weight: float = 1.0,
        curve_latent_loss_weight: float = 1.0,
        kl_loss_weight: float = 2e-4,
        adj_pos_weight: float = 5.0,
        # matching cost
        match_coord_weight: float = 1.0,
        match_exist_weight: float = 0.1,
        use_latent_pos_emb: bool = True,
        **kwargs,  # tolerate legacy keys
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.wireframe_latent_num = wireframe_latent_num
        self.max_nodes = max_nodes
        self.max_curves_num = max_curves_num
        self.curve_latent_dim = curve_latent_dim

        self.encoder = GraphEncoder(
            out_channels=latent_channels,
            curve_latent_dim=curve_latent_dim,
            attn_dim=attn_dim, num_heads=num_heads, depth=attn_encoder_depth,
            latent_num=wireframe_latent_num, node_embed_hidden=node_embed_hidden,
        )
        self.decoder = GraphDecoder(
            in_channels=latent_channels,
            attn_dim=attn_dim, num_heads=num_heads,
            self_depth=attn_decoder_self_depth, cross_depth=attn_decoder_cross_depth,
            max_nodes=max_nodes, adj_dim=adj_dim, curve_latent_dim=curve_latent_dim,
            use_latent_pos_emb=use_latent_pos_emb, latent_num=wireframe_latent_num,
        )

        self.quant_proj = nn.Linear(2 * latent_channels, 2 * latent_channels)
        self.post_quant_proj = nn.Linear(latent_channels, latent_channels)

        self.coord_loss_weight = coord_loss_weight
        self.exist_loss_weight = exist_loss_weight
        self.adj_loss_weight = adj_loss_weight
        self.curve_latent_loss_weight = curve_latent_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.match_coord_weight = match_coord_weight
        self.match_exist_weight = match_exist_weight
        self.register_buffer("adj_pos_weight", torch.tensor(float(adj_pos_weight)))

    # ------------------------------------------------------------------
    def encode(self, *, node_coords, node_mask, edge_pairs, edge_mask, edge_feat) -> GaussianLatent:
        h = self.encoder(
            node_coords=node_coords, node_mask=node_mask,
            edge_pairs=edge_pairs, edge_mask=edge_mask, edge_feat=edge_feat)
        moments = self.quant_proj(h)                 # (b, M, 2C)
        moments = rearrange(moments, "b n d -> b d n")  # (b, 2C, M)
        return GaussianLatent(moments)

    def decode(self, *, z) -> dict:
        """Latent ``(B, C, M)`` -> decoder dict (node_tokens / coord / exist)."""
        zs = rearrange(z, "b d n -> b n d")
        zs = self.post_quant_proj(zs)
        return self.decoder(zs)

    def predict_adjacency(self, node_tokens):
        return self.decoder.predict_adjacency(node_tokens)

    def predict_curve(self, node_tokens, pairs):
        return self.decoder.predict_curve(node_tokens, pairs)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _match(self, coord, exist_logit, node_coords, node_mask):
        """Hungarian-match predicted nodes to GT nodes per sample.

        Returns a list of ``(q_idx, g_idx)`` long tensors (one per sample),
        where ``q_idx`` are matched query indices and ``g_idx`` the GT node
        indices (into the *valid* GT nodes) they were assigned to.
        """
        b = coord.shape[0]
        exist_prob = torch.sigmoid(exist_logit)
        out = []
        for s in range(b):
            nv = int(node_mask[s].sum().item())
            if nv == 0:
                out.append((coord.new_zeros(0, dtype=torch.long),
                            coord.new_zeros(0, dtype=torch.long)))
                continue
            gt = node_coords[s, :nv]                          # (nv, 3)
            cost = torch.cdist(coord[s], gt) * self.match_coord_weight
            cost = cost - self.match_exist_weight * exist_prob[s].unsqueeze(1)
            qi, gi = linear_sum_assignment(cost.detach().cpu().numpy())
            out.append((torch.as_tensor(qi, dtype=torch.long, device=coord.device),
                        torch.as_tensor(gi, dtype=torch.long, device=coord.device)))
        return out

    # ------------------------------------------------------------------
    def compute_losses(self, dec, targets):
        """Matched node / existence / adjacency / curve losses (no KL)."""
        coord = dec["coord"]
        exist_logit = dec["exist_logit"]
        node_tokens = dec["node_tokens"]
        b, nq, _ = coord.shape
        device = coord.device

        node_coords = targets["node_coords"]
        node_mask = targets["node_mask"]
        edge_pairs = targets["edge_pairs"]
        edge_mask = targets["edge_mask"]
        edge_mu = targets["edge_mu"]
        edge_std = targets.get("edge_std")

        matches = self._match(coord, exist_logit, node_coords, node_mask)
        adj_logits = self.predict_adjacency(node_tokens)  # (b, nq, nq)

        coord_loss = coord.new_zeros(())
        curve_loss = coord.new_zeros(())
        n_coord = 0
        n_curve = 0
        exist_target = torch.zeros(b, nq, device=device)
        adj_target = torch.zeros(b, nq, nq, device=device)

        triu = torch.triu_indices(nq, nq, offset=1, device=device)

        for s in range(b):
            qi, gi = matches[s]
            nv = int(node_mask[s].sum().item())
            if nv == 0:
                continue
            exist_target[s, qi] = 1.0

            # map GT node id -> matched query id
            g2q = torch.full((nv,), -1, dtype=torch.long, device=device)
            g2q[gi] = qi

            # coord loss on matched pairs
            coord_loss = coord_loss + F.l1_loss(
                coord[s, qi], node_coords[s, gi], reduction="sum")
            n_coord += nv

            ne = int(edge_mask[s].sum().item())
            if ne == 0:
                continue
            ep = edge_pairs[s, :ne]                          # (ne, 2) GT node ids
            qa = g2q[ep[:, 0]]
            qb = g2q[ep[:, 1]]

            # adjacency target in query space (symmetric, no self loops)
            adj_target[s, qa, qb] = 1.0
            adj_target[s, qb, qa] = 1.0

            # curve loss: predict per-edge latent from matched endpoint tokens
            pairs = torch.stack([qa, qb], dim=-1).unsqueeze(0)  # (1, ne, 2)
            pred_mu = self.predict_curve(node_tokens[s:s + 1], pairs)[0]  # (ne, D)
            tgt_mu = edge_mu[s, :ne]
            if edge_std is not None:
                std = torch.clamp(edge_std[s, :ne], 0.0, 1.0)
                w = 1.2 - 0.5 * torch.log(std + 1.7183)
                curve_loss = curve_loss + (w * (pred_mu - tgt_mu) ** 2).sum()
            else:
                curve_loss = curve_loss + ((pred_mu - tgt_mu) ** 2).sum()
            n_curve += ne * tgt_mu.shape[-1]

        coord_loss = coord_loss / max(n_coord, 1)
        curve_loss = curve_loss / max(n_curve, 1)
        exist_loss = F.binary_cross_entropy_with_logits(exist_logit, exist_target)

        # adjacency BCE over the upper triangle, with positive up-weighting.
        a_logit = adj_logits[:, triu[0], triu[1]]
        a_tgt = adj_target[:, triu[0], triu[1]]
        adj_loss = F.binary_cross_entropy_with_logits(
            a_logit, a_tgt, pos_weight=self.adj_pos_weight)

        return {
            "coord_loss": coord_loss,
            "exist_loss": exist_loss,
            "adj_loss": adj_loss,
            "curve_latent_loss": curve_loss,
        }

    # ------------------------------------------------------------------
    def forward(
        self,
        *,
        node_coords,
        node_mask,
        edge_pairs,
        edge_mask,
        edge_mu,
        edge_std=None,
        sample_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
        return_loss: bool = False,
        **kwargs,
    ):
        posterior = self.encode(
            node_coords=node_coords, node_mask=node_mask,
            edge_pairs=edge_pairs, edge_mask=edge_mask, edge_feat=edge_mu)
        z = posterior.sample(generator) if sample_posterior else posterior.mode()
        dec = self.decode(z=z)

        if not return_loss:
            return dec

        targets = dict(
            node_coords=node_coords, node_mask=node_mask,
            edge_pairs=edge_pairs, edge_mask=edge_mask,
            edge_mu=edge_mu, edge_std=edge_std)
        parts = self.compute_losses(dec, targets)

        kl_loss = posterior.kl().mean()
        if not sample_posterior:
            kl_loss = 0.0 * kl_loss

        loss = (self.coord_loss_weight * parts["coord_loss"]
                + self.exist_loss_weight * parts["exist_loss"]
                + self.adj_loss_weight * parts["adj_loss"]
                + self.curve_latent_loss_weight * parts["curve_latent_loss"]
                + self.kl_loss_weight * kl_loss)

        all_losses = dict(
            coord_loss=parts["coord_loss"],
            exist_loss=parts["exist_loss"],
            adj_loss=parts["adj_loss"],
            curve_latent_loss=parts["curve_latent_loss"],
            kl_loss=kl_loss,
            mu=posterior.mean.abs().mean().detach(),
            std=posterior.std.mean().detach(),
        )
        return loss, all_losses
