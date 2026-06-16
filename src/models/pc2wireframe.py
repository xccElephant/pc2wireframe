"""End-to-end point-cloud -> wireframe model (the single trained stage).

Pipeline::

    point cloud (B, N, 3)
        --PTv3 + LatentCompressor-->  tokenized latent Z  (B, K, D)   # K*D <= 4096
        --WireframeDecoder-->
              node set:   per-query (coord, existence)
              relation:   global [rln] token(s)
        --RelationEdgeHead over candidate vertex pairs-->
              per-pair (alive existence, curve latent)
        --frozen curve VAE decoder--> per-edge canonical polyline
        --denormalise onto predicted endpoints--> wireframe

Vertices stay a DETR-style node set (Hungarian-matched, deeply supervised);
edges are predicted by scoring candidate *vertex pairs* with a Relationformer
relation head (``edge_decoder.RelationEdgeHead``) rather than a separate edge
query set. The endpoints of an edge are the two vertex indices of the pair, so
there is no endpoint vocabulary and no edge matching.

The per-curve ``CurveVAE`` (``AutoencoderKL1D``) is reused **frozen** from
stage 1: its decoder turns a predicted per-edge curve latent into a residual
polyline on top of the endpoint-interpolation baseline (canonical frame), which
is then denormalised onto the predicted vertices. Only the PTv3 encoder, the
latent compressor, the wireframe decoder and the relation edge head are trained.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .edge_decoder import RelationEdgeHead, shortlist_pairs_infer
from .packing import decode_curve_latent
from .pc_encoder import PCEncoder
from .wireframe_decoder import WireframeDecoder


class PC2WireframeModel(nn.Module):
    """Point cloud -> tokenized latent -> wireframe (node set + pairwise edges)."""

    def __init__(
        self,
        pc_encoder: dict[str, Any],
        decoder: dict[str, Any] | None = None,
        curve_vae: dict[str, Any] | None = None,
        edge_head: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        from .vae import AutoencoderKL1D

        self.pc_encoder = PCEncoder(**pc_encoder)
        self.curve_vae = AutoencoderKL1D(**(curve_vae or {}))

        # per-edge curve latent size = curve_vae.latent_channels * latent_len.
        curve_latent_dim = (
            int(self.curve_vae.config.latent_channels) * int(self.curve_vae.latent_len)
        )
        dec_kwargs = dict(decoder or {})
        dec_kwargs.setdefault("latent_token_dim", int(pc_encoder["latent_dim"]))
        dec_kwargs.setdefault("num_latent_tokens", int(pc_encoder["latent_num"]))
        self.decoder = WireframeDecoder(**dec_kwargs)
        self.curve_latent_dim = curve_latent_dim

        # Relationformer-style pairwise edge head (alive + curve latent).
        self.edge_head = RelationEdgeHead(
            d_model=self.decoder.d_model,
            curve_latent_dim=curve_latent_dim,
            **(edge_head or {}),
        )

    # ------------------------------------------------------------------
    def encode_pc(
        self, coord: torch.Tensor, offset: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Packed point cloud (``coord (P_sum,3)``, ``offset (B,)``) ->
        latent ``(mu, logvar)`` in ``b k d``."""
        return self.pc_encoder(coord, offset)

    def forward(
        self, coord: torch.Tensor, offset: torch.Tensor, sample: bool = False
    ) -> dict[str, Any]:
        """Packed point cloud -> latent -> node predictions + relation token."""
        mu, logvar = self.encode_pc(coord, offset)
        if sample and logvar is not None:
            z = self.pc_encoder.compressor.reparameterize(mu, logvar)
        else:
            z = mu
        preds = self.decoder(z)
        return {"z": z, "mu": mu, "logvar": logvar, "preds": preds}

    # ------------------------------------------------------------------
    def score_pairs(
        self,
        node_tokens: torch.Tensor,   # (V, d_model)
        node_pos: torch.Tensor,      # (V, 3)
        rln_token: torch.Tensor,     # (R, d_model) or (d_model,)
        pair_idx: torch.Tensor,      # (P, 2) local vertex indices
    ) -> dict[str, torch.Tensor]:
        """Score candidate vertex pairs with the relation edge head.

        Operates on a *single* sample (the candidate set differs per sample).
        Returns ``{"alive_logit": (P,), "curve_latent": (P, Dc)}``.
        """
        return self.edge_head(node_tokens, node_pos, rln_token, pair_idx)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct(
        self,
        preds: dict[str, torch.Tensor],
        *,
        vertex_threshold: float = 0.5,
        edge_threshold: float = 0.5,
        knn_k: int = 16,
        infer_max_pairs: int = 40000,
        topk_pairs: int = 0,
        num_points: int = 32,
        recon_curves: bool = True,
    ) -> list[dict[str, Any]]:
        """Decoder predictions -> explicit wireframes (one dict per sample).

          * ``sigmoid(node_exist) > vertex_threshold``  -> alive vertices;
          * candidate pairs = :func:`shortlist_pairs_infer` over the alive
            vertices (all pairs for small ``V``, kNN shortlist otherwise);
          * each pair's ``sigmoid(alive_logit)`` from the relation head; keep
            pairs above ``edge_threshold`` (optionally cap at ``topk_pairs``);
          * orient each kept edge by the coordinate rule (A = lexicographically
            smaller endpoint), decode its curve latent through the frozen curve
            VAE and denormalise onto ``[A, B]``.
        """
        from .vae.recon_utils import denorm_curves

        coord = preds["coord"]
        node_exist = torch.sigmoid(preds["node_exist_logit"])
        node_tokens = preds["node_tokens"]
        rln_token = preds["rln_token"]
        b = coord.shape[0]
        device = coord.device

        out: list[dict[str, Any]] = []
        for s in range(b):
            alive = (node_exist[s] > vertex_threshold).nonzero(as_tuple=True)[0]
            if alive.numel() < 2:
                alive = torch.topk(
                    node_exist[s], k=min(8, node_exist.shape[1])).indices
            verts = coord[s, alive]                       # (V, 3)
            v = alive.numel()
            if v < 2:
                out.append(self._empty(verts))
                continue

            cand_tokens = node_tokens[s, alive]           # (V, d_model)
            cand_rln = rln_token[s]                        # (R, d_model)
            pair_idx = shortlist_pairs_infer(
                verts, max_pairs=int(infer_max_pairs), knn_k=int(knn_k))
            if pair_idx.shape[0] == 0:
                out.append(self._empty(verts))
                continue

            res = self.score_pairs(cand_tokens, verts, cand_rln, pair_idx)
            alive_prob = torch.sigmoid(res["alive_logit"])
            keep = alive_prob > edge_threshold
            if topk_pairs and int(keep.sum()) > int(topk_pairs):
                top = torch.topk(alive_prob, k=int(topk_pairs)).indices
                keep = torch.zeros_like(keep)
                keep[top] = True
            if not bool(keep.any()):
                out.append(self._empty(verts))
                continue

            pair_idx = pair_idx[keep]
            la = pair_idx[:, 0]
            lb = pair_idx[:, 1]
            curve_latent = (
                res["curve_latent"][keep] if "curve_latent" in res else None)

            # Orient each edge so endpoint A is the lexicographically smaller
            # predicted coordinate (matches the dataset's curve orientation).
            pa = verts[la]
            pb = verts[lb]
            swap = self._lex_greater(pa, pb)              # (E,) bool
            a_idx = torch.where(swap, lb, la)
            b_idx = torch.where(swap, la, lb)

            verts_np = verts.detach().cpu().numpy().astype(np.float32)
            edge_index = torch.stack([a_idx, b_idx], dim=-1).cpu().numpy(
                ).astype(np.int64)
            sample_out: dict[str, Any] = {
                "vertices": verts_np,
                "edge_index": edge_index,
                "num_vertices": int(v),
                "num_edges": int(edge_index.shape[0]),
            }

            if recon_curves and curve_latent is not None:
                canon = decode_curve_latent(
                    self.curve_vae, curve_latent,
                    num_points=num_points, pin_endpoints=True,
                ).detach().cpu().numpy()
                corners = np.stack(
                    [verts_np[edge_index[:, 0]], verts_np[edge_index[:, 1]]],
                    axis=1)
                curves = denorm_curves(canon, corners)
                if curves is not None:
                    sample_out["edge_points"] = curves
            out.append(sample_out)
        return out

    @staticmethod
    def _lex_greater(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Per-row ``a > b`` under (x, y, z) lexicographic order. ``a, b: (E, 3)``."""
        gt = a > b
        lt = a < b
        # first differing coordinate decides; ties fall through to False.
        x_gt, y_gt, z_gt = gt[:, 0], gt[:, 1], gt[:, 2]
        x_eq = ~gt[:, 0] & ~lt[:, 0]
        y_eq = ~gt[:, 1] & ~lt[:, 1]
        return x_gt | (x_eq & (y_gt | (y_eq & z_gt)))

    @staticmethod
    def _empty(verts: torch.Tensor) -> dict[str, Any]:
        return {
            "vertices": verts.detach().cpu().numpy().astype(np.float32),
            "edge_index": np.zeros((0, 2), dtype=np.int64),
            "num_vertices": int(verts.shape[0]),
            "num_edges": 0,
        }


__all__ = ["PC2WireframeModel"]
