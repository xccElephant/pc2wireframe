"""End-to-end point-cloud -> wireframe model (the single trained stage).

Pipeline::

    point cloud (B, N, 3)
        --PTv3 + LatentCompressor-->  tokenized latent Z  (B, K, D)   # K*D <= 4096
        --WireframeDecoder-->
              node set:   per-query (coord, existence)
              edge set:   per-query (existence, endpoint-A/B distribution,
                                     curve latent)
        --frozen curve VAE decoder--> per-edge canonical polyline
        --denormalise onto predicted endpoints--> wireframe

The per-curve ``CurveVAE`` (``AutoencoderKL1D``) is reused **frozen** from
stage 1: its decoder turns a predicted per-edge curve latent into a residual
polyline on top of the endpoint-interpolation baseline (canonical frame), which
is then denormalised onto the predicted vertices. Only the PTv3 encoder, the
latent compressor and the wireframe decoder are trained.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .packing import decode_curve_latent
from .pc_encoder import PCEncoder
from .wireframe_decoder import WireframeDecoder


class PC2WireframeModel(nn.Module):
    """Point cloud -> tokenized latent -> wireframe (node set + edge set)."""

    def __init__(
        self,
        pc_encoder: dict[str, Any],
        decoder: dict[str, Any] | None = None,
        curve_vae: dict[str, Any] | None = None,
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
        dec_kwargs["curve_latent_dim"] = curve_latent_dim
        self.decoder = WireframeDecoder(**dec_kwargs)
        self.curve_latent_dim = curve_latent_dim

    # ------------------------------------------------------------------
    def encode_pc(
        self, point_cloud: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Point cloud ``(B, N, 3)`` -> latent ``(mu, logvar)`` in ``b k d``."""
        return self.pc_encoder(point_cloud)

    def forward(
        self, point_cloud: torch.Tensor, sample: bool = False
    ) -> dict[str, Any]:
        """Point cloud -> latent -> decoder predictions."""
        mu, logvar = self.encode_pc(point_cloud)
        if sample and logvar is not None:
            z = self.pc_encoder.compressor.reparameterize(mu, logvar)
        else:
            z = mu
        preds = self.decoder(z)
        return {"z": z, "mu": mu, "logvar": logvar, "preds": preds}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct(
        self,
        preds: dict[str, torch.Tensor],
        *,
        vertex_threshold: float = 0.5,
        edge_threshold: float = 0.5,
        num_points: int = 32,
        max_edges: int | None = None,
        recon_curves: bool = True,
    ) -> list[dict[str, Any]]:
        """Decoder predictions -> explicit wireframes (one dict per sample).

          * ``sigmoid(node_exist) > vertex_threshold``      -> alive vertices;
          * ``sigmoid(edge_exist) > edge_threshold``         -> alive edges;
          * each alive edge's endpoints = ``argmax`` of the two endpoint
            distributions restricted to the alive vertices;
          * per-edge curve latent -> frozen curve VAE canonical decode ->
            denormalise onto the (coordinate-oriented) predicted endpoints.
        """
        from .vae.recon_utils import denorm_curves

        coord = preds["coord"]
        node_exist = torch.sigmoid(preds["node_exist_logit"])
        edge_exist = torch.sigmoid(preds["edge_exist_logit"])
        ep_a = preds["ep_a_logits"]
        ep_b = preds["ep_b_logits"]
        curve_latent = preds["curve_latent"]
        b = coord.shape[0]
        device = coord.device
        if max_edges is None:
            max_edges = self.decoder.num_edge_queries

        out: list[dict[str, Any]] = []
        for s in range(b):
            alive = (node_exist[s] > vertex_threshold).nonzero(as_tuple=True)[0]
            if alive.numel() < 2:
                alive = torch.topk(
                    node_exist[s], k=min(8, node_exist.shape[1])).indices
            verts = coord[s, alive]                       # (V, 3)
            v = alive.numel()
            # global query id -> local vertex index (only alive queries valid).
            g2l = torch.full(
                (node_exist.shape[1],), -1, dtype=torch.long, device=device)
            g2l[alive] = torch.arange(v, device=device)

            e_keep = (edge_exist[s] > edge_threshold).nonzero(as_tuple=True)[0]
            if e_keep.numel() == 0:
                out.append(self._empty(verts))
                continue
            if e_keep.numel() > max_edges:
                top = torch.topk(edge_exist[s, e_keep], k=max_edges).indices
                e_keep = e_keep[top]

            # endpoints restricted to alive vertices.
            mask = torch.full((node_exist.shape[1],), float("-inf"), device=device)
            mask[alive] = 0.0
            a_q = (ep_a[s, e_keep] + mask).argmax(dim=-1)
            b_q = (ep_b[s, e_keep] + mask).argmax(dim=-1)
            la = g2l[a_q]
            lb = g2l[b_q]
            ok = (la >= 0) & (lb >= 0) & (la != lb)
            if not bool(ok.any()):
                out.append(self._empty(verts))
                continue
            e_keep, la, lb = e_keep[ok], la[ok], lb[ok]

            # dedupe undirected pairs (keep first occurrence).
            lo = torch.minimum(la, lb)
            hi = torch.maximum(la, lb)
            key = lo * v + hi
            _, first = np.unique(key.cpu().numpy(), return_index=True)
            sel = torch.as_tensor(np.sort(first), device=device)
            e_keep, la, lb = e_keep[sel], la[sel], lb[sel]

            verts_np = verts.detach().cpu().numpy().astype(np.float32)
            edge_index = torch.stack([la, lb], dim=-1).cpu().numpy().astype(np.int64)
            sample_out: dict[str, Any] = {
                "vertices": verts_np,
                "edge_index": edge_index,
                "num_vertices": int(v),
                "num_edges": int(edge_index.shape[0]),
            }

            if recon_curves:
                # canonical curve is in the predicted A -> B (start -> end)
                # order, so denormalise straight onto the [A, B] endpoints.
                canon = decode_curve_latent(
                    self.curve_vae, curve_latent[s, e_keep],
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
    def _empty(verts: torch.Tensor) -> dict[str, Any]:
        return {
            "vertices": verts.detach().cpu().numpy().astype(np.float32),
            "edge_index": np.zeros((0, 2), dtype=np.int64),
            "num_vertices": int(verts.shape[0]),
            "num_edges": 0,
        }


__all__ = ["PC2WireframeModel"]
