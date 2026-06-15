"""CLR-Wire wireframe models for staged training.

Two ``nn.Module`` containers, sharing the packing / reconstruction logic in
:class:`~src.models.packing.ClrPackingMixin`:

``ClrWireframeBase`` -- the (always-present) CLR-Wire wireframe VAE + curve VAE.
    Used directly by **stage 2** (train the wireframe VAE with the curve VAE
    frozen). Exposes ``encode_target`` / ``decode_latent`` plus the inherited
    ``graph_to_node_inputs`` / ``reconstruct_graph``.

``PC2WireframeModel`` -- ``ClrWireframeBase`` + a point-cloud encoder.
    Used by **stage 3** (train point cloud -> latent with both VAEs frozen)::

        point cloud --PTv3+LatentCompressor--> Z_W --CLR-Wire decoder-->
            (curve count, endpoints, differential adjacency, curve latents)
            --CLR-Wire curve decoder--> 3D curves  => wireframe

The point-cloud encoder replaces CLR-Wire's *wireframe* encoder: at inference we
never see the GT wireframe, so we regress / sample the latent from the point
cloud, then reuse the (stage-2 pretrained) CLR-Wire decoders.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .packing import ClrPackingMixin
from .pc_encoder import PCEncoder


class ClrWireframeBase(ClrPackingMixin, nn.Module):
    """CLR-Wire wireframe VAE + curve VAE container (stage 2 model)."""

    def __init__(
        self,
        wireframe_vae: dict[str, Any],
        curve_vae: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        from .vae import AutoencoderKL1D, AutoencoderKLWireframe

        self.wireframe_vae = AutoencoderKLWireframe(**wireframe_vae)
        self.curve_vae = AutoencoderKL1D(**(curve_vae or {}))
        self._init_clr_config(wireframe_vae, curve_vae)

    # ------------------------------------------------------------------
    def _init_clr_config(
        self, wireframe_vae: dict[str, Any], curve_vae: dict[str, Any] | None
    ) -> None:
        """Cache the config needed by the packing / reconstruction code."""
        self.max_curves_num = int(wireframe_vae.get("max_curves_num", 1024))
        self.max_nodes = int(wireframe_vae.get("max_nodes", 768))
        # per-curve latent = latent_channels x downsampled_len.
        cv = curve_vae or {}
        cv_lat = int(cv.get("latent_channels", 3))
        cv_pts = int(cv.get("sample_points_num", 32))
        n_down = len(cv.get(
            "down_block_types", ("DownBlock1D", "DownBlock1D", "DownBlock1D")))
        self.curve_latent_len = max(1, cv_pts // (2 ** n_down))
        self.curve_latent_dim = cv_lat * self.curve_latent_len
        # The wireframe VAE's curve head out_dim must match this.
        vae_curve_dim = int(wireframe_vae.get("curve_latent_dim", self.curve_latent_dim))
        if self.curve_latent_dim != vae_curve_dim:
            raise ValueError(
                f"curve_latent_dim mismatch: curve VAE produces "
                f"{self.curve_latent_dim} but wireframe VAE expects "
                f"{vae_curve_dim}. Set wireframe_vae.curve_latent_dim = "
                f"latent_channels * (sample_points_num / 2**n_down)."
            )

    # ------------------------------------------------------------------
    def encode_target(self, targets: dict[str, torch.Tensor]):
        """Encode a GT wireframe to the wireframe-VAE posterior (teacher / eval).

        ``targets`` is the dict returned by ``graph_to_node_inputs``. Returns the
        ``GaussianLatent``; ``posterior.mode()`` is ``(B, latent_channels,
        latent_num)`` (``b d n``).
        """
        return self.wireframe_vae.encode(
            node_coords=targets["node_coords"],
            node_mask=targets["node_mask"],
            edge_pairs=targets["edge_pairs"],
            edge_mask=targets["edge_mask"],
            edge_feat=targets["edge_mu"],
        )

    def decode_latent(self, z_bnd: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode a latent ``(B, latent_num, latent_dim)`` -> decoder dict.

        Returns ``{node_tokens, coord, exist_logit}`` (adjacency / curve latent
        are produced lazily by the wireframe VAE heads from ``node_tokens``).
        """
        from einops import rearrange

        z_bdn = rearrange(z_bnd, "b n d -> b d n")
        return self.wireframe_vae.decode(z=z_bdn)


class PC2WireframeModel(ClrWireframeBase):
    """Point cloud -> latent -> wireframe, reusing the CLR-Wire decoders."""

    def __init__(
        self,
        pc_encoder: dict[str, Any],
        wireframe_vae: dict[str, Any],
        curve_vae: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(wireframe_vae=wireframe_vae, curve_vae=curve_vae)
        self.pc_encoder = PCEncoder(**pc_encoder)
        self._check_latent_dims(pc_encoder, wireframe_vae)

    @staticmethod
    def _check_latent_dims(
        pc_encoder: dict[str, Any], wireframe_vae: dict[str, Any]
    ) -> None:
        enc_num = pc_encoder.get("latent_num", 64)
        enc_dim = pc_encoder.get("latent_dim", 64)
        vae_num = wireframe_vae.get("wireframe_latent_num", 64)
        vae_dim = wireframe_vae.get("latent_channels", 16)
        if (enc_num, enc_dim) != (vae_num, vae_dim):
            raise ValueError(
                "Latent shape mismatch between PC encoder and wireframe VAE: "
                f"encoder=({enc_num}, {enc_dim}) vs "
                f"vae=({vae_num}, {vae_dim}). They must match so the decoder "
                f"can consume the predicted latent."
            )

    # ------------------------------------------------------------------
    def encode_pc(
        self, point_cloud: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Point cloud ``(B, N, 3)`` -> latent ``(mu, logvar)`` in ``b n d``."""
        return self.pc_encoder(point_cloud)

    def forward(
        self, point_cloud: torch.Tensor, sample: bool = False
    ) -> dict[str, Any]:
        """Full feed-forward: point cloud -> latent -> decoder predictions."""
        mu, logvar = self.encode_pc(point_cloud)
        if sample and logvar is not None:
            z = self.pc_encoder.compressor.reparameterize(mu, logvar)
        else:
            z = mu
        preds = self.decode_latent(z)
        return {"z": z, "mu": mu, "logvar": logvar, "preds": preds}


__all__ = ["ClrWireframeBase", "PC2WireframeModel"]
