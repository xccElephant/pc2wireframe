"""CLR-Wire wireframe models for staged training.

Two ``nn.Module`` containers, sharing the packing / reconstruction logic in
:class:`~src.models.packing.ClrPackingMixin`:

``ClrWireframeBase`` -- the (always-present) CLR-Wire wireframe VAE + curve VAE.
    Used directly by **stage 2** (train the wireframe VAE with the curve VAE
    frozen). Exposes ``encode_target`` / ``decode_latent`` plus the inherited
    ``graph_to_clr_inputs`` / ``reconstruct``.

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

from .packing import CURVE_LATENT_DIM, ClrPackingMixin
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
        self.max_curves_num = int(wireframe_vae.get("max_curves_num", 128))
        self.max_col_diff = int(wireframe_vae.get("max_col_diff", 6))
        self.max_row_diff = int(wireframe_vae.get("max_row_diff", 32))
        # CLR-Wire per-curve latent = latent_channels x downsampled_len.
        cv = curve_vae or {}
        cv_lat = int(cv.get("latent_channels", 3))
        cv_pts = int(cv.get("sample_points_num", 16))
        n_down = len(cv.get("down_block_types", ("DownBlock1D", "DownBlock1D")))
        self.curve_latent_len = max(1, cv_pts // (2 ** n_down))
        self.curve_latent_dim = cv_lat * self.curve_latent_len
        # Keep the curve VAE configured so latent_channels * downsampled_len == 12
        # (e.g. latent_channels=3, sample_points_num=16, 2 down blocks -> 3*4).
        if self.curve_latent_dim != CURVE_LATENT_DIM:
            raise ValueError(
                f"curve_latent_dim={self.curve_latent_dim} but the CLR-Wire "
                f"wireframe VAE expects {CURVE_LATENT_DIM}. Adjust curve_vae "
                f"(latent_channels * (sample_points_num / 2**n_down) == "
                f"{CURVE_LATENT_DIM})."
            )

    # ------------------------------------------------------------------
    def encode_target(self, xs: torch.Tensor, flag_diffs: torch.Tensor):
        """Encode a GT wireframe to the wireframe-VAE posterior (teacher / eval).

        Returns the ``GaussianLatent``; ``posterior.mode()`` is
        ``(B, latent_channels, latent_num)`` (``b d n``).

        ``xs`` may be the full ``6 + 2*curve_latent_dim`` packing; only the
        ``6 + curve_latent_dim`` (endpoints + curve mu) slice feeds the encoder,
        matching the wireframe VAE's own ``forward``.
        """
        enc_width = 6 + self.curve_latent_dim
        return self.wireframe_vae.encode(
            xs=xs[..., :enc_width], flag_diffs=flag_diffs
        )

    def decode_latent(self, z_bnd: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode a latent ``(B, latent_num, latent_dim)`` -> prediction heads.

        Returns a dict with ``cls`` (curve-count logits), ``segments``
        (per-curve endpoint coords), ``diffs`` (col/row diff logits) and
        ``curve_latent`` (per-curve curve-VAE latent).
        """
        from einops import rearrange

        z_bdn = rearrange(z_bnd, "b n d -> b d n")
        dec = self.wireframe_vae.decode(z=z_bdn)
        if self.wireframe_vae.use_mlp_predict:
            cls, segments, diffs, curve_latent = self.wireframe_vae.mlp_predict(dec)
        else:
            cls, segments, diffs, curve_latent = self.wireframe_vae.linear_predict(dec)
        return {
            "cls": cls,
            "segments": segments,
            "diffs": diffs,
            "curve_latent": curve_latent,
        }


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
