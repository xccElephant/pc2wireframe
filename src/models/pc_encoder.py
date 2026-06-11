"""Point-cloud encoder: PTv3 backbone + latent-compressor head.

Maps a surface point cloud ``(B, N, 3)`` to the fixed-length wireframe latent
distribution ``Z_W`` that the CLR-Wire wireframe decoder consumes.

Pipeline::

    point cloud (B, N, 3)
        -> PointTransformerV3            # serialized sparse attention
        -> per-voxel features (V, C)     # V varies per sample
        -> group by sample + pad         # (B, Lmax, C) + key_padding_mask
        -> LatentCompressor (K queries)  # cross-attn pooling -> (B, K, D)
        -> mu / logvar                   # K * D <= 4096 floats

PTv3 hyperparameters are exposed so they can be tuned from the config. The
PTv3 import is deferred to construction time so this module can be imported
without the (heavy) spconv / flash-attn stack present.
"""
from __future__ import annotations

import torch
from torch import nn

from .latent_compressor import LatentCompressor


class PCEncoder(nn.Module):
    """PTv3 + latent-compressor point-cloud encoder.

    Args:
        in_channels: input point feature channels (3 = xyz; 6 = xyz+normals).
        grid_size: voxel size used by PTv3 serialization/sparsification.
        cls_mode: PTv3 encoder-only (True) vs encoder+decoder/unpool (False).
            ``False`` keeps denser, better-localised features (good for
            corners/edges); ``True`` is cheaper.
        enc_*/dec_*: PTv3 stage configs (depths / channels / heads / patch).
        latent_num: number of latent tokens ``K`` (= ``wireframe_latent_num``).
        latent_dim: per-token latent channels ``D`` (= ``latent_channels``).
        compressor_heads: attention heads in the pooling head.
        variational: VAE-style latent (predict logvar + reparam) if True.
        latent_budget_max: hard float cap on ``latent_num * latent_dim``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        grid_size: float = 0.01,
        cls_mode: bool = False,
        # ----- PTv3 encoder stages -----
        enc_depths: tuple[int, ...] = (2, 2, 2, 6, 2),
        enc_channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        enc_num_head: tuple[int, ...] = (2, 4, 8, 16, 32),
        enc_patch_size: tuple[int, ...] = (1024, 1024, 1024, 1024, 1024),
        # ----- PTv3 decoder stages (used when cls_mode=False) -----
        dec_depths: tuple[int, ...] = (2, 2, 2, 2),
        dec_channels: tuple[int, ...] = (64, 64, 128, 256),
        dec_num_head: tuple[int, ...] = (4, 4, 8, 16),
        dec_patch_size: tuple[int, ...] = (1024, 1024, 1024, 1024),
        stride: tuple[int, ...] = (2, 2, 2, 2),
        enable_flash: bool = True,
        # ----- latent head -----
        latent_num: int = 64,
        latent_dim: int = 64,
        compressor_heads: int = 8,
        variational: bool = True,
        latent_budget_max: int | None = None,
    ) -> None:
        super().__init__()
        from .ptv3 import PointTransformerV3  # deferred heavy import

        self.in_channels = int(in_channels)
        self.grid_size = float(grid_size)
        self.cls_mode = bool(cls_mode)

        self.backbone = PointTransformerV3(
            in_channels=in_channels,
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            enable_flash=enable_flash,
            cls_mode=cls_mode,
        )
        # Output feature dim: last encoder channel (cls_mode) else first
        # decoder channel (the unpooled / finest decoder stage).
        feat_dim = enc_channels[-1] if cls_mode else dec_channels[0]

        self.compressor = LatentCompressor(
            in_dim=feat_dim,
            num_tokens=latent_num,
            latent_dim=latent_dim,
            nhead=compressor_heads,
            variational=variational,
            latent_budget_max=latent_budget_max,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _group_by_batch(
        feat: torch.Tensor, batch: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack per-voxel features ``(V, C)`` into ``(B, Lmax, C)``.

        Returns the padded tensor and a bool ``key_padding_mask (B, Lmax)``
        with ``True`` at padded positions (the convention used by
        ``nn.MultiheadAttention``).
        """
        device = feat.device
        # Sort so each sample's rows are contiguous and ascending.
        order = torch.argsort(batch, stable=True)
        batch = batch[order]
        feat = feat[order]

        counts = torch.bincount(batch, minlength=batch_size)
        lmax = int(counts.max().item()) if counts.numel() else 0
        ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

        arange = torch.arange(feat.shape[0], device=device)
        within = arange - ptr[batch]  # index of each row within its sample

        padded = feat.new_zeros(batch_size, lmax, feat.shape[-1])
        mask = torch.ones(batch_size, lmax, dtype=torch.bool, device=device)
        padded[batch, within] = feat
        mask[batch, within] = False
        return padded, mask

    def _to_data_dict(self, point_cloud: torch.Tensor) -> dict:
        """Build the PTv3 input dict from a ``(B, N, 3)`` point cloud."""
        b, n, _ = point_cloud.shape
        coord = point_cloud.reshape(-1, 3).contiguous()
        if self.in_channels == 3:
            feat = coord
        else:
            # Extra channels (e.g. normals) are expected to be concatenated by
            # the caller; fall back to padding with zeros if absent.
            pad = coord.new_zeros(coord.shape[0], self.in_channels - 3)
            feat = torch.cat([coord, pad], dim=-1)
        offset = torch.arange(
            n, n * b + 1, n, device=point_cloud.device, dtype=torch.long
        )
        return {
            "coord": coord,
            "feat": feat,
            "offset": offset,
            "grid_size": self.grid_size,
        }

    def forward(
        self, point_cloud: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode ``point_cloud (B, N, 3)`` to the latent distribution.

        Returns ``(mu, logvar)`` each ``(B, latent_num, latent_dim)``;
        ``logvar`` is ``None`` for a deterministic (non-variational) head.
        """
        b = point_cloud.shape[0]
        point = self.backbone(self._to_data_dict(point_cloud))
        feat = point["feat"]
        batch = point["batch"]
        tokens, key_padding_mask = self._group_by_batch(feat, batch, b)
        return self.compressor(tokens, key_padding_mask=key_padding_mask)


__all__ = ["PCEncoder"]
