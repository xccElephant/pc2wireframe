"""Point-cloud encoder: frozen Utonia PTv3 backbone + trainable compressor.

Wraps the pre-trained, *frozen* `Utonia <https://huggingface.co/Pointcept/Utonia>`_
PTv3 encoder (an encoder-only Point Transformer V3, vendored under
``src/models/utonia``). Only the latent compressor on top is trained.

Pipeline::

    packed point cloud (coord (P_sum, 3), offset (B,))
        -> per-sample voxel dedup (one point per (batch, grid_coord) voxel)
        -> Utonia PTv3 encoder        # frozen, no_grad, eval (deterministic)
        -> coarse per-voxel features (V, C)
        -> group by sample + pad      # (B, Lmax, C) + key_padding_mask
        -> LatentCompressor (K queries) -> latent z (B, K, D)   # K*D <= 4096

The backbone weights (config + state_dict) come from the Utonia checkpoint, so
``in_channels`` and the coarsest encoder channel count are read from the ckpt
rather than hard-coded. The heavy ``utonia`` import is deferred to construction
so this module can be imported without the spconv / flash-attn stack present.
"""
from __future__ import annotations

import contextlib

import torch
from torch import nn

from .latent_compressor import LatentCompressor


class UtoniaEncoder(nn.Module):
    """Frozen Utonia PTv3 + trainable :class:`LatentCompressor`.

    Args:
        utonia: Utonia checkpoint identifier -- a HuggingFace model name
            (resolved against ``utonia_repo_id``) or a local ``.pth`` path.
        utonia_repo_id: HF repo the named checkpoint is downloaded from.
        download_root: optional cache dir for the downloaded checkpoint.
        grid_size: voxel size used for the per-sample dedup and PTv3 grid
            (data is normalized to ~[-1, 1]).
        latent_num: number of latent tokens ``K`` (default 64).
        latent_dim: per-token latent channels ``D`` (default 64; 64*64=4096).
        compressor_heads: attention heads in the pooling head.
        compressor_layers: ``nn.TransformerDecoder`` layers in the compressor.
        freeze: freeze the PTv3 backbone (``requires_grad=False`` + kept in
            ``eval`` so LayerNorm/DropPath are deterministic, forward in
            ``no_grad``). The compressor is always trainable.
        latent_budget_max: hard cap on ``latent_num * latent_dim``.
    """

    def __init__(
        self,
        utonia: str = "utonia",
        utonia_repo_id: str = "Pointcept/Utonia",
        download_root: str | None = None,
        grid_size: float = 0.01,
        latent_num: int = 64,
        latent_dim: int = 64,
        compressor_heads: int = 8,
        compressor_layers: int = 1,
        freeze: bool = True,
        latent_budget_max: int | None = None,
    ) -> None:
        super().__init__()
        from .utonia.model import PointTransformerV3, load  # deferred heavy import

        # Read the checkpoint once: config drives the backbone dims, so the
        # compressor stays in sync with whatever Utonia variant is loaded.
        ckpt = load(
            utonia, repo_id=utonia_repo_id, download_root=download_root,
            ckpt_only=True,
        )
        cfg = dict(ckpt["config"])
        self.in_channels = int(cfg["in_channels"])
        # Output feature dim: coarsest encoder channel for an encoder-only
        # backbone (Utonia), else the finest decoder channel.
        if bool(cfg.get("enc_mode", False)):
            feat_dim = int(cfg["enc_channels"][-1])
        else:
            feat_dim = int(cfg["dec_channels"][0])
        self.grid_size = float(grid_size)
        self.freeze = bool(freeze)

        self.backbone = PointTransformerV3(**cfg)
        self.backbone.load_state_dict(ckpt["state_dict"])
        if self.freeze:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        self.compressor = LatentCompressor(
            in_dim=feat_dim,
            num_tokens=latent_num,
            latent_dim=latent_dim,
            nhead=compressor_heads,
            num_layers=compressor_layers,
            latent_budget_max=latent_budget_max,
        )

    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> "UtoniaEncoder":
        """Keep the frozen backbone in ``eval`` regardless of module mode."""
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    @staticmethod
    def _offset2batch(offset: torch.Tensor) -> torch.Tensor:
        counts = torch.diff(
            offset, prepend=offset.new_zeros(1)
        )
        return torch.arange(
            offset.shape[0], device=offset.device, dtype=torch.long
        ).repeat_interleave(counts)

    def _voxel_dedup(
        self, coord: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One point per ``(batch, grid_coord)`` voxel (Utonia GridSample style).

        Returns the kept ``coord``, the integer ``grid_coord`` and the kept
        ``batch`` id, with rows sorted ascending by sample so a CSR ``offset``
        can be recovered downstream.
        """
        # Per-sample min so grid indices start at 0 within every sample.
        b = int(batch.max().item()) + 1 if batch.numel() else 0
        coord_min = coord.new_zeros(b, 3)
        coord_min.scatter_reduce_(
            0, batch[:, None].expand(-1, 3), coord, reduce="amin",
            include_self=False,
        )
        grid_coord = torch.div(
            coord - coord_min[batch], self.grid_size, rounding_mode="trunc"
        ).long()

        keys = torch.cat([batch[:, None], grid_coord], dim=1)  # (N, 4)
        _, inverse = torch.unique(keys, dim=0, return_inverse=True)
        # First occurrence per unique voxel (sorted by unique id -> by sample).
        order = torch.argsort(inverse, stable=True)
        inv_sorted = inverse[order]
        first = torch.ones_like(inv_sorted, dtype=torch.bool)
        first[1:] = inv_sorted[1:] != inv_sorted[:-1]
        keep = order[first]
        return coord[keep], grid_coord[keep], batch[keep]

    @staticmethod
    def _group_by_batch(
        feat: torch.Tensor, batch: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack per-voxel features ``(V, C)`` into ``(B, Lmax, C)``.

        Returns the padded tensor and a bool ``key_padding_mask (B, Lmax)``
        with ``True`` at padded positions (``nn.MultiheadAttention`` convention).
        """
        device = feat.device
        order = torch.argsort(batch, stable=True)
        batch = batch[order]
        feat = feat[order]

        counts = torch.bincount(batch, minlength=batch_size)
        lmax = int(counts.max().item()) if counts.numel() else 0
        ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

        arange = torch.arange(feat.shape[0], device=device)
        within = arange - ptr[batch]

        padded = feat.new_zeros(batch_size, lmax, feat.shape[-1])
        mask = torch.ones(batch_size, lmax, dtype=torch.bool, device=device)
        padded[batch, within] = feat
        mask[batch, within] = False
        return padded, mask

    def forward(
        self, coord: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        """Encode a packed point cloud to the latent ``(B, latent_num, latent_dim)``.

        Args:
            coord: ``(P_sum, 3)`` concatenated batch points.
            offset: ``(B,)`` cumulative per-sample point counts.
        """
        b = int(offset.shape[0])
        coord = coord.reshape(-1, 3).contiguous()
        offset = offset.to(device=coord.device, dtype=torch.long)
        batch = self._offset2batch(offset)

        coord_d, grid_coord_d, batch_d = self._voxel_dedup(coord, batch)
        offset_d = torch.bincount(batch_d, minlength=b).cumsum(0)

        if self.in_channels == 3:
            feat = coord_d
        else:
            pad = coord_d.new_zeros(coord_d.shape[0], self.in_channels - 3)
            feat = torch.cat([coord_d, pad], dim=-1)

        data_dict = {
            "coord": coord_d,
            "grid_coord": grid_coord_d.int(),
            "feat": feat,
            "offset": offset_d,
            "grid_size": self.grid_size,
        }

        ctx = torch.no_grad() if self.freeze else contextlib.nullcontext()
        with ctx:
            point = self.backbone(data_dict)
        feat_out = point["feat"]
        batch_out = point["batch"]

        tokens, key_padding_mask = self._group_by_batch(feat_out, batch_out, b)
        return self.compressor(tokens, key_padding_mask=key_padding_mask)


__all__ = ["UtoniaEncoder"]
