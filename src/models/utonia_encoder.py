"""Point-cloud encoder: frozen Utonia PTv3 backbone + trainable compressor.

Wraps the pre-trained, *frozen* `Utonia <https://huggingface.co/Pointcept/Utonia>`_
PTv3 encoder (an encoder-only Point Transformer V3, vendored under
``src/models/utonia``). Only the latent compressor on top is trained.

Pipeline::

    packed point cloud (coord (P_sum, 3), offset (B,))
        -> per-sample voxel dedup (one point per (batch, grid_coord) voxel)
        -> Utonia PTv3 encoder        # frozen, no_grad, eval (deterministic)
        -> coarsest per-voxel features (enc4 resolution)
        -> upsample back to the PTv3 input (dedup) resolution by walking the
           GridPooling ``pooling_parent`` / ``pooling_inverse`` chain
        -> scatter to *every original input point* via the dedup inverse map
        -> group by sample + pad      # (B, P_max, C) + key_padding_mask
        -> LatentCompressor (K queries) -> latent z (B, K, D)   # K*D <= 4096

Recovering a *per-input-point* feature (instead of pooling the coarse voxel
tokens directly) gives the compressor a dense, full-resolution view of the
surface so the 16x256 latent can resolve fine wireframe geometry.

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
            (data lives in the raw, ~unit frame).
        latent_num: number of latent tokens ``K`` (default 16).
        latent_dim: per-token latent channels ``D`` (default 256; 16*256=4096).
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
        latent_num: int = 16,
        latent_dim: int = 256,
        compressor_heads: int = 8,
        compressor_layers: int = 1,
        freeze: bool = True,
        latent_budget_max: int | None = None,
    ) -> None:
        super().__init__()
        from .utonia.model import PointTransformerV3, load  # deferred heavy import

        ckpt = load(
            utonia, repo_id=utonia_repo_id, download_root=download_root,
            ckpt_only=True,
        )
        cfg = dict(ckpt["config"])
        self.in_channels = int(cfg["in_channels"])
        # enc_mode returns the *coarsest* encoder features; walking the
        # pooling_inverse chain broadcasts them back to the input *resolution*
        # but leaves the channel count untouched, so the per-point feature width
        # is enc_channels[-1] (NOT enc_channels[0]).
        self.feat_dim = int(cfg["enc_channels"][-1])
        self.grid_size = float(grid_size)
        self.freeze = bool(freeze)

        self.backbone = PointTransformerV3(**cfg)
        self.backbone.load_state_dict(ckpt["state_dict"])
        if self.freeze:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        self.compressor = LatentCompressor(
            in_dim=self.feat_dim,
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
        counts = torch.diff(offset, prepend=offset.new_zeros(1))
        return torch.arange(
            offset.shape[0], device=offset.device, dtype=torch.long
        ).repeat_interleave(counts)

    def _voxel_dedup(
        self, coord: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One point per ``(batch, grid_coord)`` voxel (Utonia GridSample style).

        Returns the kept ``coord``, integer ``grid_coord``, kept ``batch`` id
        (rows sorted ascending by unique voxel id), and the **dedup inverse map**
        ``(P_orig,)`` from every original input point to its kept-voxel row, so
        the per-voxel backbone feature can be scattered back to all points.
        """
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
        _, inverse, counts = torch.unique(
            keys, dim=0, return_inverse=True, return_counts=True)
        # First occurrence per unique voxel (sorted by unique id -> by sample).
        order = torch.argsort(inverse, stable=True)
        inv_sorted = inverse[order]
        first = torch.ones_like(inv_sorted, dtype=torch.bool)
        first[1:] = inv_sorted[1:] != inv_sorted[:-1]
        keep = order[first]
        # ``inverse`` already maps original point -> unique voxel id, and the
        # kept rows are ordered by unique voxel id, so ``inverse`` doubles as the
        # original-point -> kept-row scatter map.
        return coord[keep], grid_coord[keep], batch[keep], inverse

    @staticmethod
    def _upsample_to_input(point) -> torch.Tensor:
        """Propagate coarse encoder features back to the PTv3 input resolution.

        Utonia (``enc_mode=True``) returns the coarsest per-voxel features. Each
        ``GridPooling`` recorded ``pooling_inverse`` (parent point -> pooled
        cluster) and ``pooling_parent`` (the pre-pooling :class:`Point`); walking
        that chain broadcasts the coarse feature back to the input (dedup)
        resolution, preserving the input point order.
        """
        feat = point["feat"]
        cur = point
        while "pooling_parent" in cur.keys():
            inverse = cur["pooling_inverse"]
            feat = feat[inverse]
            cur = cur["pooling_parent"]
        return feat

    @staticmethod
    def _group_by_batch(
        feat: torch.Tensor, batch: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack per-point features ``(P, C)`` into ``(B, P_max, C)``.

        Returns the padded tensor and a bool ``key_padding_mask (B, P_max)``
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

        coord_d, grid_coord_d, batch_d, dedup_inverse = self._voxel_dedup(
            coord, batch)
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
            # Coarse enc features -> PTv3 input (dedup) resolution.
            feat_dedup = self._upsample_to_input(point)
        # Scatter per-voxel features back to every original input point.
        feat_pts = feat_dedup[dedup_inverse]

        tokens, key_padding_mask = self._group_by_batch(feat_pts, batch, b)
        return self.compressor(tokens, key_padding_mask=key_padding_mask)


__all__ = ["UtoniaEncoder"]
