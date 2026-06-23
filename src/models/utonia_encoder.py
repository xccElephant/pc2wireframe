"""Multi-scale point-cloud encoder: frozen Utonia PTv3 + per-scale compressors.

Wraps the pre-trained, *frozen* `Utonia <https://huggingface.co/Pointcept/Utonia>`_
PTv3 encoder (an encoder-only Point Transformer V3, vendored under
``src/models/utonia``) and pools several of its **encoder stages** into a
*multi-scale* set of latent token sets -- one per scale, finer -> coarser.

Pipeline::

    packed point cloud (coord (P_sum, 3), offset (B,))
        -> per-sample voxel dedup (one point per (batch, grid_coord) voxel)
        -> Utonia PTv3 encoder        # frozen, no_grad, eval (deterministic)
        -> per-stage per-voxel features (NO upsampling): walk the GridPooling
           ``pooling_parent`` chain to recover each selected encoder stage's
           native-resolution voxel tokens (M_s, C_s)
        -> group by sample + pad      # (B, M_s_max, C_s) + key_padding_mask
        -> per-scale LatentCompressor (N_s queries, in_dim = C_s) -> z_s (B, N_s, D)
        -> list[ z_s ]                # consumed by the per-scale RVQ quantizer

Unlike the single-scale variant this does **not** broadcast the coarsest
features back to the input resolution; each scale is pooled at its own encoder
resolution, giving the quantizer a coarse-to-fine token pyramid (coarse tokens
carry the global structure large/sparse wireframes need).

The backbone weights (config + state_dict) come from the Utonia checkpoint, so
per-stage channel widths (``enc_channels``) are read from the ckpt config rather
than hard-coded. The heavy ``utonia`` import is deferred to construction so this
module can be imported without the spconv / flash-attn stack present.
"""
from __future__ import annotations

import contextlib

import torch
from torch import nn

from .latent_compressor import LatentCompressor


class UtoniaEncoder(nn.Module):
    """Frozen Utonia PTv3 + one trainable :class:`LatentCompressor` per scale.

    Args:
        utonia: Utonia checkpoint identifier -- a HuggingFace model name
            (resolved against ``utonia_repo_id``) or a local ``.pth`` path.
        utonia_repo_id: HF repo the named checkpoint is downloaded from.
        download_root: optional cache dir for the downloaded checkpoint.
        grid_size: voxel size used for the per-sample dedup and PTv3 grid.
        scale_tokens: ``N_s`` latent tokens per scale (finer -> coarser),
            e.g. ``[256, 128, 64]``.
        scale_stages: which PTv3 encoder stage indices to pool (ascending =
            finer first). ``None`` -> the deepest ``len(scale_tokens)`` stages.
        latent_dim: per-token latent channels ``D`` (shared across scales).
        compressor_heads: attention heads in each pooling head.
        compressor_layers: ``nn.TransformerDecoder`` layers per compressor.
        freeze: freeze the PTv3 backbone (``requires_grad=False`` + kept in
            ``eval``, forward in ``no_grad``). The compressors are trainable.
    """

    def __init__(
        self,
        utonia: str = "utonia",
        utonia_repo_id: str = "Pointcept/Utonia",
        download_root: str | None = None,
        grid_size: float = 0.01,
        scale_tokens: list[int] | None = None,
        scale_stages: list[int] | None = None,
        latent_dim: int = 256,
        compressor_heads: int = 8,
        compressor_layers: int = 1,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        from .utonia.model import PointTransformerV3, load  # deferred heavy import

        ckpt = load(
            utonia, repo_id=utonia_repo_id, download_root=download_root,
            ckpt_only=True,
        )
        cfg = dict(ckpt["config"])
        self.in_channels = int(cfg["in_channels"])
        enc_channels = list(cfg["enc_channels"])
        self.num_stages = len(enc_channels)
        self.grid_size = float(grid_size)
        self.freeze = bool(freeze)
        self.latent_dim = int(latent_dim)

        scale_tokens = list(scale_tokens) if scale_tokens else [256, 128, 64]
        if scale_stages is None:
            # the deepest len(scale_tokens) stages (coarsest carries structure).
            n = len(scale_tokens)
            scale_stages = list(range(self.num_stages - n, self.num_stages))
        scale_stages = [int(s) for s in scale_stages]
        if len(scale_stages) != len(scale_tokens):
            raise ValueError(
                f"scale_stages ({scale_stages}) and scale_tokens "
                f"({scale_tokens}) must have equal length")
        for s in scale_stages:
            if not 0 <= s < self.num_stages:
                raise ValueError(
                    f"scale stage {s} out of range [0, {self.num_stages})")
        self.scale_stages = scale_stages
        self.scale_tokens = [int(n) for n in scale_tokens]
        # per-scale input channel width = enc_channels[stage].
        self.scale_in_dims = [enc_channels[s] for s in scale_stages]

        self.backbone = PointTransformerV3(**cfg)
        self.backbone.load_state_dict(ckpt["state_dict"])
        if self.freeze:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        # One compressor per scale (in_dim = that stage's channel count).
        # The latent here is a continuous *intermediate* (the discrete budget is
        # enforced downstream by the quantizer), so the LatentCompressor float
        # budget is effectively disabled.
        self.compressors = nn.ModuleList([
            LatentCompressor(
                in_dim=in_dim,
                num_tokens=n_tok,
                latent_dim=self.latent_dim,
                nhead=compressor_heads,
                num_layers=compressor_layers,
                latent_budget_max=n_tok * self.latent_dim,
            )
            for in_dim, n_tok in zip(self.scale_in_dims, self.scale_tokens)
        ])

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One point per ``(batch, grid_coord)`` voxel (Utonia GridSample style).

        Returns the kept ``coord``, integer ``grid_coord`` and kept ``batch`` id
        (rows sorted ascending by unique voxel id).
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
        _, inverse, _ = torch.unique(
            keys, dim=0, return_inverse=True, return_counts=True)
        order = torch.argsort(inverse, stable=True)
        inv_sorted = inverse[order]
        first = torch.ones_like(inv_sorted, dtype=torch.bool)
        first[1:] = inv_sorted[1:] != inv_sorted[:-1]
        keep = order[first]
        return coord[keep], grid_coord[keep], batch[keep]

    def _collect_stage_points(self, point) -> list:
        """Recover the per-stage :class:`Point` from coarsest -> finest.

        Utonia (``enc_mode=True``) returns the coarsest stage's point; each
        ``GridPooling`` recorded ``pooling_parent`` (the pre-pooling point =
        the previous stage's block output). Walking that chain yields every
        encoder stage's native-resolution voxel features, no upsampling.
        ``chain[0]`` is the coarsest stage (``stage = num_stages - 1``);
        ``chain[d]`` is ``stage = num_stages - 1 - d``.
        """
        chain = []
        cur = point
        while True:
            chain.append(cur)
            parent = cur.get("pooling_parent", None)
            if parent is None:
                break
            cur = parent
        return chain

    @staticmethod
    def _group_by_batch(
        feat: torch.Tensor, batch: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack per-voxel features ``(M, C)`` into ``(B, M_max, C)``.

        Returns the padded tensor and a bool ``key_padding_mask (B, M_max)``
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
    ) -> list[torch.Tensor]:
        """Encode a packed point cloud to a multi-scale latent token list.

        Args:
            coord: ``(P_sum, 3)`` concatenated batch points.
            offset: ``(B,)`` cumulative per-sample point counts.

        Returns:
            ``list`` of ``z_s (B, N_s, latent_dim)`` (finer -> coarser, aligned
            with ``self.scale_stages`` / ``self.scale_tokens``).
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
            chain = self._collect_stage_points(point)
        depth_max = len(chain) - 1

        z_list: list[torch.Tensor] = []
        for stage, compressor in zip(self.scale_stages, self.compressors):
            depth = depth_max - stage
            stage_point = chain[depth]
            feat_s = stage_point.feat
            batch_s = stage_point.batch.long()
            tokens, kpm = self._group_by_batch(feat_s, batch_s, b)
            z_list.append(compressor(tokens, key_padding_mask=kpm))
        return z_list


__all__ = ["UtoniaEncoder"]
