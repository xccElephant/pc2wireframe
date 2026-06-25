"""Per-scale residual vector quantizer for the multi-scale latent.

The VQVAE branch replaces the continuous ``16x256`` latent with a **discrete**
submission: a flat list of codebook indices. The encoder emits one continuous
token set per scale (finer -> coarser); :class:`MultiScaleResidualVQ` quantizes
each scale with its **own** :class:`~vector_quantize_pytorch.ResidualVQ`
(``n_q`` residual levels each), so the submission is

    indices = concat_s [ idx_s.reshape(B, N_s * n_q) ]        # (B, T)

with a *fixed* layout (scale-major, then token-major, then quantizer-level).
The total index count ``T = sum_s N_s * n_q`` is the competition budget and is
checked ``<= budget_max`` (4096) **at construction** -- correctness by
construction, never relying on a training run.

Round-trip
----------
``forward`` returns the quantized tokens (straight-through), the flat indices
and the commitment loss; :meth:`decode_indices` rebuilds the exact same
per-scale ``z_q`` from a flat index tensor alone (via each RVQ's
``get_output_from_indices``), so the wireframe is fully reproducible from the
submitted indices -> codebooks -> decoder.

EMA codebook updates run only while the module is in ``train`` mode (the
``vector-quantize-pytorch`` codebooks gate their EMA / dead-code revival on
``self.training``); calling ``.eval()`` therefore freezes the codebooks for
deterministic export.

This module only *imports* ``vector-quantize-pytorch``; installing it is the
user's responsibility (declared in ``requirements.txt``).
"""
from __future__ import annotations

import torch
from torch import nn


class MultiScaleResidualVQ(nn.Module):
    """One independent ``ResidualVQ`` per scale, with a fixed flat index layout.

    Args:
        scale_tokens: ``N_s`` -- number of tokens at each scale (finer -> coarser,
            e.g. ``[256, 128, 64]``). Defines the index layout and budget.
        dim: per-token channel width ``D`` (= encoder ``latent_dim``).
        n_q: residual quantizer levels per scale (``num_quantizers``).
        codebook_size: codebook entries per level -- a single ``int`` shared by
            every scale, or a per-scale ``list`` (finer -> coarser). The budget
            is the index *count*, not bits, so size is otherwise free; shrinking
            the coarse scales (few tokens) to match their real utilisation is a
            codebook-collapse remedy.
        kmeans_init: k-means codebook init on the first training batch.
        threshold_ema_dead_code: dead-code revival threshold (EMA count).
        decay: codebook EMA decay.
        commitment_weight: weight of the per-RVQ commitment loss (the module
            returns the raw commit loss; the LightningModule applies its own
            ``w_commit`` ramp on top).
        budget_max: hard cap on the flat index count ``sum_s N_s * n_q``.
    """

    BUDGET_MAX_DEFAULT = 4096

    def __init__(
        self,
        scale_tokens: list[int],
        dim: int = 256,
        n_q: int = 8,
        codebook_size: int | list[int] = 8192,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 2,
        decay: float = 0.99,
        commitment_weight: float = 0.25,
        budget_max: int | None = None,
    ) -> None:
        super().__init__()
        from vector_quantize_pytorch import ResidualVQ  # user-installed

        self.scale_tokens = [int(n) for n in scale_tokens]
        self.n_q = int(n_q)
        self.dim = int(dim)
        # codebook_size: a shared int or one entry per scale (finer -> coarser).
        if isinstance(codebook_size, (list, tuple)):
            self.codebook_sizes = [int(c) for c in codebook_size]
            if len(self.codebook_sizes) != len(self.scale_tokens):
                raise ValueError(
                    f"codebook_size list {self.codebook_sizes} must have one "
                    f"entry per scale ({len(self.scale_tokens)})")
        else:
            self.codebook_sizes = [int(codebook_size)] * len(self.scale_tokens)
        # Max size (for perplexity bincount minlength; per-scale RVQs differ).
        self.codebook_size = int(max(self.codebook_sizes))
        # per-scale flat sizes: N_s * n_q (used for split + reshape round-trip).
        self.scale_sizes = [n * self.n_q for n in self.scale_tokens]
        self.total_indices = int(sum(self.scale_sizes))

        cap = self.BUDGET_MAX_DEFAULT if budget_max is None else int(budget_max)
        if self.total_indices > cap:
            raise ValueError(
                f"flat index budget sum_s N_s*n_q = {self.scale_sizes} = "
                f"{self.total_indices} > {cap} (competition cap is "
                f"{self.BUDGET_MAX_DEFAULT} float32 values)"
            )

        self.rvqs = nn.ModuleList([
            ResidualVQ(
                dim=self.dim,
                num_quantizers=self.n_q,
                codebook_size=cb,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                threshold_ema_dead_code=threshold_ema_dead_code,
                decay=decay,
                commitment_weight=commitment_weight,
            )
            for cb in self.codebook_sizes
        ])

    # ------------------------------------------------------------------
    @property
    def budget(self) -> int:
        return self.total_indices

    def forward(
        self, z_list: list[torch.Tensor]
    ) -> dict[str, object]:
        """Quantize each scale.

        Args:
            z_list: per-scale continuous tokens, each ``(B, N_s, D)``.

        Returns:
            dict with::

                z_q       list[(B, N_s, D)]   straight-through quantized tokens
                indices   (B, T) long         flat submission indices (fixed layout)
                idx_list  list[(B, N_s, n_q)] per-scale indices (for perplexity)
                commit    () scalar           summed per-scale commitment loss
        """
        if len(z_list) != len(self.rvqs):
            raise ValueError(
                f"expected {len(self.rvqs)} scales, got {len(z_list)}")
        z_q_list: list[torch.Tensor] = []
        idx_list: list[torch.Tensor] = []
        flat_parts: list[torch.Tensor] = []
        commit = z_list[0].new_zeros(())
        for s, (rvq, z) in enumerate(zip(self.rvqs, z_list)):
            if z.shape[1] != self.scale_tokens[s]:
                raise ValueError(
                    f"scale {s}: expected N_s={self.scale_tokens[s]} tokens, "
                    f"got {z.shape[1]}")
            z_q, idx, cl = rvq(z)                  # (B,N,D), (B,N,n_q), (B,n_q)
            z_q_list.append(z_q)
            idx_list.append(idx)
            flat_parts.append(idx.reshape(idx.shape[0], -1))  # (B, N_s*n_q)
            commit = commit + cl.float().mean()
        indices = torch.cat(flat_parts, dim=1)                # (B, T)
        return {
            "z_q": z_q_list,
            "indices": indices,
            "idx_list": idx_list,
            "commit": commit,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> list[torch.Tensor]:
        """Rebuild per-scale ``z_q`` from flat submission indices.

        Inverse of the layout in :meth:`forward`: split ``(B, T)`` (or ``(T,)``)
        into per-scale chunks of ``N_s * n_q``, reshape to ``(B, N_s, n_q)`` and
        run each RVQ's ``get_output_from_indices``. Reproduces the exact ``z_q``
        used at encode time (guarantees the indices -> wireframe round-trip).
        """
        idx = indices
        if idx.dim() == 1:
            idx = idx[None, :]
        idx = idx.long()
        if idx.shape[1] != self.total_indices:
            raise ValueError(
                f"flat indices width {idx.shape[1]} != expected "
                f"{self.total_indices}")
        z_q_list: list[torch.Tensor] = []
        off = 0
        for s, rvq in enumerate(self.rvqs):
            size = self.scale_sizes[s]
            chunk = idx[:, off:off + size].reshape(
                idx.shape[0], self.scale_tokens[s], self.n_q)
            z_q_list.append(rvq.get_output_from_indices(chunk))
            off += size
        return z_q_list

    # ------------------------------------------------------------------
    @staticmethod
    @torch.no_grad()
    def perplexity(idx_list: list[torch.Tensor], codebook_size: int
                   ) -> list[torch.Tensor]:
        """Per-scale mean codebook perplexity (averaged over the n_q levels).

        ``perplexity = exp(entropy)`` of the (single-batch) code histogram; a
        proxy for codebook utilisation -- a collapse shows up as a perplexity
        far below ``codebook_size``.
        """
        out: list[torch.Tensor] = []
        for idx in idx_list:
            flat = idx.reshape(-1, idx.shape[-1])            # (B*N, n_q)
            per_level = []
            for q in range(flat.shape[-1]):
                codes = flat[:, q]
                counts = torch.bincount(codes, minlength=codebook_size).float()
                p = counts / counts.sum().clamp_min(1.0)
                nz = p > 0
                ent = -(p[nz] * p[nz].log()).sum()
                per_level.append(torch.exp(ent))
            out.append(torch.stack(per_level).mean())
        return out


__all__ = ["MultiScaleResidualVQ"]
