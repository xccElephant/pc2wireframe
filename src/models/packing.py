"""Shared CLR-Wire packing / reconstruction utilities for staged training.

These turn the dataloader's packed-graph batches into the CLR-Wire wireframe
VAE inputs ``(xs, flag_diffs)`` and turn decoder predictions back into explicit
wireframes. They are shared by the staged-training models:

  * :class:`~src.models.pc2wireframe.ClrWireframeBase` (stage 2, wireframe VAE)
  * :class:`~src.models.pc2wireframe.PC2WireframeModel` (stage 3, reconstruction)

The methods live in a mixin so both models share one implementation; the host
must provide ``curve_vae`` plus the cached ``max_*`` / ``curve_latent_*``
config attributes (see ``ClrWireframeBase._init_clr_config``).

``normalized_curves_from_batch`` is a config-free helper used by stage 1 (the
curve-VAE-only training) and needs no host state.
"""
from __future__ import annotations

from typing import Any

import torch

# The vendored CLR-Wire wireframe VAE hardcodes a 12-d per-curve latent
# (``predict_curve_latent`` out_dim=12; loss slices ``[:12]`` / ``[12:]``).
CURVE_LATENT_DIM = 12


def normalized_curves_from_batch(batch: dict[str, Any]) -> torch.Tensor | None:
    """Per-curve canonical curves ``(Esum, U, 3)`` (endpoints -> [-1,0,0]/[1,0,0]).

    The collate fn precomputes ``edge_points_norm`` in the dataloader workers,
    so the common path is a zero-copy read of that tensor (already on device).
    Falls back to normalising ``edge_points`` on the fly for batches that lack
    the precomputed key. Returns ``None`` if the batch contains no edges.
    """
    ep_norm = batch.get("edge_points_norm")
    if ep_norm is not None:
        return ep_norm if ep_norm.shape[0] > 0 else None

    import numpy as np

    from .vae.geometry import normalize_curves

    ep = batch.get("edge_points")
    if ep is None or ep.shape[0] == 0:
        return None
    device = ep.device
    arr = ep.detach().cpu().numpy().astype(np.float64)
    norm = normalize_curves(arr).astype(np.float32)
    return torch.from_numpy(norm).to(device)


class ClrPackingMixin:
    """Packing / reconstruction methods shared by the CLR-Wire models.

    Requires the host to define: ``curve_vae``, ``max_curves_num``,
    ``max_col_diff``, ``max_row_diff``, ``curve_latent_dim`` and
    ``curve_latent_len``.
    """

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_curve_latent(
        self, norm_curves: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode canonical curves ``(E, U, 3)`` -> ``(mu, std)`` each ``(E, D)``.

        ``D = curve_latent_dim`` (default 12 = 3 channels x 4 positions). The
        curve VAE is always frozen when this is called (stage 2 / 3), hence the
        ``no_grad``.
        """
        import torch.nn.functional as F
        from einops import rearrange

        x = rearrange(norm_curves, "e u c -> e c u")
        target_len = int(self.curve_vae.sample_points_num)
        if x.shape[-1] != target_len:
            x = F.interpolate(x, size=target_len, mode="linear", align_corners=True)
        posterior = self.curve_vae.encode(x)
        # Flatten channel-major ("e c l -> e (c l)"); decode_curves inverts this.
        mu = rearrange(posterior.mode(), "e c l -> e (c l)")
        std = rearrange(posterior.std, "e c l -> e (c l)")
        return mu, std

    # ------------------------------------------------------------------
    def graph_to_clr_inputs(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Pack a packed-graph batch into CLR-Wire ``(xs, flag_diffs)``.

        Mirrors CLR-Wire ``WireframeDataset.__getitem__`` + ``compute_diffs``,
        but starts from this project's packed-graph collate (native-size graphs
        concatenated with CSR pointers) instead of pre-canonicalised ``.npz``.

        Returns a dict with:
            ``xs``         ``(B, max_curves, 6 + 2*curve_latent_dim)``
                           = [endpoints(6) | curve mu(D) | curve std(D)]
            ``flag_diffs`` ``(B, max_curves, 3)`` = [valid | col_diff | row_diff]
        """
        import numpy as np

        from .vae.geometry import normalize_curves
        from . import wireframe_ops as wops

        device = batch["point_cloud"].device
        b = int(batch["num_graphs"])
        max_c = self.max_curves_num
        d = self.curve_latent_dim

        vertices = batch["vertices"].detach().cpu().numpy()
        edge_index = batch["edge_index"].detach().cpu().numpy()  # (2, Esum) global
        edge_points = batch["edge_points"].detach().cpu().numpy()  # (Esum, U, 3)
        vptr = batch["vertex_ptr"].tolist()
        eptr = batch["edge_ptr"].tolist()

        segments = np.zeros((b, max_c, 6), dtype=np.float32)
        flag_diffs = np.zeros((b, max_c, 3), dtype=np.int64)
        # Collect oriented+normalized curves across the whole batch for one
        # curve-VAE encode pass, with bookkeeping back to (sample, slot).
        all_norm_curves: list[np.ndarray] = []
        slot_index: list[tuple[int, int]] = []

        for s in range(b):
            v0, v1 = vptr[s], vptr[s + 1]
            e0, e1 = eptr[s], eptr[s + 1]
            nv = v1 - v0
            ne = min(e1 - e0, max_c)
            verts_s = vertices[v0:v1]
            # local edge ids in [0, nv)
            eidx_s = (edge_index[:, e0:e1] - v0).T  # (E, 2)
            epts_s = edge_points[e0:e1]             # (E, U, 3)
            if ne == 0:
                continue
            eidx_s = eidx_s[:ne]
            epts_s = epts_s[:ne]

            can = wops.canonicalize(
                nv, eidx_s,
                max_col_diff=self.max_col_diff,
                max_row_diff=self.max_row_diff,
            )
            order = can["order"]
            adj = can["adj"]            # (ne, 2) new ids, oriented+sorted
            diffs = can["diffs"]        # (ne, 2)

            relabelled = can["perm"][eidx_s]
            swap = relabelled[:, 0] > relabelled[:, 1]

            # oriented endpoint coords (first=min-id vertex), then edge-sorted
            coords_pair = verts_s[eidx_s]            # (ne, 2, 3) old order
            coords_pair[swap] = coords_pair[swap][:, ::-1]
            seg = coords_pair.reshape(ne, 6)[order]
            segments[s, :ne] = seg

            flag_diffs[s, :ne, 0] = 1
            flag_diffs[s, :ne, 1:] = diffs

            # oriented + reordered curve polylines, normalized to canonical frame
            epts_oriented = epts_s.copy()
            epts_oriented[swap] = epts_oriented[swap][:, ::-1]
            epts_oriented = epts_oriented[order]
            norm = normalize_curves(epts_oriented.astype(np.float64)).astype(np.float32)
            for k in range(ne):
                slot_index.append((s, k))
            all_norm_curves.append(norm)

        # one curve-VAE encode pass for the whole batch
        curve_mu = np.zeros((b, max_c, d), dtype=np.float32)
        curve_std = np.zeros((b, max_c, d), dtype=np.float32)
        if all_norm_curves:
            stacked = np.concatenate(all_norm_curves, axis=0)  # (sumE, U, 3)
            stacked_t = torch.from_numpy(stacked).to(device=device, dtype=torch.float32)
            mu, std = self.encode_curve_latent(stacked_t)
            mu = mu.detach().cpu().numpy()
            std = std.detach().cpu().numpy()
            for j, (s, k) in enumerate(slot_index):
                curve_mu[s, k] = mu[j]
                curve_std[s, k] = std[j]

        xs = np.concatenate([segments, curve_mu, curve_std], axis=-1)  # (B, C, 6+2D)
        xs_t = torch.from_numpy(xs).to(device=device, dtype=torch.float32)
        flag_diffs_t = torch.from_numpy(flag_diffs).to(device=device, dtype=torch.long)
        return {"xs": xs_t, "flag_diffs": flag_diffs_t}

    # ------------------------------------------------------------------
    def decode_curves(
        self,
        curve_latent: torch.Tensor,
        num_points: int = 32,
        pin_endpoints: bool = True,
    ) -> torch.Tensor:
        """Decode per-curve latents ``(B, N, D)`` -> curves ``(B, N, num_points, 3)``.

        Mirrors CLR-Wire ``sample.py``: reshape to the curve-VAE latent layout,
        query uniform ``t in [0, 1]`` and (optionally) pin the canonical
        endpoints to ``[-1,0,0]`` / ``[1,0,0]``.
        """
        from einops import rearrange

        bsz = curve_latent.shape[0]
        ch = self.curve_vae.config.latent_channels
        # Inverse of encode_curve_latent's ``reshape(E, ch*L)`` (channel-major),
        # i.e. flat layout is "(c l)". Keep encode/decode self-consistent.
        z = rearrange(curve_latent, "b n (c l) -> (b n) c l", c=ch)  # (B*N, ch, L)
        t = torch.linspace(0.0, 1.0, num_points, device=curve_latent.device)
        t = t.unsqueeze(0).expand(z.shape[0], -1)
        dec = self.curve_vae.decode(z, t)  # (B*N, 3, num_points)
        if pin_endpoints:
            dec[:, :, 0] = torch.tensor([-1.0, 0.0, 0.0], device=dec.device)
            dec[:, :, -1] = torch.tensor([1.0, 0.0, 0.0], device=dec.device)
        dec = rearrange(dec, "(b n) c d -> b n d c", b=bsz)  # (B, N, num_points, 3)
        return dec

    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct(
        self,
        preds: dict[str, torch.Tensor],
        recon_curves: bool = True,
        num_points: int = 32,
    ) -> list[dict[str, Any]]:
        """Decoder predictions -> explicit wireframes (one dict per sample).

        Mirrors CLR-Wire ``sample/reconstruction.py``:
          * ``argmax(cls) + 1`` -> number of curves;
          * ``argmax`` col/row diffs (+1 for row) -> cumsum -> adjacency;
          * average shared-vertex endpoints (``refine_segment_coords_by_adj``);
          * optionally decode + denormalise curves onto the endpoints.
        """
        import numpy as np
        from einops import rearrange

        from . import wireframe_ops as wops
        from .vae.recon_utils import denorm_curves

        cls = preds["cls"].detach().cpu().numpy()
        num_curves = cls.argmax(axis=-1) + 1
        segments = preds["segments"].detach().cpu().numpy()
        diffs = preds["diffs"].detach().cpu().numpy()
        col = diffs[..., : self.max_col_diff].argmax(axis=-1)
        row = diffs[..., self.max_col_diff :].argmax(axis=-1) + 1
        adj_all = wops.diffs_to_adjacency(col, row)  # (B, C, 2)

        dec_curves = None
        if recon_curves:
            dec_curves = self.decode_curves(
                preds["curve_latent"], num_points=num_points
            ).detach().cpu().numpy()  # (B, C, num_points, 3)

        out: list[dict[str, Any]] = []
        b = cls.shape[0]
        for s in range(b):
            nc = int(num_curves[s])
            adj_s = adj_all[s, :nc]
            seg_s = wops.refine_segment_coords_by_adj(adj_s, segments[s, :nc])

            node_ids = np.unique(adj_s)
            remap = {int(o): i for i, o in enumerate(node_ids)}
            verts = np.zeros((len(node_ids), 3), dtype=np.float32)
            for i, (a, bb) in enumerate(adj_s):
                verts[remap[int(a)]] = seg_s[i, :3]
                verts[remap[int(bb)]] = seg_s[i, 3:]
            edge_index = np.array(
                [[remap[int(a)], remap[int(bb)]] for a, bb in adj_s],
                dtype=np.int64,
            ).reshape(-1, 2)

            sample_out: dict[str, Any] = {
                "vertices": verts,
                "edge_index": edge_index,
                "edge_endpoints": seg_s.astype(np.float32),
                "num_vertices": len(node_ids),
                "num_edges": nc,
            }
            if dec_curves is not None and nc > 0:
                corners = rearrange(seg_s, "n (c d) -> n c d", c=2)
                curves = denorm_curves(dec_curves[s, :nc], corners)
                sample_out["edge_points"] = curves
            out.append(sample_out)
        return out


__all__ = ["CURVE_LATENT_DIM", "ClrPackingMixin", "normalized_curves_from_batch"]
