"""Graph packing / reconstruction utilities for the wireframe VAE.

These turn the dataloader's packed-graph batches into the graph wireframe VAE
inputs (a node-level GT: node coords + oriented edge pairs + frozen curve-VAE
latents) and turn decoder predictions back into explicit wireframes.

Edge orientation is fixed by a deterministic **coordinate-lexicographic** rule
(the endpoint with the smaller ``(x, y, z)`` tuple is the canonical "start").
The same rule is used both when encoding the GT curve latent and when
denormalising a decoded curve onto its endpoints, so the curve VAE always sees
a consistent orientation.

Shared by the staged-training models:
  * :class:`~src.models.pc2wireframe.ClrWireframeBase` (stage 2, wireframe VAE)
  * :class:`~src.models.pc2wireframe.PC2WireframeModel` (stage 3, reconstruction)

``normalized_curves_from_batch`` is a config-free helper used by stage 1 (the
curve-VAE-only training) and needs no host state.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

# Per-curve latent dim = curve_vae.latent_channels * downsampled_len
# (default 3 * (32 / 2**3) = 12). Hosts derive ``curve_latent_dim`` from the
# curve VAE config; this is only the expected default.
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

    from .vae.geometry import normalize_curves

    ep = batch.get("edge_points")
    if ep is None or ep.shape[0] == 0:
        return None
    device = ep.device
    arr = ep.detach().cpu().numpy().astype(np.float64)
    norm = normalize_curves(arr).astype(np.float32)
    return torch.from_numpy(norm).to(device)


def coord_orient_swap(ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """Return ``True`` where ``ca`` is lexicographically greater than ``cb``.

    Used to orient each edge so the smaller-coordinate endpoint is first
    (a deterministic, geometry-only rule recoverable at reconstruction time).
    """
    ca = np.asarray(ca)
    cb = np.asarray(cb)
    greater = np.zeros(ca.shape[0], dtype=bool)
    equal = np.ones(ca.shape[0], dtype=bool)
    for k in range(3):
        greater |= equal & (ca[:, k] > cb[:, k])
        equal &= ca[:, k] == cb[:, k]
    return greater


class ClrPackingMixin:
    """Packing / reconstruction methods shared by the wireframe models.

    Requires the host to define: ``curve_vae``, ``wireframe_vae``,
    ``max_nodes``, ``max_curves_num``, ``curve_latent_dim`` and
    ``curve_latent_len``.
    """

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_curve_latent(
        self, norm_curves: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode canonical curves ``(E, U, 3)`` -> ``(mu, std)`` each ``(E, D)``.

        ``D = curve_latent_dim``. The curve VAE is always frozen when this is
        called (stage 2 / 3), hence the ``no_grad``.
        """
        import torch.nn.functional as F
        from einops import rearrange

        x = rearrange(norm_curves, "e u c -> e c u")
        target_len = int(self.curve_vae.sample_points_num)
        if x.shape[-1] != target_len:
            x = F.interpolate(x, size=target_len, mode="linear", align_corners=True)
        posterior = self.curve_vae.encode(x)
        mu = rearrange(posterior.mode(), "e c l -> e (c l)")
        std = rearrange(posterior.std, "e c l -> e (c l)")
        return mu, std

    # ------------------------------------------------------------------
    def graph_to_node_inputs(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Pack a packed-graph batch into node-level VAE inputs.

        Returns a dict with (``N = max_nodes``, ``E = max_curves_num``,
        ``D = curve_latent_dim``):

            ``node_coords`` ``(B, N, 3)``   vertex coordinates (padded)
            ``node_mask``   ``(B, N)``      bool, first ``nv`` valid
            ``edge_pairs``  ``(B, E, 2)``   coord-oriented node-slot indices
            ``edge_mask``   ``(B, E)``      bool, first ``ne`` valid
            ``edge_mu``     ``(B, E, D)``   frozen curve-VAE latent mean
            ``edge_std``    ``(B, E, D)``   frozen curve-VAE latent std
        """
        from .vae.geometry import normalize_curves

        device = batch["point_cloud"].device
        b = int(batch["num_graphs"])
        N = self.max_nodes
        E = self.max_curves_num
        d = self.curve_latent_dim

        vertices = batch["vertices"].detach().cpu().numpy()
        edge_index = batch["edge_index"].detach().cpu().numpy()  # (2, Esum) global
        edge_points = batch["edge_points"].detach().cpu().numpy()  # (Esum, U, 3)
        vptr = batch["vertex_ptr"].tolist()
        eptr = batch["edge_ptr"].tolist()

        node_coords = np.zeros((b, N, 3), dtype=np.float32)
        node_mask = np.zeros((b, N), dtype=bool)
        edge_pairs = np.zeros((b, E, 2), dtype=np.int64)
        edge_mask = np.zeros((b, E), dtype=bool)

        all_norm_curves: list[np.ndarray] = []
        slot_index: list[tuple[int, int]] = []

        for s in range(b):
            v0, v1 = vptr[s], vptr[s + 1]
            e0, e1 = eptr[s], eptr[s + 1]
            nv = min(v1 - v0, N)
            ne = min(e1 - e0, E)
            verts_s = vertices[v0:v1][:nv]
            node_coords[s, :nv] = verts_s
            node_mask[s, :nv] = True
            if ne == 0:
                continue

            eidx_s = (edge_index[:, e0:e1] - v0).T[:ne]  # (ne, 2) local ids
            epts_s = edge_points[e0:e1][:ne]             # (ne, U, 3)
            # drop edges that reference a clipped-away vertex
            keep = (eidx_s[:, 0] < nv) & (eidx_s[:, 1] < nv)
            eidx_s = eidx_s[keep]
            epts_s = epts_s[keep]
            ne = eidx_s.shape[0]
            if ne == 0:
                continue

            ca = verts_s[eidx_s[:, 0]]
            cb = verts_s[eidx_s[:, 1]]
            swap = coord_orient_swap(ca, cb)

            oriented_idx = eidx_s.copy()
            oriented_idx[swap] = oriented_idx[swap][:, ::-1]
            edge_pairs[s, :ne] = oriented_idx
            edge_mask[s, :ne] = True

            epts_oriented = epts_s.copy()
            epts_oriented[swap] = epts_oriented[swap][:, ::-1]
            norm = normalize_curves(epts_oriented.astype(np.float64)).astype(np.float32)
            for k in range(ne):
                slot_index.append((s, k))
            all_norm_curves.append(norm)

        curve_mu = np.zeros((b, E, d), dtype=np.float32)
        curve_std = np.zeros((b, E, d), dtype=np.float32)
        if all_norm_curves:
            stacked = np.concatenate(all_norm_curves, axis=0)  # (sumE, U, 3)
            stacked_t = torch.from_numpy(stacked).to(device=device, dtype=torch.float32)
            mu, std = self.encode_curve_latent(stacked_t)
            mu = mu.detach().cpu().numpy()
            std = std.detach().cpu().numpy()
            for j, (s, k) in enumerate(slot_index):
                curve_mu[s, k] = mu[j]
                curve_std[s, k] = std[j]

        to = lambda a, dt: torch.from_numpy(a).to(device=device, dtype=dt)
        return {
            "node_coords": to(node_coords, torch.float32),
            "node_mask": to(node_mask, torch.bool),
            "edge_pairs": to(edge_pairs, torch.long),
            "edge_mask": to(edge_mask, torch.bool),
            "edge_mu": to(curve_mu, torch.float32),
            "edge_std": to(curve_std, torch.float32),
        }

    # ------------------------------------------------------------------
    def decode_curves(
        self,
        curve_latent: torch.Tensor,
        num_points: int = 32,
        pin_endpoints: bool = True,
    ) -> torch.Tensor:
        """Decode per-curve latents ``(B, N, D)`` -> curves ``(B, N, num_points, 3)``.

        Reshapes to the curve-VAE latent layout, queries uniform ``t in [0, 1]``
        and (optionally) pins the canonical endpoints to ``[-1,0,0]`` /
        ``[1,0,0]``.
        """
        from einops import rearrange

        bsz = curve_latent.shape[0]
        ch = self.curve_vae.config.latent_channels
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
    def reconstruct_graph(
        self,
        dec: dict[str, torch.Tensor],
        *,
        exist_thresh: float = 0.5,
        edge_thresh: float = 0.5,
        recon_curves: bool = True,
        num_points: int = 32,
    ) -> list[dict[str, Any]]:
        """Decoder dict -> explicit wireframes (one dict per sample).

          * ``sigmoid(exist) > exist_thresh``        -> alive vertices;
          * ``sigmoid(adjacency) > edge_thresh`` (upper triangle of the alive
            sub-block) -> edges;
          * per-edge curve latent (from the two endpoint node tokens, ordered
            by the coordinate rule) -> decode + denormalise onto the endpoints.
        """
        from einops import rearrange

        from .vae.recon_utils import denorm_curves

        coord = dec["coord"]
        exist_logit = dec["exist_logit"]
        node_tokens = dec["node_tokens"]
        adj_logits = self.wireframe_vae.predict_adjacency(node_tokens)

        coord_np = coord.detach().cpu().numpy()
        exist_p = torch.sigmoid(exist_logit).detach().cpu().numpy()
        adj_p = torch.sigmoid(adj_logits).detach().cpu().numpy()

        b = coord.shape[0]
        out: list[dict[str, Any]] = []
        for s in range(b):
            alive = np.nonzero(exist_p[s] > exist_thresh)[0]
            if alive.shape[0] == 0:
                out.append({
                    "vertices": np.zeros((0, 3), dtype=np.float32),
                    "edge_index": np.zeros((0, 2), dtype=np.int64),
                    "num_vertices": 0, "num_edges": 0,
                })
                continue

            verts = coord_np[s, alive].astype(np.float32)
            sub = adj_p[s][np.ix_(alive, alive)]
            iu, ju = np.triu_indices(alive.shape[0], k=1)
            sel = sub[iu, ju] > edge_thresh
            li, lj = iu[sel], ju[sel]  # local (alive-space) endpoints

            if li.shape[0] == 0:
                out.append({
                    "vertices": verts,
                    "edge_index": np.zeros((0, 2), dtype=np.int64),
                    "num_vertices": int(alive.shape[0]), "num_edges": 0,
                })
                continue

            # orient each edge by the coordinate rule (smaller coord first)
            ca, cb = verts[li], verts[lj]
            swap = coord_orient_swap(ca, cb)
            a_local = np.where(swap, lj, li)
            b_local = np.where(swap, li, lj)
            edge_index = np.stack([a_local, b_local], axis=-1).astype(np.int64)

            sample_out: dict[str, Any] = {
                "vertices": verts,
                "edge_index": edge_index,
                "num_vertices": int(alive.shape[0]),
                "num_edges": int(edge_index.shape[0]),
            }

            if recon_curves:
                qa = torch.as_tensor(alive[a_local], device=coord.device).long()
                qb = torch.as_tensor(alive[b_local], device=coord.device).long()
                pairs = torch.stack([qa, qb], dim=-1).unsqueeze(0)  # (1, m, 2)
                curve_lat = self.wireframe_vae.predict_curve(
                    node_tokens[s:s + 1], pairs)  # (1, m, D)
                dec_curves = self.decode_curves(
                    curve_lat, num_points=num_points).detach().cpu().numpy()[0]
                corners = np.stack(
                    [verts[a_local], verts[b_local]], axis=1)  # (m, 2, 3)
                curves = denorm_curves(dec_curves, corners)
                if curves is not None:
                    sample_out["edge_points"] = curves
            out.append(sample_out)
        return out


__all__ = [
    "CURVE_LATENT_DIM",
    "ClrPackingMixin",
    "coord_orient_swap",
    "normalized_curves_from_batch",
]
