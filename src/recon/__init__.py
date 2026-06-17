"""Traditional (non-learned, deterministic) wireframe reconstruction.

Turns a generated point set ``(N, 4) = (x, y, z, type)`` into an explicit
wireframe ``{vertices, edge_index, edge_points}`` via vertex clustering plus a
nearest-two-vertices voting scheme. See :mod:`src.recon.traditional`.
"""
from .grouped import group_wireframe
from .traditional import reconstruct_wireframe

__all__ = ["reconstruct_wireframe", "group_wireframe"]
