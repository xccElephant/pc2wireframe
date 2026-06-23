"""Wireframe reconstruction from the WireframeAE decoder fields.

Turns the decoder's per-query vertex fields + pairwise edge predictions into an
explicit wireframe ``{vertices, edge_index, edge_points}``. See
:mod:`src.recon.wireframe`.
"""
from .wireframe import decode_wireframe

__all__ = ["decode_wireframe"]
