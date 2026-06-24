"""Wireframe reconstruction from the edge-set decoder fields.

Turns the decoder's per-edge existence + ordered curve points into an explicit
wireframe ``{vertices, edge_index, edge_points}`` by union-find endpoint
aggregation. See :mod:`src.recon.wireframe`.
"""
from .wireframe import aggregate_wireframe

__all__ = ["aggregate_wireframe"]
