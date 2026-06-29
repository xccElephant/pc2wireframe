"""Wireframe reconstruction from the edge-set decoder predictions.

:func:`assemble_wireframe` turns the edge-set decoder's predictions (per-edge
existence + endpoints + decoded canonical curve) into the GT schema
``{vertices, edge_index, edge_points}`` consumed by :mod:`src.metrics` and the
submission export: it thresholds the edges, merges the free endpoints into a
shared vertex set (confidence-weighted), suppresses near-duplicate edges with an
E-NMS and denormalises the decoded canonical curves onto the merged endpoints
(see :mod:`src.recon.edge_wireframe`).
"""
from .edge_wireframe import assemble_wireframe

__all__ = ["assemble_wireframe"]
