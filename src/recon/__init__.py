"""Wireframe reconstruction from the joint decoder fields.

:func:`assemble_wireframe` turns the joint vertex+edge decoder's predictions into
the GT schema ``{vertices, edge_index, edge_points}`` consumed by
:mod:`src.metrics` and the submission export: it thresholds the predicted
vertices/edges, picks each edge's endpoints from the edge->vertex association
matrix (top-2 per edge) and denormalises the decoded canonical curves onto them
(see :mod:`src.recon.joint_wireframe`).
"""
from .joint_wireframe import assemble_wireframe

__all__ = ["assemble_wireframe"]
