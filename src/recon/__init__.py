"""Wireframe reconstruction from the stage-2 grouper's per-point fields.

Turns the learned per-point fields (endpoint offsets / embedding / curve type /
anchors / arclen) into an explicit wireframe
``{vertices, edge_index, edge_points}``. See :mod:`src.recon.grouped`.
"""
from .grouped import group_wireframe

__all__ = ["group_wireframe"]
