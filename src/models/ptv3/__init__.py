"""Vendored Point Transformer V3 (from Manual-Assembly / Pointcept).

Kept verbatim except for the package location. Requires ``spconv``,
``torch_scatter``, ``timm``, ``addict`` and (optionally) ``flash_attn``.
The import is done lazily so that the rest of the package can be imported
on machines where these heavy deps are not installed yet.
"""
from __future__ import annotations

__all__ = ["PointTransformerV3", "Point"]


def __getattr__(name: str):
    # Lazy re-export so importing :mod:`src.models` does not hard-require the
    # PTv3 dependency stack (spconv / flash_attn) until PTv3 is actually used.
    if name in __all__:
        from . import ptv3 as _ptv3

        return getattr(_ptv3, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
