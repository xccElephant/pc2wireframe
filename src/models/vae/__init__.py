"""VAE stack: per-curve VAE (``vae_curve``) + wireframe VAE (``vae_wireframe``).

The **curve VAE** (``AutoencoderKL1D``) is a custom pure-PyTorch attention/token
VAE (see ``vae_curve.py``) with no diffusers / x_transformers dependency.

The **wireframe VAE** (``AutoencoderKLWireframe``) is still derived from CLR-Wire
(https://github.com/qixuema/CLR-Wire, SIGGRAPH 2025) and depends on
``diffusers`` / ``x_transformers``; only the VAE pieces are kept (the
flow-matching / diffusion prior is replaced by the point-cloud encoder).

Imports are lazy so that pulling in the (light) curve VAE does not drag in the
wireframe VAE's heavy dependencies.
"""
from __future__ import annotations

__all__ = [
    "AutoencoderKLWireframe",
    "AutoencoderKLWireframeFastEncode",
    "AutoencoderKLWireframeFastDecode",
    "AutoencoderKL1D",
    "AutoencoderKL1DFastEncode",
    "AutoencoderKL1DFastDecode",
]

_WIREFRAME = {
    "AutoencoderKLWireframe",
    "AutoencoderKLWireframeFastEncode",
    "AutoencoderKLWireframeFastDecode",
}
_CURVE = {
    "AutoencoderKL1D",
    "AutoencoderKL1DFastEncode",
    "AutoencoderKL1DFastDecode",
}


def __getattr__(name: str):
    if name in _WIREFRAME:
        from . import vae_wireframe as _m

        return getattr(_m, name)
    if name in _CURVE:
        from . import vae_curve as _m

        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
