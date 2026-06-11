"""Curve denormalisation helpers (numpy), vendored from CLR-Wire
``sample/utils.py`` minus the matplotlib plotting code.

Used at reconstruction time to map a decoded curve (whose endpoints are pinned
to ``[-1,0,0]`` / ``[1,0,0]``) back onto the predicted edge endpoints. This is
the inverse of ``geometry.normalize_curves`` (two-stage: canonical -> START_END
-> target corner).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Intermediate canonical frame used by ``geometry.normalize_curves``.
START_END = np.array(
    [[0.0, 0.0, 0.0], [0.54020254, -0.77711392, 0.32291667]]
)


def inverse_transform_polyline(
    transformed_points: np.ndarray,
    start_and_end: np.ndarray,
    handleCollinear: bool = True,
    epsilon: float = 1e-6,
) -> Optional[np.ndarray]:
    """Map ``transformed_points`` so its endpoints land on ``start_and_end``."""
    tgt_start, tgt_end = start_and_end
    offset = -transformed_points[0]

    lengths = np.linalg.norm(transformed_points[-1] - transformed_points[0])

    transformed_points = transformed_points + offset

    tgt_direction = tgt_end - tgt_start
    scale_factor = np.linalg.norm(tgt_direction)
    if scale_factor == 0:
        raise ValueError("start == end; scale factor undefined.")

    scaled_back_points = transformed_points * scale_factor / (lengths + epsilon)

    target_vector = tgt_direction
    pn_prime = scaled_back_points[-1]
    if np.linalg.norm(pn_prime) == 0:
        raise ValueError("endpoint at origin; direction undefined.")

    pn_prime_norm = pn_prime / (np.linalg.norm(pn_prime) + epsilon)
    target_norm = target_vector / (np.linalg.norm(target_vector) + epsilon)

    dot_product = np.dot(pn_prime_norm, target_norm)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    if np.abs(dot_product + 1) < epsilon:
        if not handleCollinear:
            return None
        arbitrary_vector = np.array([1, 0, 0])
        if np.allclose(pn_prime_norm, arbitrary_vector) or np.allclose(
            pn_prime_norm, -arbitrary_vector
        ):
            arbitrary_vector = np.array([0, 1, 0])
        axis = np.cross(pn_prime_norm, arbitrary_vector)
        axis = axis / (np.linalg.norm(axis) + epsilon)
        angle = np.pi
    elif np.abs(dot_product - 1) < epsilon:
        if not handleCollinear:
            return None
        axis = np.array([0, 0, 1])
        angle = 0.0
    else:
        axis = np.cross(pn_prime_norm, target_norm)
        axis = axis / (np.linalg.norm(axis) + epsilon)

    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    rotated_back_points = np.dot(scaled_back_points, R.T)
    restored_points = rotated_back_points + tgt_start
    return restored_points


def denorm_curves(
    norm_curves: np.ndarray, corners: np.ndarray
) -> Optional[np.ndarray]:
    """Denormalise canonical curves onto their predicted ``corners`` (E,2,3)."""
    curves = []
    for i, corner in enumerate(corners):
        if np.linalg.norm(corner[0] - corner[1]) == 0:
            continue
        curve_i_temp = inverse_transform_polyline(
            norm_curves[i], start_and_end=START_END
        )
        curve_i = inverse_transform_polyline(curve_i_temp, start_and_end=corner)
        if curve_i is None:
            continue
        curves.append(curve_i)
    if curves:
        return np.stack(curves, axis=0)
    return None


__all__ = ["START_END", "inverse_transform_polyline", "denorm_curves"]
