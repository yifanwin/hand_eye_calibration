from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_transform(value: ArrayLike, *, name: str = "transform") -> NDArray[np.float64]:
    """Return a validated float64 SE(3) matrix.

    Naming convention throughout the package is T_A_B @ p_B = p_A.
    """
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation determinant must be +1")
    return matrix.copy()


def transform_to_list(value: ArrayLike) -> list[list[float]]:
    return as_transform(value).tolist()

