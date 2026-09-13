from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray


def load_trajectory(path: str | Path, *, joint_count: int | None = None) -> NDArray[np.float64]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("trajectory must have schema_version: 1")
    if data.get("units") != "rad":
        raise ValueError("trajectory units must be rad")
    waypoints = np.asarray(data.get("waypoints", []), dtype=np.float64)
    if waypoints.ndim != 2 or len(waypoints) == 0 or not np.all(np.isfinite(waypoints)):
        raise ValueError("trajectory waypoints must be a non-empty finite 2D array")
    if joint_count is not None and waypoints.shape[1] != joint_count:
        raise ValueError(f"expected {joint_count} joints, got {waypoints.shape[1]}")
    return waypoints


def save_trajectory(path: str | Path, waypoints: NDArray[np.float64], *, robot: str) -> None:
    points = np.asarray(waypoints, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0 or not np.all(np.isfinite(points)):
        raise ValueError("trajectory waypoints must be a non-empty finite 2D array")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "units": "rad",
        "robot": robot,
        "joint_count": int(points.shape[1]),
        "waypoints": points.tolist(),
    }
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

