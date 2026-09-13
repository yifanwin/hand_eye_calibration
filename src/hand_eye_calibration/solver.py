from __future__ import annotations

import cv2
import numpy as np

from .models import CalibrationResult, Observation
from .se3 import as_transform


class HandEyeSolver:
    METHODS = {"park": cv2.CALIB_HAND_EYE_PARK}

    def __init__(self, method: str = "park"):
        normalized = method.lower()
        if normalized not in self.METHODS:
            raise ValueError(f"unsupported hand-eye method: {method}")
        self.method = normalized

    def solve(self, observations: list[Observation]) -> CalibrationResult:
        if len(observations) < 3:
            raise ValueError("OpenCV hand-eye calibration requires at least 3 observations")
        T_ee_base = [np.linalg.inv(obs.T_base_ee) for obs in observations]
        T_camera_target = [obs.T_camera_target for obs in observations]
        rotation, translation = cv2.calibrateHandEye(
            [item[:3, :3] for item in T_ee_base],
            [item[:3, 3] for item in T_ee_base],
            [item[:3, :3] for item in T_camera_target],
            [item[:3, 3] for item in T_camera_target],
            method=self.METHODS[self.method],
        )
        T_base_camera = np.eye(4, dtype=np.float64)
        T_base_camera[:3, :3] = rotation
        T_base_camera[:3, 3] = np.asarray(translation).reshape(3)
        return CalibrationResult(
            T_base_camera=as_transform(T_base_camera, name="T_base_camera"),
            method=self.method,
            observation_ids=tuple(obs.observation_id for obs in observations),
        )

