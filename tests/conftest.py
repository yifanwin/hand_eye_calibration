from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hand_eye_calibration.models import Observation


def make_transform(euler_deg, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", euler_deg, degrees=True).as_matrix()
    transform[:3, 3] = translation
    return transform


@pytest.fixture
def synthetic_case():
    T_base_camera = make_transform([8.0, -17.0, 31.0], [0.42, -0.18, 0.91])
    T_ee_target = make_transform([2.0, 5.0, -4.0], [0.0, 0.0, 0.14])
    observations = []
    for index in range(18):
        euler = [
            -35.0 + 4.3 * index,
            22.0 * np.sin(index * 0.71),
            -28.0 * np.cos(index * 0.47),
        ]
        translation = [
            0.35 + 0.12 * np.sin(index * 0.4),
            -0.15 + 0.11 * np.cos(index * 0.53),
            0.42 + 0.08 * np.sin(index * 0.31),
        ]
        T_base_ee = make_transform(euler, translation)
        T_camera_target = np.linalg.inv(T_base_camera) @ T_base_ee @ T_ee_target
        observations.append(Observation(
            observation_id=f"obs-{index}", run_id=f"run-{index // 6}", sequence=index % 6,
            captured_at_utc="2026-01-01T00:00:00+00:00",
            robot_timestamp_s=float(index), camera_timestamp_s=float(index),
            joint_positions_rad=np.linspace(0.0, 0.6, 7) + index * 0.01,
            T_base_ee=T_base_ee, T_camera_target=T_camera_target,
            reprojection_rmse_px=0.2,
        ))
    return T_base_camera, observations

