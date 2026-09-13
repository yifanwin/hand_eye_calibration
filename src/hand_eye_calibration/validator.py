from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .models import CalibrationResult, Observation, ValidationReport


@dataclass(frozen=True)
class ValidationThresholds:
    min_observations: int = 10
    min_max_relative_rotation_deg: float = 15.0
    min_rotation_axis_rank: int = 2
    reprojection_median_px: float = 1.0
    reprojection_p95_px: float = 2.0
    target_translation_rms_mm: float = 5.0
    target_rotation_rms_deg: float = 0.5
    translation_span_m: float = 0.1
    rotation_span_deg: float = 30.0


class Validator:
    def __init__(self, thresholds: ValidationThresholds | None = None):
        self.thresholds = thresholds or ValidationThresholds()

    def check_solvable(self, observations: list[Observation]) -> dict[str, float]:
        if len(observations) < self.thresholds.min_observations:
            raise ValueError(
                f"need at least {self.thresholds.min_observations} valid observations, got {len(observations)}"
            )
        transforms = [obs.T_base_ee for obs in observations]
        relative_vectors = []
        for transform in transforms[1:]:
            relative = np.linalg.inv(transforms[0][:3, :3]) @ transform[:3, :3]
            vector, _ = cv2.Rodrigues(relative)
            relative_vectors.append(vector.reshape(3))
        relative_array = np.asarray(relative_vectors)
        max_rotation_deg = float(np.max(np.linalg.norm(relative_array, axis=1)) * 180.0 / np.pi)
        axis_rank = int(np.linalg.matrix_rank(relative_array, tol=np.deg2rad(1.0)))
        if max_rotation_deg < self.thresholds.min_max_relative_rotation_deg:
            raise ValueError(f"robot rotation excitation is too small: {max_rotation_deg:.3f} deg")
        if axis_rank < self.thresholds.min_rotation_axis_rank:
            raise ValueError(f"robot rotation axes are degenerate: rank={axis_rank}")
        return {"observation_count": float(len(observations)), "rotation_axis_rank": float(axis_rank)}

    def validate(self, observations: list[Observation], result: CalibrationResult) -> ValidationReport:
        metrics = self.check_solvable(observations)
        reprojection = np.array([obs.reprojection_rmse_px for obs in observations], dtype=np.float64)
        T_ee_target = [
            np.linalg.inv(obs.T_base_ee) @ result.T_base_camera @ obs.T_camera_target
            for obs in observations
        ]
        translations = np.array([item[:3, 3] for item in T_ee_target])
        rotations = Rotation.from_matrix([item[:3, :3] for item in T_ee_target])
        mean_rotation = rotations.mean()
        translation_errors_mm = np.linalg.norm(translations - translations.mean(axis=0), axis=1) * 1000.0
        rotation_errors_deg = (mean_rotation.inv() * rotations).magnitude() * 180.0 / np.pi
        positions = np.array([obs.T_base_ee[:3, 3] for obs in observations])
        translation_span = float(np.max(np.linalg.norm(positions[:, None] - positions[None, :], axis=2)))
        rotation_span = self._max_rotation_span_deg(observations)
        metrics.update({
            "reprojection_median_px": float(np.median(reprojection)),
            "reprojection_p95_px": float(np.percentile(reprojection, 95)),
            "target_translation_rms_mm": float(np.sqrt(np.mean(translation_errors_mm ** 2))),
            "target_rotation_rms_deg": float(np.sqrt(np.mean(rotation_errors_deg ** 2))),
            "robot_translation_span_m": translation_span,
            "robot_rotation_span_deg": rotation_span,
        })
        warnings: list[str] = []
        checks = (
            ("reprojection_median_px", self.thresholds.reprojection_median_px),
            ("reprojection_p95_px", self.thresholds.reprojection_p95_px),
            ("target_translation_rms_mm", self.thresholds.target_translation_rms_mm),
            ("target_rotation_rms_deg", self.thresholds.target_rotation_rms_deg),
        )
        for name, limit in checks:
            if metrics[name] > limit:
                warnings.append(f"{name}={metrics[name]:.6g} exceeds {limit}")
        if translation_span < self.thresholds.translation_span_m:
            warnings.append(f"robot_translation_span_m={translation_span:.6g} is below {self.thresholds.translation_span_m}")
        if rotation_span < self.thresholds.rotation_span_deg:
            warnings.append(f"robot_rotation_span_deg={rotation_span:.6g} is below {self.thresholds.rotation_span_deg}")
        outliers = tuple(
            obs.observation_id for obs, te, re in zip(observations, translation_errors_mm, rotation_errors_deg)
            if te > 3 * self.thresholds.target_translation_rms_mm
            or re > 3 * self.thresholds.target_rotation_rms_deg
        )
        return ValidationReport(
            status="warning" if warnings else "pass",
            metrics=metrics,
            warnings=tuple(warnings),
            outlier_observation_ids=outliers,
        )

    @staticmethod
    def _max_rotation_span_deg(observations: list[Observation]) -> float:
        maximum = 0.0
        rotations = [obs.T_base_ee[:3, :3] for obs in observations]
        for index, left in enumerate(rotations):
            for right in rotations[index + 1:]:
                vector, _ = cv2.Rodrigues(left.T @ right)
                maximum = max(maximum, float(np.linalg.norm(vector) * 180.0 / np.pi))
        return maximum

