from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import CalibrationResult, Observation
from .se3 import as_transform
from .validator import Validator

# ===== 鲁棒剔除超参数(可在配置文件 rejection 段覆盖)=====
MAD_TO_SIGMA = 1.4826  # 正态假设下 MAD 换算等效标准差的系数
NUMERIC_ZERO_TOLERANCE = 1e-9  # MAD 尺度低于该值视为残差无离散,退回绝对阈值判据


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


@dataclass(frozen=True)
class RejectionSettings:
    mad_multiplier: float = 3.0        # MAD 判据倍数(约等于 3σ 的鲁棒版本)
    max_iterations: int = 10           # 最大剔除轮数
    max_removal_fraction: float = 0.2  # 每轮最多剔除当前观测数的比例


@dataclass(frozen=True)
class RobustSolveOutcome:
    result: CalibrationResult
    kept: list[Observation]
    rejected_ids: tuple[str, ...]
    iterations: int


class RobustHandEyeSolver:
    """迭代剔除离群观测并重解。

    每轮流程:求解 -> 计算每个观测的靶标一致性误差 -> MAD 自适应判据标记
    离群候选 -> 按误差严重程度剔除最多 max_removal_fraction -> 重解。
    保护:剔除后剩余数不得低于 validator 的 min_observations;
    MAD 尺度退化(残差几乎一致)时退回绝对阈值判据,避免误剔完美数据。
    """

    def __init__(
        self,
        base_solver: HandEyeSolver,
        validator: Validator,
        settings: RejectionSettings | None = None,
    ):
        self.base_solver = base_solver
        self.validator = validator
        self.settings = settings or RejectionSettings()

    def solve(self, observations: list[Observation]) -> RobustSolveOutcome:
        kept = list(observations)
        rejected_ids: list[str] = []
        min_keep = self.validator.thresholds.min_observations
        iterations = 0
        result = self.base_solver.solve(kept)
        for round_index in range(1, self.settings.max_iterations + 1):
            translation_errors_mm, rotation_errors_deg = self.validator.per_observation_errors(kept, result)
            thresholds = self.validator.thresholds
            mask = (
                self._mad_mask(translation_errors_mm, thresholds.target_translation_rms_mm)
                | self._mad_mask(rotation_errors_deg, thresholds.target_rotation_rms_deg)
            )
            if not mask.any():
                break
            max_drop = min(int(len(kept) * self.settings.max_removal_fraction), len(kept) - min_keep)
            if max_drop <= 0:
                break
            # 严重程度 = 误差超出对应验证阈值的倍数(取平移/旋转中更大者),只剔最差的一批
            scores = np.maximum(
                translation_errors_mm / thresholds.target_translation_rms_mm,
                rotation_errors_deg / thresholds.target_rotation_rms_deg,
            )
            order = np.argsort(np.where(mask, scores, -np.inf))[::-1]
            drop = {int(index) for index in order[:max_drop]}
            rejected_ids.extend(obs.observation_id for index, obs in enumerate(kept) if index in drop)
            kept = [obs for index, obs in enumerate(kept) if index not in drop]
            iterations = round_index
            result = self.base_solver.solve(kept)
        return RobustSolveOutcome(
            result=result, kept=kept, rejected_ids=tuple(rejected_ids), iterations=iterations
        )

    def _mad_mask(self, errors: np.ndarray, absolute_floor: float) -> np.ndarray:
        median = float(np.median(errors))
        scale = MAD_TO_SIGMA * float(np.median(np.abs(errors - median)))
        if scale < NUMERIC_ZERO_TOLERANCE:
            return errors > absolute_floor
        return errors > median + self.settings.mad_multiplier * scale

