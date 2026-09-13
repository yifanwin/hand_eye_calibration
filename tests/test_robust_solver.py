import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hand_eye_calibration.models import Observation
from hand_eye_calibration.solver import HandEyeSolver, RejectionSettings, RobustHandEyeSolver
from hand_eye_calibration.validator import Validator


def contaminate(obs: Observation, shift_m: float, rotation_deg: float = 0.0) -> Observation:
    """给 T_camera_target 注入平移/旋转扰动,模拟一次坏观测。"""
    data = obs.to_dict()
    T = np.asarray(data["T_camera_target"], dtype=np.float64).copy()
    T[:3, 3] += np.array([shift_m, 0.0, 0.0])
    if rotation_deg:
        T[:3, :3] = Rotation.from_euler("z", rotation_deg, degrees=True).as_matrix() @ T[:3, :3]
    data["T_camera_target"] = T.tolist()
    return Observation.from_dict(data)


def transform_errors(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    translation_error = np.linalg.norm(actual[:3, 3] - expected[:3, 3])
    rotation_error_deg = np.rad2deg(
        Rotation.from_matrix(actual[:3, :3].T @ expected[:3, :3]).magnitude()
    )
    return float(translation_error), float(rotation_error_deg)


def test_robust_solver_removes_contaminated_observations(synthetic_case):
    expected, observations = synthetic_case
    contaminated = [
        contaminate(obs, shift_m=0.05) if index in {0, 6, 12} else obs
        for index, obs in enumerate(observations)
    ]
    plain = HandEyeSolver().solve(contaminated)
    plain_translation, plain_rotation = transform_errors(plain.T_base_camera, expected)
    assert plain_translation > 0.001  # 污染确实破坏了普通求解

    outcome = RobustHandEyeSolver(HandEyeSolver(), Validator()).solve(contaminated)
    assert set(outcome.rejected_ids) == {"obs-0", "obs-6", "obs-12"}
    assert len(outcome.kept) == 15
    assert outcome.iterations >= 1
    translation_error, rotation_error_deg = transform_errors(outcome.result.T_base_camera, expected)
    assert translation_error < 1e-6
    assert rotation_error_deg < 1e-5


def test_robust_solver_keeps_clean_dataset_intact(synthetic_case):
    _, observations = synthetic_case
    outcome = RobustHandEyeSolver(HandEyeSolver(), Validator()).solve(observations)
    assert outcome.rejected_ids == ()
    assert outcome.iterations == 0
    assert len(outcome.kept) == len(observations)
    expected = HandEyeSolver().solve(observations)
    np.testing.assert_allclose(outcome.result.T_base_camera, expected.T_base_camera, atol=1e-9)


def test_robust_solver_respects_min_keep(synthetic_case):
    _, observations = synthetic_case
    contaminated = [
        contaminate(obs, shift_m=0.02 + 0.005 * index) if index < 16 else obs
        for index, obs in enumerate(observations)
    ]
    settings = RejectionSettings(max_removal_fraction=1.0)
    outcome = RobustHandEyeSolver(HandEyeSolver(), Validator(), settings).solve(contaminated)
    min_keep = Validator().thresholds.min_observations
    assert len(outcome.kept) == min_keep
    assert len(outcome.rejected_ids) == len(observations) - min_keep


def test_mad_mask_flags_tails_in_normal_case():
    solver = RobustHandEyeSolver(HandEyeSolver(), Validator())
    errors = np.array([1.0, 1.2, 0.9, 1.1, 1.05, 5.0])
    mask = solver._mad_mask(errors, absolute_floor=100.0)
    assert mask.tolist() == [False, False, False, False, False, True]


def test_mad_mask_falls_back_to_absolute_floor_when_scale_degenerates():
    solver = RobustHandEyeSolver(HandEyeSolver(), Validator())
    errors = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 8.0])
    mask = solver._mad_mask(errors, absolute_floor=5.0)
    assert mask.tolist() == [False, False, False, False, False, True]
