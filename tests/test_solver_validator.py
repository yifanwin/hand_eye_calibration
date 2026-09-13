import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hand_eye_calibration.models import Observation
from hand_eye_calibration.solver import HandEyeSolver
from hand_eye_calibration.validator import ValidationThresholds, Validator


def test_joint_solver_recovers_eye_to_hand_transform(synthetic_case):
    expected, observations = synthetic_case
    result = HandEyeSolver("park").solve(observations)
    translation_error = np.linalg.norm(result.T_base_camera[:3, 3] - expected[:3, 3])
    rotation_error = Rotation.from_matrix(result.T_base_camera[:3, :3].T @ expected[:3, :3]).magnitude()
    assert translation_error < 1e-6
    assert np.rad2deg(rotation_error) < 1e-5
    assert len(result.observation_ids) == 18


def test_solver_calls_opencv_once_with_all_rounds(monkeypatch, synthetic_case):
    expected, observations = synthetic_case
    import hand_eye_calibration.solver as solver_module
    original = solver_module.cv2.calibrateHandEye
    calls = []

    def wrapped(robot_rotations, robot_translations, target_rotations, target_translations, *, method):
        calls.append((len(robot_rotations), len(target_rotations)))
        return original(
            robot_rotations, robot_translations, target_rotations, target_translations, method=method
        )

    monkeypatch.setattr(solver_module.cv2, "calibrateHandEye", wrapped)
    result = HandEyeSolver().solve(observations)
    assert calls == [(18, 18)]
    np.testing.assert_allclose(result.T_base_camera, expected, atol=1e-6)


def test_validator_reports_consistency(synthetic_case):
    _, observations = synthetic_case
    result = HandEyeSolver().solve(observations)
    report = Validator().validate(observations, result)
    assert report.status == "pass"
    assert report.metrics["target_translation_rms_mm"] < 1e-6
    assert report.metrics["target_rotation_rms_deg"] < 1e-6


def test_validator_hard_fails_on_too_few_observations(synthetic_case):
    _, observations = synthetic_case
    with pytest.raises(ValueError, match="at least 10"):
        Validator().check_solvable(observations[:9])


def test_validator_hard_fails_on_degenerate_rotation(synthetic_case):
    _, observations = synthetic_case
    degenerate = []
    for obs in observations[:10]:
        payload = obs.to_dict()
        payload["T_base_ee"] = np.eye(4).tolist()
        degenerate.append(Observation.from_dict(payload))
    with pytest.raises(ValueError, match="rotation excitation"):
        Validator().check_solvable(degenerate)


def test_validator_warns_without_rejecting_result(synthetic_case):
    _, observations = synthetic_case
    noisy = []
    for obs in observations:
        data = obs.to_dict()
        data["reprojection_rmse_px"] = 3.0
        noisy.append(Observation.from_dict(data))
    result = HandEyeSolver().solve(noisy)
    report = Validator().validate(noisy, result)
    assert report.status == "warning"
    assert any("reprojection" in item for item in report.warnings)
