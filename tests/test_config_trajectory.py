import numpy as np

from hand_eye_calibration.config import construct, load_config
from hand_eye_calibration.detectors.checkerboard import ChessboardDetector
from hand_eye_calibration.trajectory import load_trajectory, save_trajectory


def test_class_path_factory_needs_no_core_registry():
    detector = construct({
        "factory": "hand_eye_calibration.detectors.checkerboard:ChessboardDetector",
        "options": {"columns": 7, "rows": 5, "square_size_m": 0.02},
    })
    assert isinstance(detector, ChessboardDetector)


def test_trajectory_is_radians_only(tmp_path):
    path = tmp_path / "trajectory.yaml"
    expected = np.arange(21, dtype=float).reshape(3, 7) / 10
    save_trajectory(path, expected, robot="fake")
    np.testing.assert_allclose(load_trajectory(path, joint_count=7), expected)


def test_example_configuration_loads():
    config = load_config("configs/fr3_d435.yaml")
    assert config["solver"]["method"] == "park"

