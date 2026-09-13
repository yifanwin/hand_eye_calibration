import json

import numpy as np

from hand_eye_calibration.exporter import Exporter
from hand_eye_calibration.models import (
    CameraCalibration, DeviceInfo, IntrinsicCalibration, StreamProfile, ValidationReport,
)
from hand_eye_calibration.session import CalibrationSession
from hand_eye_calibration.solver import HandEyeSolver


def camera_calibration():
    intrinsics = IntrinsicCalibration(640, 480, 500.0, 501.0, 320.0, 240.0, "none", (0, 0, 0, 0, 0))
    return CameraCalibration(
        device=DeviceInfo("fake:Camera", "Fake RGB-D", "camera-1"),
        color_profile=StreamProfile(640, 480, 30, "bgr8"),
        depth_profile=StreamProfile(640, 480, 30, "z16"),
        color_intrinsics=intrinsics, depth_intrinsics=intrinsics,
        depth_scale_m_per_unit=0.001, T_color_depth=np.eye(4),
    )


def test_session_appends_multiple_runs_and_exports_distinct_transforms(tmp_path, synthetic_case):
    _, observations = synthetic_case
    session = CalibrationSession.create(
        tmp_path / "session-a", config_snapshot={"schema_version": 1},
        robot_info=DeviceInfo("fake:Robot", "Fake robot", "robot-1"),
        camera_calibration=camera_calibration(),
    )
    for run_number in range(3):
        run_id = session.begin_run(trajectory="synthetic", waypoint_count=6)
        selected = observations[run_number * 6:(run_number + 1) * 6]
        for obs in selected:
            payload = obs.to_dict()
            payload["run_id"] = run_id
            session.append_observation(type(obs).from_dict(payload))
        session.finish_run(run_id, attempted=6, accepted=6)
    loaded = CalibrationSession.load(session.path)
    assert len(loaded.observations()) == 18
    result = HandEyeSolver().solve(loaded.observations())
    report = ValidationReport("pass", {"observation_count": 18.0})
    loaded.save_result(result, report)
    stored_result = json.loads((loaded.path / "result.json").read_text())
    assert "camera_calibration" in stored_result
    output = Exporter().export(loaded, result, report, tmp_path / "calibration.json")
    data = json.loads(output.read_text())
    assert "T_base_camera" in data
    assert "T_color_depth" in data["camera_calibration"]
    assert "extrinsics" not in data


def test_session_rejects_different_camera(tmp_path):
    session = CalibrationSession.create(
        tmp_path / "session-b", config_snapshot={"schema_version": 1},
        robot_info=DeviceInfo("fake:Robot", "Fake robot", "robot-1"),
        camera_calibration=camera_calibration(),
    )
    changed = camera_calibration().to_dict()
    changed["device"]["serial_number"] = "camera-2"
    with np.testing.assert_raises(ValueError):
        session.assert_compatible(
            DeviceInfo("fake:Robot", "Fake robot", "robot-1"),
            CameraCalibration.from_dict(changed),
        )
