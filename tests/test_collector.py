import time

import numpy as np

from hand_eye_calibration.collector import Collector
from hand_eye_calibration.interfaces import CameraAdapter, RobotAdapter, TargetDetector
from hand_eye_calibration.models import (
    CameraCalibration, CameraFrame, ColorProjection, DetectionResult, DeviceInfo,
    IntrinsicCalibration, RobotState, StreamProfile,
)
from hand_eye_calibration.session import CalibrationSession


class FakeRobot(RobotAdapter):
    def __init__(self): self.q = np.zeros(3)
    @property
    def joint_count(self): return 3
    def connect(self): pass
    def close(self): pass
    def get_info(self): return DeviceInfo("tests:FakeRobot", "fake", "robot")
    def move_joints(self, joint_positions_rad): self.q = np.asarray(joint_positions_rad)
    def read_state(self):
        transform = np.eye(4)
        transform[:3, 3] = self.q
        return RobotState(self.q, transform, time.monotonic())


class FakeCamera(CameraAdapter):
    def __init__(self):
        intr = IntrinsicCalibration(16, 12, 10, 10, 8, 6, "none", (0, 0, 0, 0, 0))
        self.calibration = CameraCalibration(
            DeviceInfo("tests:FakeCamera", "fake", "camera"),
            StreamProfile(16, 12, 30, "bgr8"), StreamProfile(16, 12, 30, "z16"),
            intr, intr, 0.001, np.eye(4),
        )
    def start(self): pass
    def close(self): pass
    def get_calibration(self): return self.calibration
    def capture(self):
        projection = ColorProjection(self.calibration.color_intrinsics.camera_matrix, np.zeros(5), "none")
        return CameraFrame(np.zeros((12, 16, 3), dtype=np.uint8), projection, time.time())


class FakeDetector(TargetDetector):
    def detect(self, color_bgr, color_projection):
        return DetectionResult(np.eye(4), np.zeros((4, 2)), 0.1)


def test_collector_appends_runs_without_solving(tmp_path):
    robot, camera = FakeRobot(), FakeCamera()
    session = CalibrationSession.create(
        tmp_path / "session", config_snapshot={"schema_version": 1},
        robot_info=robot.get_info(), camera_calibration=camera.get_calibration(),
    )
    collector = Collector(robot, camera, FakeDetector(), session, settle_time_s=0, save_images=False)
    trajectory = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    collector.collect_run(trajectory, trajectory_name="fake")
    collector.collect_run(trajectory, trajectory_name="fake")
    assert len(session.observations()) == 4
    assert len({item.run_id for item in session.observations()}) == 2
    assert not (session.path / "result.json").exists()

