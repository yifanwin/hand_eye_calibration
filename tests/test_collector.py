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
        return RobotState(self.q, transform, time.time())


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
    # 机器人与相机时间戳处于同一 wall-clock 时钟域，偏差应在容差内
    for item in session.observations():
        assert item.camera_host_timestamp_s is not None
        assert abs(item.camera_host_timestamp_s - item.robot_timestamp_s) < 1.0


class DriftingRobot(FakeRobot):
    """采图前后关节角仍在漂移，模拟未真正静止的机器人。"""
    def __init__(self):
        super().__init__()
        self.read_count = 0
    def read_state(self):
        self.read_count += 1
        self.q = self.q + 0.01  # 每次读数漂移 0.01 rad，超过静止容差
        return super().read_state()


class OffsetClockRobot(FakeRobot):
    """时间戳与相机处于不同时钟域，模拟采图与关节角时刻不对齐。"""
    def read_state(self):
        state = super().read_state()
        return RobotState(state.joint_positions_rad, state.T_base_ee, time.time() + 10.0)


def _make_session(tmp_path, robot, camera):
    return CalibrationSession.create(
        tmp_path / "session", config_snapshot={"schema_version": 1},
        robot_info=robot.get_info(), camera_calibration=camera.get_calibration(),
    )


def test_collector_skips_unsettled_robot(tmp_path):
    robot, camera = DriftingRobot(), FakeCamera()
    session = _make_session(tmp_path, robot, camera)
    collector = Collector(robot, camera, FakeDetector(), session, settle_time_s=0, save_images=False)
    collector.collect_run(np.array([[0.1, 0.2, 0.3]]), trajectory_name="fake")
    assert session.observations() == []


def test_collector_skips_clock_misaligned_robot(tmp_path):
    robot, camera = OffsetClockRobot(), FakeCamera()
    session = _make_session(tmp_path, robot, camera)
    collector = Collector(robot, camera, FakeDetector(), session, settle_time_s=0, save_images=False)
    collector.collect_run(np.array([[0.1, 0.2, 0.3]]), trajectory_name="fake")
    assert session.observations() == []

