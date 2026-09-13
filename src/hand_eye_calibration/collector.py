from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray

from .interfaces import CameraAdapter, RobotAdapter, TargetDetector
from .models import Observation, utc_now
from .session import CalibrationSession

# 采图与关节角对齐参数
SYNC_TOLERANCE_S = 0.1             # 图像获取时刻与机器人状态时刻允许的最大偏差（秒）
STATIONARITY_TOLERANCE_RAD = 0.002  # 采图前后关节角允许的最大变化（弧度，约 0.11 度）
ALIGN_MAX_ATTEMPTS = 3             # 对齐校验失败时的最大重试次数


class Collector:
    def __init__(
        self,
        robot: RobotAdapter,
        camera: CameraAdapter,
        detector: TargetDetector,
        session: CalibrationSession,
        *,
        settle_time_s: float = 1.0,
        save_images: bool = True,
        sync_tolerance_s: float = SYNC_TOLERANCE_S,
        stationarity_tolerance_rad: float = STATIONARITY_TOLERANCE_RAD,
        align_max_attempts: int = ALIGN_MAX_ATTEMPTS,
    ):
        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.session = session
        self.settle_time_s = settle_time_s
        self.save_images = save_images
        self.sync_tolerance_s = sync_tolerance_s
        self.stationarity_tolerance_rad = stationarity_tolerance_rad
        self.align_max_attempts = align_max_attempts

    def collect_run(self, trajectory: NDArray[np.float64], *, trajectory_name: str) -> str:
        waypoints = np.asarray(trajectory, dtype=np.float64)
        if waypoints.ndim != 2 or waypoints.shape[1] != self.robot.joint_count:
            raise ValueError("trajectory shape does not match robot joint count")
        run_id = self.session.begin_run(trajectory=trajectory_name, waypoint_count=len(waypoints))
        attempted = accepted = 0
        try:
            for sequence, waypoint in enumerate(waypoints):
                attempted += 1
                try:
                    self.robot.move_joints(waypoint)
                    time.sleep(self.settle_time_s)
                    frame, state = self._capture_aligned(sequence)
                    if frame is None:
                        continue
                    detection = self.detector.detect(frame.color_bgr, frame.color_projection)
                    if detection is None:
                        continue
                    observation_id = uuid4().hex
                    raw_path = debug_path = None
                    if self.save_images:
                        # 只保存划线标注后的图片，不保存原始图
                        image_dir = self.session.path / "images"
                        if detection.debug_image_bgr is not None:
                            debug = image_dir / f"{observation_id}_debug.png"
                            if not cv2.imwrite(str(debug), detection.debug_image_bgr):
                                raise OSError(f"failed to write {debug}")
                            debug_path = str(debug.relative_to(self.session.path))
                    observation = Observation(
                        observation_id=observation_id,
                        run_id=run_id,
                        sequence=sequence,
                        captured_at_utc=utc_now(),
                        robot_timestamp_s=state.timestamp_s,
                        camera_timestamp_s=frame.device_timestamp_s,
                        camera_host_timestamp_s=frame.host_timestamp_s,
                        joint_positions_rad=state.joint_positions_rad,
                        T_base_ee=state.T_base_ee,
                        T_camera_target=detection.T_camera_target,
                        reprojection_rmse_px=detection.reprojection_rmse_px,
                        color_image_path=raw_path,
                        debug_image_path=debug_path,
                    )
                    self.session.append_observation(observation)
                    accepted += 1
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    print(f"waypoint {sequence} skipped: {exc}")
            self.session.finish_run(run_id, attempted=attempted, accepted=accepted)
            return run_id
        except BaseException as exc:
            self.session.finish_run(run_id, attempted=attempted, accepted=accepted, error=str(exc))
            raise

    def _capture_aligned(self, sequence: int):
        """采图与关节角对齐：图像曝光时刻被夹在前后两次机器人读数之间。

        通过"采图前读数 → 采图 → 采图后读数"的包围校验，确保机器人在整个
        采图窗口内保持静止（前后关节角一致），且机器人状态时刻与图像获取
        时刻的偏差在容差内。校验失败时延时重试，超过次数返回 None。
        """
        for attempt in range(self.align_max_attempts):
            state_before = self.robot.read_state()
            frame = self.camera.capture()
            state_after = self.robot.read_state()
            # 机器人与相机时间戳均为宿主 wall-clock 时钟域，可直接比较
            sync_error_s = abs(state_after.timestamp_s - frame.host_timestamp_s)
            joint_drift_rad = float(np.max(np.abs(
                state_after.joint_positions_rad - state_before.joint_positions_rad
            )))
            if joint_drift_rad <= self.stationarity_tolerance_rad and sync_error_s <= self.sync_tolerance_s:
                # 机器人已验证静止，前后状态等价，取时刻更靠近采图的后读数
                return frame, state_after
            print(
                f"waypoint {sequence} alignment attempt {attempt + 1}/{self.align_max_attempts} "
                f"rejected: joint_drift={joint_drift_rad:.5f} rad, sync_error={sync_error_s:.3f} s"
            )
            time.sleep(self.settle_time_s)
        print(f"waypoint {sequence} skipped: alignment not achieved after {self.align_max_attempts} attempts")
        return None, None

