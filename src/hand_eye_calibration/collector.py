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
    ):
        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.session = session
        self.settle_time_s = settle_time_s
        self.save_images = save_images

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
                    state = self.robot.read_state()
                    frame = self.camera.capture()
                    detection = self.detector.detect(frame.color_bgr, frame.color_projection)
                    if detection is None:
                        continue
                    observation_id = uuid4().hex
                    raw_path = debug_path = None
                    if self.save_images:
                        image_dir = self.session.path / "images"
                        raw = image_dir / f"{observation_id}_color.png"
                        if not cv2.imwrite(str(raw), frame.color_bgr):
                            raise OSError(f"failed to write {raw}")
                        raw_path = str(raw.relative_to(self.session.path))
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

