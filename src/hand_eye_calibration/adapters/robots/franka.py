from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from ...interfaces import AdapterError, RobotAdapter
from ...models import DeviceInfo, RobotState


class FrankaRobotAdapter(RobotAdapter):
    def __init__(
        self,
        host: str,
        relative_dynamics_factor: float = 0.05,
        base_frame: str = "base",
        ee_frame: str = "end_effector",
    ):
        self.host = host
        if not 0.0 < relative_dynamics_factor <= 1.0:
            raise ValueError("relative_dynamics_factor must be in (0, 1]")
        self.relative_dynamics_factor = relative_dynamics_factor
        self.base_frame = base_frame
        self.ee_frame = ee_frame
        self._robot = None
        self._joint_motion = None

    @property
    def joint_count(self) -> int:
        return 7

    def connect(self) -> None:
        try:
            from franky import JointMotion, Robot
        except ImportError as exc:
            raise AdapterError("Franka adapter requires the 'franky-control' package") from exc
        try:
            self._robot = Robot(self.host)
            self._robot.relative_dynamics_factor = self.relative_dynamics_factor
            self._robot.recover_from_errors()
            self._joint_motion = JointMotion
        except Exception as exc:
            self._robot = None
            raise AdapterError(f"failed to connect to Franka at {self.host}: {exc}") from exc

    def close(self) -> None:
        self._robot = None
        self._joint_motion = None

    def get_info(self) -> DeviceInfo:
        self._require_connected()
        return DeviceInfo(
            adapter=f"{type(self).__module__}:{type(self).__name__}",
            name="Franka Research 3",
            metadata={"host": self.host, "base_frame": self.base_frame, "ee_frame": self.ee_frame},
        )

    def move_joints(self, joint_positions_rad: NDArray[np.float64]) -> None:
        self._require_connected()
        joints = np.asarray(joint_positions_rad, dtype=np.float64).reshape(-1)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError("Franka joint target must contain 7 finite radians")
        try:
            self._robot.move(self._joint_motion(joints.tolist()))
        except Exception as exc:
            try:
                self._robot.recover_from_errors()
            except Exception:
                pass
            raise AdapterError(f"Franka joint motion failed: {exc}") from exc

    def read_state(self) -> RobotState:
        self._require_connected()
        try:
            joints = np.asarray(self._robot.current_joint_state.position, dtype=np.float64)
            pose = np.asarray(
                self._robot.current_cartesian_state.pose.end_effector_pose.matrix,
                dtype=np.float64,
            )
        except Exception as exc:
            raise AdapterError(f"failed to read Franka state: {exc}") from exc
        return RobotState(
            joint_positions_rad=joints,
            T_base_ee=pose,
            timestamp_s=time.monotonic(),
            base_frame=self.base_frame,
            ee_frame=self.ee_frame,
        )

    def _require_connected(self) -> None:
        if self._robot is None:
            raise AdapterError("Franka robot is not connected")
