from __future__ import annotations

import importlib
import time

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ...interfaces import AdapterError, RobotAdapter
from ...models import DeviceInfo, RobotState


class HansCPSAdapter(RobotAdapter):
    def __init__(
        self,
        host: str,
        port: int = 10003,
        cps_module: str = "utils.CPS",
        box_id: int = 0,
        robot_id: int = 0,
        tcp_name: str = "TCP",
        ucs_name: str = "Base",
        speed_percent: float = 15.0,
        acceleration_percent: float = 100.0,
    ):
        self.host = host
        self.port = port
        self.cps_module = cps_module
        self.box_id = box_id
        self.robot_id = robot_id
        self.tcp_name = tcp_name
        self.ucs_name = ucs_name
        self.speed_percent = speed_percent
        self.acceleration_percent = acceleration_percent
        self._client = None

    @property
    def joint_count(self) -> int:
        return 6

    def connect(self) -> None:
        try:
            module = importlib.import_module(self.cps_module)
            self._client = module.CPSClient()
        except (ImportError, AttributeError) as exc:
            raise AdapterError(f"cannot load CPSClient from {self.cps_module}: {exc}") from exc
        result = self._client.HRIF_Connect(self.box_id, self.host, self.port)
        if result != 0:
            self._client = None
            raise AdapterError(f"Hans CPS connection failed with code {result}")

    def close(self) -> None:
        if self._client is not None:
            self._client.HRIF_DisConnect(self.box_id)
            self._client = None

    def get_info(self) -> DeviceInfo:
        self._require_connected()
        return DeviceInfo(
            adapter=f"{type(self).__module__}:{type(self).__name__}",
            name="Hans robot (CPS/HRIF)",
            metadata={
                "host": self.host, "port": self.port, "box_id": self.box_id,
                "robot_id": self.robot_id, "base_frame": self.ucs_name, "ee_frame": self.tcp_name,
            },
        )

    def move_joints(self, joint_positions_rad: NDArray[np.float64]) -> None:
        self._require_connected()
        joints = np.asarray(joint_positions_rad, dtype=np.float64).reshape(-1)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ValueError("Hans joint target must contain 6 finite radians")
        result = self._client.HRIF_MoveJ(
            self.box_id, self.robot_id, [0.0] * 6, np.rad2deg(joints).tolist(),
            self.tcp_name, self.ucs_name, self.speed_percent, self.acceleration_percent,
            0, 1, 0, 0, 0, "1",
        )
        if result != 0:
            raise AdapterError(f"Hans joint motion failed with code {result}")
        self._client.waitMoveDone(self.box_id, self.robot_id)

    def read_state(self) -> RobotState:
        self._require_connected()
        result_values: list[str] = []
        result = self._client.HRIF_ReadActPos(self.box_id, self.robot_id, result_values)
        if result != 0 or len(result_values) < 12:
            raise AdapterError(f"Hans state read failed with code {result}")
        joint_positions_rad = np.deg2rad(np.asarray(result_values[:6], dtype=np.float64))
        x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = map(float, result_values[6:12])
        T_base_ee = np.eye(4, dtype=np.float64)
        T_base_ee[:3, :3] = Rotation.from_euler(
            "xyz", [rx_deg, ry_deg, rz_deg], degrees=True
        ).as_matrix()
        T_base_ee[:3, 3] = np.array([x_mm, y_mm, z_mm]) / 1000.0
        return RobotState(
            joint_positions_rad=joint_positions_rad,
            T_base_ee=T_base_ee,
            timestamp_s=time.monotonic(),
            base_frame=self.ucs_name,
            ee_frame=self.tcp_name,
        )

    def _require_connected(self) -> None:
        if self._client is None:
            raise AdapterError("Hans robot is not connected")

