from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import NDArray

from .models import CameraCalibration, CameraFrame, DetectionResult, DeviceInfo, RobotState


class AdapterError(RuntimeError):
    pass


class RobotAdapter(ABC):
    @property
    @abstractmethod
    def joint_count(self) -> int: ...

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def get_info(self) -> DeviceInfo: ...

    @abstractmethod
    def move_joints(self, joint_positions_rad: NDArray[Any]) -> None: ...

    @abstractmethod
    def read_state(self) -> RobotState: ...


class CameraAdapter(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def get_calibration(self) -> CameraCalibration: ...

    @abstractmethod
    def capture(self) -> CameraFrame: ...


class TargetDetector(ABC):
    @abstractmethod
    def detect(self, color_bgr: NDArray[Any], color_projection: Any) -> DetectionResult | None: ...

