from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .se3 import as_transform


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class DeviceInfo:
    adapter: str
    name: str
    serial_number: str = ""
    firmware_version: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "name": self.name,
            "serial_number": self.serial_number,
            "firmware_version": self.firmware_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeviceInfo":
        return cls(**data)


@dataclass(frozen=True)
class StreamProfile:
    width: int
    height: int
    fps: int
    format: str

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fps <= 0 or not self.format:
            raise ValueError("invalid stream profile")

    def to_dict(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height, "fps": self.fps, "format": self.format}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StreamProfile":
        return cls(**data)


@dataclass(frozen=True)
class IntrinsicCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    distortion_coeffs: tuple[float, ...]

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.fx, self.fy, self.cx, self.cy, *self.distortion_coeffs], dtype=np.float64
        )
        if self.width <= 0 or self.height <= 0 or self.fx <= 0 or self.fy <= 0:
            raise ValueError("invalid camera intrinsic dimensions or focal length")
        if not np.all(np.isfinite(values)) or not self.distortion_model:
            raise ValueError("invalid camera intrinsic values")

    @property
    def camera_matrix(self) -> NDArray[np.float64]:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion_model": self.distortion_model,
            "distortion_coeffs": list(self.distortion_coeffs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntrinsicCalibration":
        copy = dict(data)
        copy["distortion_coeffs"] = tuple(copy["distortion_coeffs"])
        return cls(**copy)


@dataclass(frozen=True)
class ColorProjection:
    """OpenCV-compatible color projection used by TargetDetector."""

    camera_matrix: NDArray[np.float64]
    distortion_coeffs: NDArray[np.float64]
    distortion_model: str = "opencv_brown_conrady"

    def __post_init__(self) -> None:
        k = np.asarray(self.camera_matrix, dtype=np.float64)
        d = np.asarray(self.distortion_coeffs, dtype=np.float64).reshape(-1)
        if k.shape != (3, 3) or not np.all(np.isfinite(k)) or not np.all(np.isfinite(d)):
            raise ValueError("invalid color projection")
        if self.distortion_model not in {"none", "opencv_brown_conrady"}:
            raise ValueError(f"unsupported detector projection model: {self.distortion_model}")
        object.__setattr__(self, "camera_matrix", k)
        object.__setattr__(self, "distortion_coeffs", d)


@dataclass(frozen=True)
class CameraCalibration:
    device: DeviceInfo
    color_profile: StreamProfile
    depth_profile: StreamProfile
    color_intrinsics: IntrinsicCalibration
    depth_intrinsics: IntrinsicCalibration
    depth_scale_m_per_unit: float
    T_color_depth: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not np.isfinite(self.depth_scale_m_per_unit) or self.depth_scale_m_per_unit <= 0:
            raise ValueError("depth_scale_m_per_unit must be positive")
        object.__setattr__(self, "T_color_depth", as_transform(self.T_color_depth, name="T_color_depth"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device.to_dict(),
            "color_profile": self.color_profile.to_dict(),
            "depth_profile": self.depth_profile.to_dict(),
            "color_intrinsics": self.color_intrinsics.to_dict(),
            "depth_intrinsics": self.depth_intrinsics.to_dict(),
            "depth_scale_m_per_unit": self.depth_scale_m_per_unit,
            "T_color_depth": self.T_color_depth.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraCalibration":
        return cls(
            device=DeviceInfo.from_dict(data["device"]),
            color_profile=StreamProfile.from_dict(data["color_profile"]),
            depth_profile=StreamProfile.from_dict(data["depth_profile"]),
            color_intrinsics=IntrinsicCalibration.from_dict(data["color_intrinsics"]),
            depth_intrinsics=IntrinsicCalibration.from_dict(data["depth_intrinsics"]),
            depth_scale_m_per_unit=float(data["depth_scale_m_per_unit"]),
            T_color_depth=data["T_color_depth"],
        )


@dataclass(frozen=True)
class RobotState:
    joint_positions_rad: NDArray[np.float64]
    T_base_ee: NDArray[np.float64]
    timestamp_s: float
    base_frame: str = "base"
    ee_frame: str = "ee"

    def __post_init__(self) -> None:
        q = np.asarray(self.joint_positions_rad, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(q)) or not np.isfinite(self.timestamp_s):
            raise ValueError("invalid robot state")
        object.__setattr__(self, "joint_positions_rad", q)
        object.__setattr__(self, "T_base_ee", as_transform(self.T_base_ee, name="T_base_ee"))


@dataclass(frozen=True)
class CameraFrame:
    color_bgr: NDArray[np.uint8]
    color_projection: ColorProjection
    host_timestamp_s: float
    device_timestamp_s: float | None = None
    depth_raw: NDArray[Any] | None = None

    def __post_init__(self) -> None:
        image = np.asarray(self.color_bgr)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("color_bgr must be an HxWx3 uint8 image")
        object.__setattr__(self, "color_bgr", image)


@dataclass(frozen=True)
class DetectionResult:
    T_camera_target: NDArray[np.float64]
    image_points_px: NDArray[np.float64]
    reprojection_rmse_px: float
    debug_image_bgr: NDArray[np.uint8] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "T_camera_target", as_transform(self.T_camera_target, name="T_camera_target"))
        object.__setattr__(self, "image_points_px", np.asarray(self.image_points_px, dtype=np.float64).reshape(-1, 2))


@dataclass(frozen=True)
class Observation:
    observation_id: str
    run_id: str
    sequence: int
    captured_at_utc: str
    robot_timestamp_s: float
    camera_timestamp_s: float | None
    joint_positions_rad: NDArray[np.float64]
    T_base_ee: NDArray[np.float64]
    T_camera_target: NDArray[np.float64]
    reprojection_rmse_px: float
    color_image_path: str | None = None
    debug_image_path: str | None = None

    def __post_init__(self) -> None:
        q = np.asarray(self.joint_positions_rad, dtype=np.float64).reshape(-1)
        if (
            not np.all(np.isfinite(q)) or self.sequence < 0
            or not np.isfinite(self.reprojection_rmse_px) or self.reprojection_rmse_px < 0
        ):
            raise ValueError("invalid observation")
        object.__setattr__(self, "joint_positions_rad", q)
        object.__setattr__(self, "T_base_ee", as_transform(self.T_base_ee, name="T_base_ee"))
        object.__setattr__(self, "T_camera_target", as_transform(self.T_camera_target, name="T_camera_target"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "captured_at_utc": self.captured_at_utc,
            "robot_timestamp_s": self.robot_timestamp_s,
            "camera_timestamp_s": self.camera_timestamp_s,
            "joint_positions_rad": self.joint_positions_rad.tolist(),
            "T_base_ee": self.T_base_ee.tolist(),
            "T_camera_target": self.T_camera_target.tolist(),
            "reprojection_rmse_px": self.reprojection_rmse_px,
            "color_image_path": self.color_image_path,
            "debug_image_path": self.debug_image_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Observation":
        return cls(**data)


@dataclass(frozen=True)
class CalibrationResult:
    T_base_camera: NDArray[np.float64]
    method: str
    observation_ids: tuple[str, ...]
    solved_at_utc: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "T_base_camera", as_transform(self.T_base_camera, name="T_base_camera"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "T_base_camera": self.T_base_camera.tolist(),
            "method": self.method,
            "observation_ids": list(self.observation_ids),
            "solved_at_utc": self.solved_at_utc,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationResult":
        copy = dict(data)
        copy["observation_ids"] = tuple(copy["observation_ids"])
        return cls(**copy)


@dataclass(frozen=True)
class ValidationReport:
    status: str
    metrics: dict[str, float]
    warnings: tuple[str, ...] = ()
    outlier_observation_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "metrics": self.metrics,
            "warnings": list(self.warnings),
            "outlier_observation_ids": list(self.outlier_observation_ids),
        }
