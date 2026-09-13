from __future__ import annotations

import time

import numpy as np

from ...interfaces import AdapterError, CameraAdapter
from ...models import (
    CameraCalibration, CameraFrame, ColorProjection, DeviceInfo,
    IntrinsicCalibration, StreamProfile,
)


class RealSenseAdapter(CameraAdapter):
    def __init__(
        self,
        serial_number: str = "",
        color_width: int = 1280,
        color_height: int = 720,
        color_fps: int = 30,
        color_format: str = "bgr8",
        depth_width: int = 640,
        depth_height: int = 480,
        depth_fps: int = 30,
        depth_format: str = "z16",
        timeout_ms: int = 5000,
        expected_name_contains: str = "",
        streams: dict | None = None,
    ):
        if streams:
            color = streams.get("color", {})
            depth = streams.get("depth", {})
            color_width = int(color["width"])
            color_height = int(color["height"])
            color_fps = int(color["fps"])
            color_format = str(color["format"])
            depth_width = int(depth["width"])
            depth_height = int(depth["height"])
            depth_fps = int(depth["fps"])
            depth_format = str(depth["format"])
        self.serial_number = serial_number
        self.color_spec = (color_width, color_height, color_fps, color_format)
        self.depth_spec = (depth_width, depth_height, depth_fps, depth_format)
        self.timeout_ms = timeout_ms
        self.expected_name_contains = expected_name_contains
        self._rs = self._pipeline = self._profile = None
        self._calibration = None
        self._projection = None

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise AdapterError("RealSense adapter requires pyrealsense2") from exc
        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number:
            config.enable_device(self.serial_number)
        try:
            cw, ch, cfps, cformat = self.color_spec
            dw, dh, dfps, dformat = self.depth_spec
            config.enable_stream(rs.stream.color, cw, ch, getattr(rs.format, cformat), cfps)
            config.enable_stream(rs.stream.depth, dw, dh, getattr(rs.format, dformat), dfps)
            profile = pipeline.start(config)
            color_profile = profile.get_stream(rs.stream.color)
            depth_profile = profile.get_stream(rs.stream.depth)
            color_video = color_profile.as_video_stream_profile()
            depth_video = depth_profile.as_video_stream_profile()
            color_intr = color_video.get_intrinsics()
            depth_intr = depth_video.get_intrinsics()
            device = profile.get_device()
            depth_scale = float(device.first_depth_sensor().get_depth_scale())
            extrinsics = depth_profile.get_extrinsics_to(color_profile)
            T_color_depth = self._extrinsics_to_transform(extrinsics)
            device_name = self._device_info(device, rs.camera_info.name)
            if self.expected_name_contains and self.expected_name_contains.lower() not in device_name.lower():
                raise AdapterError(
                    f"connected RealSense '{device_name}' does not match expected '{self.expected_name_contains}'"
                )
            info = DeviceInfo(
                adapter=f"{type(self).__module__}:{type(self).__name__}",
                name=device_name,
                serial_number=self._device_info(device, rs.camera_info.serial_number),
                firmware_version=self._device_info(device, rs.camera_info.firmware_version),
                metadata={"product_line": self._device_info(device, rs.camera_info.product_line)},
            )
            color = self._intrinsics(color_intr)
            depth = self._intrinsics(depth_intr)
            if (color.width, color.height) != (cw, ch) or (depth.width, depth.height) != (dw, dh):
                raise AdapterError("RealSense active calibration dimensions do not match requested profiles")
            self._calibration = CameraCalibration(
                device=info,
                color_profile=StreamProfile(cw, ch, cfps, cformat),
                depth_profile=StreamProfile(dw, dh, dfps, dformat),
                color_intrinsics=color,
                depth_intrinsics=depth,
                depth_scale_m_per_unit=depth_scale,
                T_color_depth=T_color_depth,
            )
            coeffs = np.asarray(color.distortion_coeffs, dtype=np.float64)
            model = color.distortion_model.lower().replace("distortion.", "")
            if np.allclose(coeffs, 0.0):
                detector_model, detector_coeffs = "none", np.zeros_like(coeffs)
            elif model == "brown_conrady":
                detector_model, detector_coeffs = "opencv_brown_conrady", coeffs
            else:
                raise AdapterError(
                    f"color distortion model '{color.distortion_model}' has non-zero coefficients and "
                    "cannot be losslessly passed to OpenCV solvePnP in v1"
                )
            self._projection = ColorProjection(color.camera_matrix, detector_coeffs, detector_model)
        except Exception:
            try:
                pipeline.stop()
            except Exception:
                pass
            self._calibration = self._projection = None
            raise
        self._rs, self._pipeline, self._profile = rs, pipeline, profile

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = self._profile = None
        self._calibration = self._projection = None

    def get_calibration(self) -> CameraCalibration:
        if self._calibration is None:
            raise AdapterError("RealSense camera is not started")
        return self._calibration

    def capture(self) -> CameraFrame:
        if self._pipeline is None:
            raise AdapterError("RealSense camera is not started")
        frames = self._pipeline.wait_for_frames(self.timeout_ms)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise AdapterError("RealSense frameset is missing color or depth")
        color_bgr = np.asanyarray(color.get_data())
        depth_raw = np.asanyarray(depth.get_data())
        return CameraFrame(
            color_bgr=color_bgr,
            depth_raw=depth_raw,
            color_projection=self._projection,
            host_timestamp_s=time.time(),
            device_timestamp_s=float(color.get_timestamp()) / 1000.0,
        )

    @staticmethod
    def _intrinsics(value) -> IntrinsicCalibration:
        return IntrinsicCalibration(
            width=int(value.width), height=int(value.height), fx=float(value.fx), fy=float(value.fy),
            cx=float(value.ppx), cy=float(value.ppy), distortion_model=str(value.model),
            distortion_coeffs=tuple(float(item) for item in value.coeffs),
        )

    @staticmethod
    def _device_info(device, key) -> str:
        return str(device.get_info(key)) if device.supports(key) else ""

    @staticmethod
    def _extrinsics_to_transform(extrinsics) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(extrinsics.rotation, dtype=np.float64).reshape(3, 3)
        # librealsense extrinsic translation is already expressed in metres.
        transform[:3, 3] = np.asarray(extrinsics.translation, dtype=np.float64)
        return transform
