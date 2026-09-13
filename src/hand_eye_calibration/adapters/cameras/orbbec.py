from __future__ import annotations

import time

import cv2
import numpy as np

from ...interfaces import AdapterError, CameraAdapter
from ...models import (
    CameraCalibration, CameraFrame, ColorProjection, DeviceInfo,
    IntrinsicCalibration, StreamProfile,
)


class OrbbecAdapter(CameraAdapter):
    def __init__(
        self,
        color_width: int = 1920,
        color_height: int = 1080,
        color_fps: int = 30,
        color_format: str = "MJPG",
        depth_width: int = 640,
        depth_height: int = 360,
        depth_fps: int = 30,
        depth_format: str = "Y16",
        timeout_ms: int = 5000,
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
        self.color_spec = (color_width, color_height, color_fps, color_format)
        self.depth_spec = (depth_width, depth_height, depth_fps, depth_format)
        self.timeout_ms = timeout_ms
        self._sdk = self._pipeline = None
        self._calibration = self._projection = None

    def start(self) -> None:
        try:
            import pyorbbecsdk as sdk
        except ImportError as exc:
            raise AdapterError("Orbbec adapter requires pyorbbecsdk v2") from exc
        pipeline, config = sdk.Pipeline(), sdk.Config()
        try:
            cw, ch, cfps, cformat = self.color_spec
            dw, dh, dfps, dformat = self.depth_spec
            color_profile = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR).get_video_stream_profile(
                cw, ch, getattr(sdk.OBFormat, cformat), cfps
            )
            depth_profile = pipeline.get_stream_profile_list(sdk.OBSensorType.DEPTH_SENSOR).get_video_stream_profile(
                dw, dh, getattr(sdk.OBFormat, dformat), dfps
            )
            if color_profile is None or depth_profile is None:
                raise AdapterError("requested Orbbec stream profile is unavailable")
            config.enable_stream(color_profile)
            config.enable_stream(depth_profile)
            pipeline.start(config)
            frames = pipeline.wait_for_frames(self.timeout_ms)
            if not frames or not frames.get_depth_frame() or not frames.get_color_frame():
                raise AdapterError("Orbbec did not return a complete initial frameset")
            camera_param = pipeline.get_camera_param()
            color_intr = self._intrinsics(camera_param.rgb_intrinsic, camera_param.rgb_distortion)
            depth_intr = self._intrinsics(camera_param.depth_intrinsic, camera_param.depth_distortion)
            if (color_intr.width, color_intr.height) != (cw, ch) or (depth_intr.width, depth_intr.height) != (dw, dh):
                raise AdapterError("Orbbec active calibration dimensions do not match requested profiles")
            external = camera_param.transform
            T_color_depth = self._extrinsics_to_transform(external)
            depth_scale_m_per_unit = float(frames.get_depth_frame().get_depth_scale()) / 1000.0
            device_info = pipeline.get_device().get_device_info()
            info = DeviceInfo(
                adapter=f"{type(self).__module__}:{type(self).__name__}",
                name=str(device_info.get_name()),
                serial_number=str(device_info.get_serial_number()),
                firmware_version=str(device_info.get_firmware_version()),
            )
            self._calibration = CameraCalibration(
                device=info,
                color_profile=StreamProfile(cw, ch, cfps, cformat),
                depth_profile=StreamProfile(dw, dh, dfps, dformat),
                color_intrinsics=color_intr,
                depth_intrinsics=depth_intr,
                depth_scale_m_per_unit=depth_scale_m_per_unit,
                T_color_depth=T_color_depth,
            )
            self._projection = ColorProjection(
                color_intr.camera_matrix,
                np.asarray(color_intr.distortion_coeffs, dtype=np.float64),
                "opencv_brown_conrady" if not np.allclose(color_intr.distortion_coeffs, 0.0) else "none",
            )
        except Exception:
            try:
                pipeline.stop()
            except Exception:
                pass
            self._calibration = self._projection = None
            raise
        self._sdk, self._pipeline = sdk, pipeline

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._calibration = self._projection = None

    def get_calibration(self) -> CameraCalibration:
        if self._calibration is None:
            raise AdapterError("Orbbec camera is not started")
        return self._calibration

    def capture(self) -> CameraFrame:
        if self._pipeline is None:
            raise AdapterError("Orbbec camera is not started")
        frames = self._pipeline.wait_for_frames(self.timeout_ms)
        if not frames:
            raise AdapterError("Orbbec frame timeout")
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise AdapterError("Orbbec frameset is missing color or depth")
        color_format = color.get_format()
        raw = np.frombuffer(color.get_data(), dtype=np.uint8)
        if color_format == self._sdk.OBFormat.MJPG:
            color_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        else:
            color_bgr = raw.reshape(color.get_height(), color.get_width(), 3)
            if color_format == self._sdk.OBFormat.RGB:
                color_bgr = cv2.cvtColor(color_bgr, cv2.COLOR_RGB2BGR)
        depth_raw = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(depth.get_height(), depth.get_width())
        return CameraFrame(
            color_bgr=color_bgr,
            depth_raw=depth_raw,
            color_projection=self._projection,
            host_timestamp_s=time.time(),
            device_timestamp_s=float(color.get_timestamp_us()) / 1_000_000.0,
        )

    @staticmethod
    def _intrinsics(intrinsic, distortion) -> IntrinsicCalibration:
        return IntrinsicCalibration(
            width=int(intrinsic.width), height=int(intrinsic.height), fx=float(intrinsic.fx), fy=float(intrinsic.fy),
            # OrbbecSDK v2 exposes the K6 Brown-Conrady coefficients but not a
            # separate model property on OBCameraDistortion.
            cx=float(intrinsic.cx), cy=float(intrinsic.cy), distortion_model="brown_conrady_k6",
            distortion_coeffs=tuple(float(getattr(distortion, name)) for name in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")),
        )

    @staticmethod
    def _extrinsics_to_transform(extrinsics) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(extrinsics.rot, dtype=np.float64).reshape(3, 3)
        # Orbbec OBExtrinsic translation is expressed in millimetres.
        transform[:3, 3] = np.asarray(extrinsics.transform, dtype=np.float64) / 1000.0
        return transform
