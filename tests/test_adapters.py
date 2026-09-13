import sys
from types import SimpleNamespace

import numpy as np

from hand_eye_calibration.adapters.cameras.orbbec import OrbbecAdapter
from hand_eye_calibration.adapters.cameras.realsense import RealSenseAdapter
from hand_eye_calibration.adapters.robots.hans_cps import HansCPSAdapter


def test_camera_extrinsic_units_are_normalized_to_metres():
    real = SimpleNamespace(rotation=np.eye(3).reshape(-1), translation=[0.01, 0.02, 0.03])
    orbbec = SimpleNamespace(rot=np.eye(3).reshape(-1), transform=[10.0, 20.0, 30.0])
    np.testing.assert_allclose(RealSenseAdapter._extrinsics_to_transform(real)[:3, 3], [0.01, 0.02, 0.03])
    np.testing.assert_allclose(OrbbecAdapter._extrinsics_to_transform(orbbec)[:3, 3], [0.01, 0.02, 0.03])


def test_realsense_reads_factory_calibration_without_estimation(monkeypatch):
    class Intrinsics:
        width, height, fx, fy, ppx, ppy = 640, 480, 501.0, 502.0, 319.0, 241.0
        model, coeffs = "distortion.inverse_brown_conrady", [0.0] * 5

    class VideoProfile:
        def get_intrinsics(self): return Intrinsics()

    class Profile:
        def as_video_stream_profile(self): return VideoProfile()
        def get_extrinsics_to(self, _):
            return SimpleNamespace(rotation=np.eye(3).reshape(-1), translation=[0.02, 0.0, 0.0])

    class Device:
        values = {"name": "Intel RealSense D435", "serial": "123", "firmware": "5.0", "product": "D400"}
        def first_depth_sensor(self): return SimpleNamespace(get_depth_scale=lambda: 0.00025)
        def supports(self, _): return True
        def get_info(self, key): return self.values[key]

    class PipelineProfile:
        def get_stream(self, _): return Profile()
        def get_device(self): return Device()

    class Pipeline:
        def start(self, _): return PipelineProfile()
        def stop(self): pass

    class Config:
        def enable_device(self, _): pass
        def enable_stream(self, *_): pass

    fake = SimpleNamespace(
        pipeline=Pipeline, config=Config,
        stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(bgr8="bgr8", z16="z16"),
        camera_info=SimpleNamespace(name="name", serial_number="serial", firmware_version="firmware", product_line="product"),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake)
    adapter = RealSenseAdapter(color_width=640, color_height=480, expected_name_contains="D435")
    adapter.start()
    calibration = adapter.get_calibration()
    assert calibration.color_intrinsics.fx == 501.0
    assert calibration.depth_intrinsics.fy == 502.0
    assert calibration.depth_scale_m_per_unit == 0.00025
    np.testing.assert_allclose(calibration.T_color_depth[:3, 3], [0.02, 0.0, 0.0])


def test_hans_adapter_converts_public_radians_and_sdk_millimetres(monkeypatch):
    calls = {}

    class Client:
        def HRIF_Connect(self, *_): return 0
        def HRIF_DisConnect(self, *_): return 0
        def HRIF_MoveJ(self, *args):
            calls["joints_deg"] = args[3]
            return 0
        def waitMoveDone(self, *_): return None
        def HRIF_ReadActPos(self, _box, _robot, result):
            result.extend(["0", "90", "0", "0", "0", "0", "100", "200", "300", "0", "0", "90"])
            return 0

    monkeypatch.setattr("importlib.import_module", lambda _: SimpleNamespace(CPSClient=Client))
    adapter = HansCPSAdapter("127.0.0.1", cps_module="fake.cps")
    adapter.connect()
    adapter.move_joints(np.array([0.0, np.pi / 2, 0.0, 0.0, 0.0, 0.0]))
    state = adapter.read_state()
    np.testing.assert_allclose(calls["joints_deg"], [0, 90, 0, 0, 0, 0], atol=1e-10)
    np.testing.assert_allclose(state.joint_positions_rad[1], np.pi / 2)
    np.testing.assert_allclose(state.T_base_ee[:3, 3], [0.1, 0.2, 0.3])
