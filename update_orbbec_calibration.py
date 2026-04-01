# -*- coding: utf-8 -*-
"""
更新 Orbbec 相机标定文件中的内参
从相机硬件读取真实内参并覆盖到 orbbec_calibration.json
"""

import pyorbbecsdk as obs
from pyorbbecsdk import OBSensorType, OBFormat
import numpy as np
import json
import os
from datetime import datetime

# ========== 全局参数 ==========
CONFIG_PATH = "/media/hcp/disk/workspace/oea/oea-rekep-real-plugin/runtime/real_calibration/orbbec_config/orbbec_calibration.json"

COLOR_WIDTH = 1920
COLOR_HEIGHT = 1080
COLOR_FPS = 30

DEPTH_WIDTH = 640
DEPTH_HEIGHT = 360
DEPTH_FPS = 30


def read_intrinsics_from_camera():
    """从相机硬件读取内参"""
    print("正在初始化相机管线...")

    pipeline = obs.Pipeline()
    config = obs.Config()

    # 配置彩色流
    color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    try:
        color_profile = color_profile_list.get_video_stream_profile(
            COLOR_WIDTH, COLOR_HEIGHT, OBFormat.MJPG, COLOR_FPS
        )
    except Exception:
        color_profile = None

    if color_profile is None:
        try:
            color_profile = color_profile_list.get_video_stream_profile(
                COLOR_WIDTH, COLOR_HEIGHT, OBFormat.RGB, COLOR_FPS
            )
        except Exception:
            color_profile = color_profile_list.get_default_video_stream_profile()

    config.enable_stream(color_profile)

    # 配置深度流
    depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    try:
        depth_profile = depth_profile_list.get_video_stream_profile(
            DEPTH_WIDTH, DEPTH_HEIGHT, OBFormat.Y16, DEPTH_FPS
        )
    except Exception:
        depth_profile = depth_profile_list.get_default_video_stream_profile()

    if depth_profile is not None:
        config.enable_stream(depth_profile)

    # 启动管线
    pipeline.start(config)
    print("相机已启动，等待稳定...")
    import time
    time.sleep(1.5)

    # 读取内参
    color_params = color_profile.get_intrinsic()
    color_distortion = color_profile.get_distortion()

    depth_params = depth_profile.get_intrinsic() if depth_profile else None
    depth_distortion = depth_profile.get_distortion() if depth_profile else None

    pipeline.stop()

    # 构建内参字典
    color_intrinsics = {
        "width": color_params.width,
        "height": color_params.height,
        "fx": color_params.fx,
        "fy": color_params.fy,
        "cx": color_params.cx,
        "cy": color_params.cy,
        "model": "distortion.brown_conrady",
        "coeffs": [
            float(color_distortion.k1),
            float(color_distortion.k2),
            float(color_distortion.p1),
            float(color_distortion.p2),
            float(color_distortion.k3),
            float(color_distortion.k4),
            float(color_distortion.k5),
            float(color_distortion.k6)
        ],
        "distortion_model": "distortion.brown_conrady"
    }

    depth_intrinsics = {
        "width": depth_params.width,
        "height": depth_params.height,
        "fx": depth_params.fx,
        "fy": depth_params.fy,
        "cx": depth_params.cx,
        "cy": depth_params.cy,
        "model": "distortion.brown_conrady",
        "coeffs": [
            float(depth_distortion.k1),
            float(depth_distortion.k2),
            float(depth_distortion.p1),
            float(depth_distortion.p2),
            float(depth_distortion.k3),
            float(depth_distortion.k4),
            float(depth_distortion.k5),
            float(depth_distortion.k6)
        ],
        "distortion_model": "distortion.brown_conrady"
    }

    return color_intrinsics, depth_intrinsics


def update_calibration_file(color_intrinsics, depth_intrinsics):
    """更新标定JSON文件"""
    with open(CONFIG_PATH, 'r') as f:
        calib_data = json.load(f)

    # 更新内参
    calib_data["color_intrinsics"] = color_intrinsics
    calib_data["depth_intrinsics"] = depth_intrinsics
    calib_data["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 写回文件
    with open(CONFIG_PATH, 'w') as f:
        json.dump(calib_data, f, indent=2)

    print(f"标定文件已更新: {CONFIG_PATH}")


def main():
    print("=" * 50)
    print("Orbbec 相机内参更新脚本")
    print("=" * 50)

    # 1. 从相机读取内参
    color_intrinsics, depth_intrinsics = read_intrinsics_from_camera()

    # 2. 打印读取到的内参
    print("\n--- 彩色相机内参 ---")
    print(f"分辨率: {color_intrinsics['width']} x {color_intrinsics['height']}")
    print(f"fx: {color_intrinsics['fx']}")
    print(f"fy: {color_intrinsics['fy']}")
    print(f"cx: {color_intrinsics['cx']}")
    print(f"cy: {color_intrinsics['cy']}")
    print(f"畸变系数: {color_intrinsics['coeffs']}")

    print("\n--- 深度相机内参 ---")
    print(f"分辨率: {depth_intrinsics['width']} x {depth_intrinsics['height']}")
    print(f"fx: {depth_intrinsics['fx']}")
    print(f"fy: {depth_intrinsics['fy']}")
    print(f"cx: {depth_intrinsics['cx']}")
    print(f"cy: {depth_intrinsics['cy']}")
    print(f"畸变系数: {depth_intrinsics['coeffs']}")

    # 3. 更新标定文件
    update_calibration_file(color_intrinsics, depth_intrinsics)

    print("\n完成!")


if __name__ == "__main__":
    main()
