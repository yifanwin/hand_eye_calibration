"""
Franka 机械臂手眼标定程序 - 计算标定结果部分
基于 cps_hand_eye_calibration.py 改造，使用 Franka 替代 Hans 机器人

使用方式：
1. 运行脚本前，确保已完成数据采集（franky_calibration_capture.py）
2. 自动读取 YAML 数据文件，计算手眼标定外参
3. 输出标定结果为 JSON 和 NPY 格式
"""

import os
import sys
import time
from pathlib import Path
import yaml
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

timestamp = time.strftime("%Y%m%d_%H%M%S")

# ======================== 全局可配置项 ========================
# 文件路径
CALIB_DATA_DIR = './hand_eye_calibration/calib_data'
CALIB_DATA_PATTERN = "h_e_calib_data_20260331_202337_*.yaml"

# 输出路径
OUTPUT_JSON = '../oea-rekep-real-plugin/runtime/real_calibration/orbbec_config/orbbec_calibration.json'
OUTPUT_NPY = f'./hand_eye_calib_output/T_cam2base_{timestamp}.npy'
# =============================================================


def average_rotations(rotation_matrices):
    """计算旋转矩阵的平均值（通过四元数方法）"""
    quats = []
    for R_mat in rotation_matrices:
        r = R.from_matrix(R_mat)
        q = r.as_quat()
        quats.append(q)

    quats = np.array(quats)

    # 统一四元数方向（使 w 分量为正）
    for i in range(len(quats)):
        if quats[i][3] < 0:
            quats[i] = -quats[i]

    # 计算平均四元数并归一化
    mean_quat = np.mean(quats, axis=0)
    mean_quat /= np.linalg.norm(mean_quat)

    return R.from_quat(mean_quat).as_matrix()


def average_translations(translation_vectors):
    """计算位移向量的算术平均值"""
    return np.mean(translation_vectors, axis=0)


def compute_calibration(calib_data_dir, data_pattern):
    """
    计算手眼标定外参

    Args:
        calib_data_dir: 标定数据目录
        data_pattern: 数据文件匹配模式

    Returns:
        T_base_camera: 4x4 外参矩阵
    """
    import glob

    calib_data_files = sorted(glob.glob(f"{calib_data_dir}/{data_pattern}"))
    if not calib_data_files:
        raise FileNotFoundError(f"未找到符合模式 {data_pattern} 的数据文件\n请先运行 franky_calibration_capture.py 进行数据采集")

    print(f"找到 {len(calib_data_files)} 个数据文件:")
    for f in calib_data_files:
        print(f"  - {f}")

    rotation_matrices = []
    translation_vectors = []

    for yaml_path in calib_data_files:
        print(f"\n--- 处理: {Path(yaml_path).name} ---")
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            rvecs = np.array(data['target_rvecs'])
            tvecs = np.array(data['target_tvecs'])
            robot_xyz = np.array(data['robot_pose'])

            # 构建手眼标定输入
            R_gripper2base = []
            t_gripper2base = []
            R_target2cam = []
            t_target2cam = []

            for i in range(len(rvecs)):
                M_ee2base = robot_xyz[i]
                M_base2ee = np.linalg.inv(M_ee2base)

                R_gripper2base.append(M_base2ee[:3, :3])
                t_gripper2base.append(M_base2ee[:3, 3])

                temp_cam = R.from_rotvec(rvecs[i])
                R_target2cam.append(temp_cam.as_matrix())
                t_target2cam.append(tvecs[i])

            R_gripper2base = np.array(R_gripper2base)
            t_gripper2base = np.array(t_gripper2base)
            R_target2cam = np.array(R_target2cam)
            t_target2cam = np.array(t_target2cam)

            # OpenCV 手眼标定
            rotation_matrix, translation_vector = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                method=cv2.CALIB_HAND_EYE_PARK
            )
            print(f"旋转矩阵:\n{rotation_matrix}")
            print(f"平移向量: {translation_vector.flatten()}")

            rotation_matrices.append(rotation_matrix)
            translation_vectors.append(translation_vector.flatten())

        except Exception as e:
            print(f"处理失败: {e}")
            continue

    if not rotation_matrices:
        raise RuntimeError("没有成功处理任何数据文件")

    # 计算平均值
    mean_rotation = average_rotations(rotation_matrices)
    mean_translation = average_translations(translation_vectors).reshape(3, 1)

    # 构建 4x4 变换矩阵
    T_base_camera = np.eye(4, dtype=np.float64)
    T_base_camera[:3, :3] = mean_rotation
    T_base_camera[:3, 3] = mean_translation.ravel()

    print(f"\n平均结果 (基于 {len(rotation_matrices)} 个数据文件):")
    print(f"旋转矩阵:\n{mean_rotation}")
    print(f"平移向量: {mean_translation.ravel()}")

    return T_base_camera


def save_calibration_json(T_base_camera, calib_data_dir, data_pattern, output_path):
    """
    保存完整标定结果为 JSON 文件

    Args:
        T_base_camera: 相机外参 4x4 矩阵
        calib_data_dir: 标定数据目录
        data_pattern: 数据文件匹配模式
        output_path: 输出文件路径
    """
    import glob
    import json

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 从数据文件读取内参
    calib_data_files = sorted(glob.glob(f"{calib_data_dir}/{data_pattern}"))
    if calib_data_files:
        with open(calib_data_files[0], "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        K_color = np.array(data['camera_matrix'])
        D_color = np.array(data['dist_coeffs'])
    else:
        # 默认内参
        K_color = np.array([[606.0, 0, 320.0], [0, 606.0, 240.0], [0, 0, 1]], dtype=np.float64)
        D_color = np.zeros(5, dtype=np.float64)

    # 估算深度内参（缩放）
    depth_scale = 2.0  # 深度分辨率通常是彩色的一半
    K_depth = K_color.copy()
    K_depth[0, 0] /= depth_scale
    K_depth[1, 1] /= depth_scale
    K_depth[0, 2] /= depth_scale
    K_depth[1, 2] /= depth_scale

    calib_json = {
        "device_info": {
            "serial_number": "",
            "name": "Orbbec Camera",
            "firmware_version": "",
            "product_line": "Orbbec",
            "device_id": ""
        },
        "timestamp": timestamp,
        "color_intrinsics": {
            "width": 1280,
            "height": 720,
            "fx": float(K_color[0, 0]),
            "fy": float(K_color[1, 1]),
            "cx": float(K_color[0, 2]),
            "cy": float(K_color[1, 2]),
            "model": "distortion.brown_conrady",
            "coeffs": D_color.tolist() if hasattr(D_color, 'tolist') else list(D_color),
            "distortion_model": "distortion.brown_conrady"
        },
        "depth_intrinsics": {
            "width": 640,
            "height": 360,
            "fx": float(K_depth[0, 0]),
            "fy": float(K_depth[1, 1]),
            "cx": float(K_depth[0, 2]),
            "cy": float(K_depth[1, 2]),
            "model": "distortion.brown_conrady",
            "coeffs": D_color.tolist() if hasattr(D_color, 'tolist') else list(D_color),
            "distortion_model": "distortion.brown_conrady"
        },
        "depth_scale": 0.001,
        "extrinsics": {
            "rotation": T_base_camera[:3, :3].flatten().tolist(),
            "translation": T_base_camera[:3, 3].tolist(),
            "transform_matrix": T_base_camera.tolist()
        },
        "stream_config": {
            "color": {
                "width": 1280,
                "height": 720,
                "format": "MJPG",
                "fps": 30
            },
            "depth": {
                "width": 640,
                "height": 360,
                "format": "Y16",
                "fps": 30
            }
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(calib_json, f, indent=2, ensure_ascii=False)

    print(f"完整标定结果已保存至: {output_path}")


def main():
    try:
        print("========== 计算手眼标定结果 ==========")

        # 计算标定结果
        T_base_camera = compute_calibration(CALIB_DATA_DIR, CALIB_DATA_PATTERN)

        # 保存 NPY
        output_npy_dir = Path(OUTPUT_NPY).parent
        output_npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(OUTPUT_NPY, T_base_camera)
        print(f"NPY 结果已保存至: {OUTPUT_NPY}")

        # 保存完整 JSON
        save_calibration_json(T_base_camera, CALIB_DATA_DIR, CALIB_DATA_PATTERN, OUTPUT_JSON)

        print("\n========== 标定完成 ==========")

    except Exception as e:
        print(f"计算标定结果时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
