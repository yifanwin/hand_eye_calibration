"""
Franka 机械臂手眼标定程序
基于 cps_hand_eye_calibration.py 改造，使用 Franka 替代 Hans 机器人
"""

import os
import sys
import time
from pathlib import Path
import math
import yaml
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# 导入 Franka 机器人封装
from frank_robot_wrapper import FrankRobotWrapper

# 导入 Orbbec 相机（同一目录）
from orbbec_camera_v6 import OrbbecCamera

timestamp = time.strftime("%Y%m%d_%H%M%S")

# ======================== 全局可配置项 ========================
# Franka 配置
FRANKA_IP = "10.90.90.1"

# 文件路径
TRAJECTORY_FILE = 'hand_eye_calibration/trajectory/trajectory_joints.npy'
CALIB_DATA_DIR = 'hand_eye_calibration/calib_data'
CAPTURE_DIR = 'hand_eye_calibration/captures'

# 标定参数
LOOP_COUNT = 3
SPEED = 15.0
WAIT_TIME = 1.0
PATTERN_SIZE = (11, 8)
SQUARE_LEN = 0.01  # 棋盘格方块尺寸 10mm

# 输出路径
OUTPUT_JSON = '/home/fanfan/proj/oea/oea-rekep-real-plugin/runtime/real_calibration/orbbec_config/orbbec_calibration.json'
OUTPUT_NPY = f'hand_eye_calibration/hand_eye_calib_output/T_cam2base_{timestamp}.npy'
CALIB_DATA_PATTERN = f"h_e_calib_data_{timestamp}_*.yaml"
# =============================================================


def detect_chessboard_pose(img, pattern_size, square_len, K, D):
    """
    检测棋盘格并求解位姿

    Args:
        img: BGR 图像
        pattern_size: 棋盘格内角点数量 (cols, rows)
        square_len: 方块尺寸 [米]
        K: 相机内参矩阵
        D: 畸变系数

    Returns:
        success: 是否成功
        rvec: 旋转向量
        tvec: 平移向量
        vis_img: 可视化图像
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ret:
        return False, None, None, img

    # 亚像素精度优化
    corners2 = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    # 构建 3D 标定板坐标
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_len

    # PnP 解算
    success, rvec, tvec = cv2.solvePnP(
        objp, corners2, K, D, flags=cv2.SOLVEPNP_ITERATIVE
    )

    # 可视化
    vis = img.copy()
    cv2.drawChessboardCorners(vis, pattern_size, corners2, ret)
    if success:
        cv2.drawFrameAxes(vis, K, D, rvec, tvec, 0.1)

    return success, rvec, tvec, vis


def matrix_to_list(matrix: np.ndarray) -> list:
    """将矩阵转换为列表（兼容 YAML 存储）"""
    return matrix.tolist()


def run_calibration_collection(robot, camera, loop_index=1):
    """
    执行标定数据采集

    Args:
        robot: 已连接的机器人实例
        camera: 已初始化的相机实例
        loop_index: 循环索引
    """
    if not os.path.exists(TRAJECTORY_FILE):
        print(f"错误: 找不到轨迹文件 {TRAJECTORY_FILE}")
        return

    print(f"\n========== 第 {loop_index} 次执行 ==========")
    target_joints_array_deg = np.load(TRAJECTORY_FILE)
    print(f"共加载 {len(target_joints_array_deg)} 个点位")

    # 获取相机内参
    K = camera.get_color_intrinsics_matrix()
    D = camera.get_color_distortion_coeffs()

    collected_data = {
        "robot_pose": [],
        "target_rvecs": [],
        "target_tvecs": [],
        "camera_matrix": K.tolist(),
        "dist_coeffs": D.tolist()
    }

    save_file = f"{CALIB_DATA_DIR}/h_e_calib_data_{timestamp}_{loop_index}.yaml"
    save_path = Path(save_file)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    success_count = 0
    capture_dir = Path(CAPTURE_DIR)
    capture_dir.mkdir(parents=True, exist_ok=True)

    for i, joint_deg in enumerate(target_joints_array_deg):
        print(f"\n--- 执行点位 {i+1}/{len(target_joints_array_deg)} ---")

        try:
            robot.move_j(joint_deg.tolist(), vel=SPEED)
        except Exception as e:
            print(f"移动报错: {e}，跳过此点")
            continue

        try:
            tcp_matrix = robot.get_tcp_pose_matrix()
        except Exception as e:
            print(f"获取TCP位姿失败: {e}，跳过")
            continue

        # 获取图像
        color_frame, _ = camera.get_frames()
        img, _ = camera.get_images_from_frames(color_frame, None)

        if img is None:
            print("图像获取失败")
            continue

        # 棋盘格检测
        ret, rvec, tvec, vis_img = detect_chessboard_pose(img, PATTERN_SIZE, SQUARE_LEN, K, D)

        # 保存可视化图像
        filename = capture_dir / f"{loop_index}_{i+1:03d}_{timestamp}.png"
        cv2.imwrite(str(filename), vis_img)

        if ret:
            print("棋盘格检测成功")
            collected_data["robot_pose"].append(matrix_to_list(tcp_matrix))
            collected_data["target_rvecs"].append(rvec.reshape(3).tolist())
            collected_data["target_tvecs"].append(tvec.reshape(3).tolist())
            success_count += 1
        else:
            print("未检测到棋盘格")
        time.sleep(WAIT_TIME)

    # 保存数据
    if success_count > 0:
        with open(save_path, 'w') as f:
            yaml.dump(collected_data, f)
        print(f"\n第 {loop_index} 次采集完成！有效数据: {success_count} 组。")
    else:
        print("\n采集失败，没有收集到有效数据。")


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
        raise FileNotFoundError(f"未找到符合模式 {data_pattern} 的数据文件")

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


def save_calibration_json(T_base_camera, camera, output_path):
    """
    保存完整标定结果为 JSON 文件

    Args:
        T_base_camera: 相机外参 4x4 矩阵
        camera: OrbbecCamera 实例
        output_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 获取相机内参
    color_intrinsics = camera.get_color_intrinsics()
    depth_intrinsics = camera.get_depth_intrinsics()
    K_color = camera.get_color_intrinsics_matrix()
    D_color = camera.get_color_distortion_coeffs()
    K_depth = camera.get_depth_intrinsics_matrix()
    D_depth = camera.get_depth_distortion_coeffs()

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
            "width": color_intrinsics.get('width', 1280),
            "height": color_intrinsics.get('height', 720),
            "fx": float(K_color[0, 0]),
            "fy": float(K_color[1, 1]),
            "cx": float(K_color[0, 2]),
            "cy": float(K_color[1, 2]),
            "model": "distortion.brown_conrady",
            "coeffs": D_color.tolist() if hasattr(D_color, 'tolist') else list(D_color),
            "distortion_model": "distortion.brown_conrady"
        },
        "depth_intrinsics": {
            "width": depth_intrinsics.get('width', 640),
            "height": depth_intrinsics.get('height', 576),
            "fx": float(K_depth[0, 0]),
            "fy": float(K_depth[1, 1]),
            "cx": float(K_depth[0, 2]),
            "cy": float(K_depth[1, 2]),
            "model": "distortion.brown_conrady",
            "coeffs": D_depth.tolist() if hasattr(D_depth, 'tolist') else list(D_depth),
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
                "height": 576,
                "format": "Y16",
                "fps": 30
            }
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(calib_json, f, indent=2, ensure_ascii=False)

    print(f"完整标定结果已保存至: {output_path}")


def main():
    robot = None
    camera = None

    try:
        # 初始化 Franka 机器人
        print("初始化 Franka 机械臂...")
        robot = FrankRobotWrapper(ip=FRANKA_IP)
        robot.connect()

        # 初始化 Orbbec 相机
        print("初始化 Orbbec 相机...")
        camera = OrbbecCamera(
            color_width=1280,
            color_height=720,
            color_fps=30,
            enable_depth=False,
            enable_alignment=False
        )
        time.sleep(2)

        # 执行标定数据采集
        for loop_idx in range(1, LOOP_COUNT + 1):
            try:
                run_calibration_collection(robot, camera, loop_index=loop_idx)
                time.sleep(1)
            except Exception as e:
                print(f"第 {loop_idx} 次任务执行出现严重错误: {e}")

        # 计算标定结果
        print("\n========== 计算标定结果 ==========")
        T_base_camera = compute_calibration(CALIB_DATA_DIR, CALIB_DATA_PATTERN)

        # 保存 NPY
        output_npy_dir = Path(OUTPUT_NPY).parent
        output_npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(OUTPUT_NPY, T_base_camera)
        print(f"NPY 结果已保存至: {OUTPUT_NPY}")

        # 保存完整 JSON
        save_calibration_json(T_base_camera, camera, OUTPUT_JSON)

    except Exception as e:
        print(f"全局初始化或运行时错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n正在断开连接...")
        if robot:
            robot.disconnect()
        if camera:
            camera.stop()
        print("程序结束。")


if __name__ == "__main__":
    main()
