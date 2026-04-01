"""
Franka 机械臂手眼标定程序 - 数据采集部分
基于 cps_hand_eye_calibration.py 改造，使用 Franka 替代 Hans 机器人

使用方式：
1. 运行脚本，连接 Franka 机器人和 Orbbec 相机
2. 自动执行轨迹运动并采集标定数据
3. 数据保存为 YAML 文件，供后续计算使用
"""

import os
import time
from pathlib import Path
import yaml
import numpy as np
import cv2

# 导入 Franka 机器人封装
from frank_robot_wrapper import FrankRobotWrapper

# 导入 Orbbec 相机（同一目录）
from orbbec_camera_v6 import OrbbecCamera

timestamp = time.strftime("%Y%m%d_%H%M%S")

# ======================== 全局可配置项 ========================
# Franka 配置
FRANKA_IP = "172.16.0.2"

# 文件路径
TRAJECTORY_FILE = './trajectory/trajectory_joints_20260331_194232.npy'
CALIB_DATA_DIR = './hand_eye_calibration/calib_data'
CAPTURE_DIR = './hand_eye_calibration/captures'

# 标定参数
SPEED = 15.0
WAIT_TIME = 1.0
PATTERN_SIZE = (11, 8)
SQUARE_LEN = 0.01  # 棋盘格方块尺寸 10mm
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
            # 异步运动，不等待完成
            robot.move_j_async(joint_deg.tolist())
        except Exception as e:
            print(f"移动报错: {e}，跳过此点")
            robot.recover()
            continue

        # 等待运动完成
        try:
            robot.join_motion()
        except Exception as e:
            print(f"运动被中止: {e}，跳过此点")
            robot.recover()
            continue

        # 稳定等待时间
        time.sleep(WAIT_TIME)

        try:
            tcp_matrix = robot.get_tcp_pose_matrix()
        except Exception as e:
            print(f"获取TCP位姿失败: {e}，跳过")
            robot.recover()
            continue

        # 获取图像
        try:
            color_frame, _ = camera.get_frames()
            img, _ = camera.get_images_from_frames(color_frame, None)
        except Exception as e:
            print(f"图像获取失败: {e}，跳过此点")
            continue

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
        print(f"数据已保存至: {save_path}")
    else:
        print("\n采集失败，没有收集到有效数据。")


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
            color_width=2560,
            color_height=1440,
            color_fps=30,
            enable_depth=False,
            enable_alignment=False
        )
        time.sleep(2)

        # 询问采集次数
        loop_count = int(input("请输入采集次数（建议3次）: ") or "3")

        # 执行标定数据采集
        for loop_idx in range(1, loop_count + 1):
            try:
                run_calibration_collection(robot, camera, loop_index=loop_idx)
                time.sleep(1)
            except Exception as e:
                print(f"第 {loop_idx} 次任务执行出现严重错误: {e}")

        print("\n========== 数据采集完成 ==========")
        print(f"请运行 franky_calibration_compute.py 计算标定结果")

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
