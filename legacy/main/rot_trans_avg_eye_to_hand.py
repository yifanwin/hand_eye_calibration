import numpy as np
from scipy.spatial.transform import Rotation

import cv2 
import yaml
from pathlib import Path
import glob
import time
import json

timestamp = time.strftime("%Y%m%d_%H%M%S")

# ======================== Global Variables ========================
# File paths
CALIB_DATA_DIR = "hand_eye_calibration/calib_data"
CALIB_DATA_PATTERN = "h_e_calib_data_20260305_163308_*.yaml"  # 数据文件名模式（支持多个文件）
SAVE_FILE_NPY = f"hand_eye_calibration/hand_eye_calib_output/T_cam2base_{timestamp}.npy"
SAVE_FILE_JSON = f"hand_eye_calibration/hand_eye_calib_output/T_cam2base_{timestamp}.json"

# Loop count (用于读取对应的数据文件)
LOOP_COUNT = 3

# ======================== Functions ========================

def average_rotations(rotation_matrices):
    """计算旋转矩阵的平均值（通过四元数方法）"""
    quats = []
    for R in rotation_matrices:
        # 将旋转矩阵转换为四元数
        r = Rotation.from_matrix(R)
        q = r.as_quat()  # 格式 [x, y, z, w]
        quats.append(q)
    
    quats = np.array(quats)
    
    # 统一四元数方向（使w分量为正）
    for i in range(len(quats)):
        if quats[i][3] < 0:
            quats[i] = -quats[i]
    
    # 计算平均四元数并归一化
    mean_quat = np.mean(quats, axis=0)
    mean_quat /= np.linalg.norm(mean_quat)
    
    # 将平均四元数转回旋转矩阵
    r_mean = Rotation.from_quat(mean_quat)
    return r_mean.as_matrix()

def average_translations(translation_vectors):
    """计算位移向量的算术平均值"""
    return np.mean(translation_vectors, axis=0)

def get_eye_to_hand_calibration(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rvecs = np.array(data['target_rvecs'])
    tvecs = np.array(data['target_tvecs'])
    robot_xyz = np.array(data['robot_pose'])
    print("robot_xyz shape:", robot_xyz.shape)
    print("robot_xyz:", robot_xyz[0])

    print("读取完成")


    # gripper to base transformation matrix
    R_gripper2base = []
    t_gripper2base = []
    # board to camera 
    R_target2cam = []
    t_target2cam = []
    for i in range(len(rvecs)):
        M_ee2base = robot_xyz[i]
        M_base2ee = np.linalg.inv(M_ee2base)
        # M_base2ee = M_ee2base 
        R_gripper2base.append(M_base2ee[:3,:3])
        t_gripper2base.append(M_base2ee[:3,3])
        temp_cam=Rotation.from_rotvec(rvecs[i])
        cam_matrix = temp_cam.as_matrix()

        R_target2cam.append(cam_matrix)
        t_target2cam.append(tvecs[i])

    R_gripper2base = np.array(R_gripper2base)
    t_gripper2base = np.array(t_gripper2base) 
    R_target2cam = np.array(R_target2cam)
    t_target2cam = np.array(t_target2cam)
    # transformation matrix 
    print(R_gripper2base.shape, t_gripper2base.shape, R_target2cam.shape, t_target2cam.shape)
    rotation_matrix, translation_vector = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=cv2.CALIB_HAND_EYE_PARK)   # cv2.CALIB_HAND_EYE_TSAI  cv2.CALIB_HAND_EYE_PARK
    print("rotation_matrix:",rotation_matrix, "translation_vector:",translation_vector)
    return rotation_matrix, translation_vector 

# 示例用法
if __name__ == "__main__":
    
    # 查找所有匹配模式的标定数据文件
    calib_data_files = sorted(glob.glob(f"{CALIB_DATA_DIR}/{CALIB_DATA_PATTERN}"))
    
    if not calib_data_files:
        print(f"错误: 未找到符合模式 {CALIB_DATA_PATTERN} 的数据文件在 {CALIB_DATA_DIR} 目录中")
        exit(1)
    
    print(f"找到 {len(calib_data_files)} 个数据文件:")
    for f in calib_data_files:
        print(f"  - {f}")
    
    left_or_right = "right"
    rotation_matrices = []
    translation_vectors = []
    
    # 遍历所有找到的数据文件
    for idx, yaml_path in enumerate(calib_data_files, 1):
        print(f"\n--- 处理文件 {idx}: {Path(yaml_path).name} ---")
        try:
            rotation_matrix, translation_vector = get_eye_to_hand_calibration(yaml_path)
            rotation_matrices.append(rotation_matrix)
            translation_vectors.append(translation_vector)
        except Exception as e:
            print(f"处理失败: {e}，跳过此文件")
            continue

    if not rotation_matrices:
        print("错误: 没有成功处理任何数据文件")
        exit(1)

    # 确保位移向量为3x1格式
    t_vectors = [t.flatten() for t in translation_vectors]  # 转换为1D数组

    # 计算平均值
    mean_rotation = average_rotations(rotation_matrices)
    mean_translation = average_translations(t_vectors).reshape(3, 1)  # 转回3x1格式

    print("\n" + "="*50)
    print(f"平均结果 (基于 {len(rotation_matrices)} 个数据文件):")
    print("="*50)
    print("平均旋转矩阵:")
    print(mean_rotation)
    print("\n平均位移向量:")
    print(mean_translation)

    T_base_camera = np.eye(4, dtype=np.float64)
    T_base_camera[:3, :3] = mean_rotation
    T_base_camera[:3, 3] = mean_translation.ravel() # 确保 t 是 1D
    
    # 保存为 NPY 格式
    np.save(SAVE_FILE_NPY, T_base_camera)
    print(f"\n结果已保存至 (NPY): {SAVE_FILE_NPY}")
    
    # 保存为 JSON 格式
    T_base_camera_dict = {
        "T_base_camera": T_base_camera.tolist(),
        "description": "Base to Camera transformation matrix (4x4)",
        "timestamp": timestamp,
        "num_files_processed": len(rotation_matrices)
    }
    with open(SAVE_FILE_JSON, 'w', encoding='utf-8') as f:
        json.dump(T_base_camera_dict, f, indent=2, ensure_ascii=False)
    print(f"结果已保存至 (JSON): {SAVE_FILE_JSON}")
