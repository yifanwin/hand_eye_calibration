"""
Franka 机械臂轨迹采集程序（示教模式）
用于手眼标定的轨迹示教，采集关节角度和笛卡尔位姿

使用方式：
1. 运行脚本，连接 Franka 机器人
2. 将机器人切换到自由拖动模式（Free-Drive）
3. 拖动机器人到目标位置，按 Enter 记录点
4. 输入 q 完成采集并保存
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from franky import Robot

# --- Franka 连接配置 ---
FRANKA_IP = "10.90.90.1"

# 时间戳
timestamp = time.strftime("%Y%m%d_%H%M%S")

# 输出路径
TRA_JOINTS_DIR = './hand_eye_calibration/trajectory'
tra_joints_path = f'{TRA_JOINTS_DIR}/trajectory_joints_{timestamp}.npy'
tra_car_path = f'{TRA_JOINTS_DIR}/trajectory_cartesian_{timestamp}.npy'

# 存储点
joint_points_list = []
cartesian_points_list = []

print("=" * 40)
print("Franka 机械臂轨迹采集程序")
print("=" * 40)

# 连接机器人
print(f"\n正在连接 Franka Robot @ {FRANKA_IP}...")
robot = Robot(FRANKA_IP)
robot.relative_dynamics_factor = 0.05
print("连接成功！")

try:
    print("\n" + "=" * 40)
    input("请确保机器人已使能且处于 Free-Drive 模式，按 [Enter] 键开始示教...")

    print("\n开始示教采集...")
    print("拖动机器人到目标位置，按 [Enter] 记录点 | 输入 [q] 完成并保存")
    print("-" * 40)

    while True:
        key_input = input(f"已采集 {len(joint_points_list)} 个点。操作: ")

        if key_input.lower() == 'q':
            break

        # 获取当前关节角度
        joint_state = robot.current_joint_state
        current_joints_deg = np.rad2deg(joint_state.position)  # 弧度 -> 度

        # 获取当前笛卡尔位姿
        cartesian_state = robot.current_cartesian_state
        ee_pose = cartesian_state.pose.end_effector_pose

        # Affine -> 位置 [m] + 四元数 [x,y,z,w]
        pos_m = ee_pose.translation  # 米
        quat = ee_pose.rotation      # [x,y,z,w]

        # 转换为 [x,y,z,rx,ry,rz] 毫米/度
        pos_mm = pos_m * 1000  # 米转毫米
        r = R.from_quat(quat)
        euler_deg = r.as_euler('xyz', degrees=True)  # 度

        cartesian_pose = np.concatenate([pos_mm, euler_deg])

        # 存储
        joint_points_list.append(current_joints_deg)
        cartesian_points_list.append(cartesian_pose)

        print(f"  点 {len(joint_points_list)}: "
              f"Joints=[{', '.join([f'{j:.1f}' for j in current_joints_deg])}] | "
              f"XYZ=[{pos_mm[0]:.1f}, {pos_mm[1]:.1f}, {pos_mm[2]:.1f}] mm | "
              f"RPY=[{euler_deg[0]:.1f}, {euler_deg[1]:.1f}, {euler_deg[2]:.1f}] deg")

    print("\n采集结束。")

except Exception as e:
    print(f"\n发生错误: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\n正在断开连接...")

# 保存
if len(joint_points_list) > 0:
    np_joints = np.array(joint_points_list)
    np_cartesian = np.array(cartesian_points_list)

    # 确保目录存在
    import os
    os.makedirs(TRA_JOINTS_DIR, exist_ok=True)

    np.save(tra_joints_path, np_joints)
    np.save(tra_car_path, np_cartesian)

    print("\n" + "=" * 40)
    print("轨迹保存成功！")
    print(f"  关节数据: {tra_joints_path}")
    print(f"  笛卡尔数据: {tra_car_path}")
    print(f"  共 {len(joint_points_list)} 个点")
    print("=" * 40)
else:
    print("\n未采集任何点，未保存文件。")

print("\n轨迹采集脚本执行完毕。")
