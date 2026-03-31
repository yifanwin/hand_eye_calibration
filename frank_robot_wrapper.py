"""
Franka 机械臂封装类 for 手眼标定
使用 franky 库通过 FCI (Franka Control Interface) 连接控制
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from franky import Robot, JointMotion


class FrankRobotWrapper:
    """Franka 机械臂封装类，兼容原有标定程序的接口设计"""

    def __init__(self, ip: str = "10.90.90.1"):
        """
        初始化 Franka 机器人连接

        Args:
            ip: 机器人 IP 地址 (FCI 默认 10.90.90.1)
        """
        self.ip = ip
        self.robot = Robot(ip)
        # 初始设置为较低速度，确保标定过程安全
        self.robot.relative_dynamics_factor = 0.05

    def connect(self):
        """建立连接并恢复错误状态"""
        print(f"正在连接 Franka Robot @ {self.ip}...")
        self.robot.recover_from_errors()
        print("Franka Robot 已连接")

    def disconnect(self):
        """断开连接（franky 无需显式断开）"""
        pass

    def move_j(self, target_joints_deg, vel=20, acc=100, radius=0, wait=True):
        """
        关节空间运动（兼容原有接口）

        Args:
            target_joints_deg: 目标关节角度 [度]
            vel: 速度百分比（franky 中通过 relative_dynamics_factor 影响）
            acc: 加速度百分比（未使用）
            radius: 圆滑半径（未使用）
            wait: 是否阻塞等待运动完成
        """
        # 将角度转换为弧度
        target_joints_rad = np.deg2rad(target_joints_deg)

        # 创建关节运动指令
        motion = JointMotion(target_joints_rad.tolist())

        if wait:
            self.robot.move(motion)
        else:
            self.robot.move(motion, asynchronous=True)

    def get_tcp_pose_matrix(self) -> np.ndarray:
        """
        获取当前 TCP (末端执行器) 在基座坐标系下的 4x4 变换矩阵

        Returns:
            4x4 齐次变换矩阵 (base_to_ee)
        """
        # 获取当前笛卡尔状态
        cartesian_state = self.robot.current_cartesian_state
        ee_pose = cartesian_state.pose.end_effector_pose

        # Affine 对象包含 .translation [x,y,z] (米) 和 .rotation (四元数 [x,y,z,w])
        position = ee_pose.translation  # [x, y, z] in meters
        quaternion = ee_pose.rotation    # [x, y, z, w]

        # 四元数转旋转矩阵
        rotation = R.from_quat(quaternion)
        rmat = rotation.as_matrix()

        # 构建 4x4 变换矩阵
        T = np.eye(4)
        T[:3, :3] = rmat
        T[:3, 3] = position

        return T

    def get_joint_positions(self) -> np.ndarray:
        """
        获取当前关节位置

        Returns:
            关节角度 [弧度]
        """
        joint_state = self.robot.current_joint_state
        return joint_state.position


if __name__ == "__main__":
    # 简单测试
    robot = FrankRobotWrapper("10.90.90.1")
    robot.connect()

    print("获取当前 TCP 位姿...")
    T = robot.get_tcp_pose_matrix()
    print("TCP Pose (4x4):")
    print(T)

    print("\n获取当前关节角度...")
    joints = robot.get_joint_positions()
    print(f"关节角度: {np.rad2deg(joints)} 度")

    robot.disconnect()
    print("测试完成")
