import os
import time
from pathlib import Path
import math
import yaml
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from utils.orbbec_camera_v6 import OrbbecCamera
from utils.CPS import CPSClient  
from utils.config import ROBOT_IP, ROBOT_PORT

timestamp = time.strftime("%Y%m%d_%H%M%S")

# ------------------ 全局可配置项 (顶端统一管理) ------------------
TRAJECTORY_FILE = 'hand_eye_calibration/trajectory/trajectory_joints.npy'
CALIB_DATA_DIR = 'hand_eye_calibration/calib_data'
CAPTURE_DIR = 'hand_eye_calibration/captures'
LOOP_COUNT = 3  
SPEED = 15.0
WAIT_TIME = 1.0
PATTERN_SIZE = (11, 8) 
SQUARE_LEN = 0.01      
# ---------------------------------------------------------------
def hans_pose_to_matrix(pose_list):
    x, y, z, rx, ry, rz = pose_list
    tvec = np.array([x, y, z]) / 1000.0
    r = R.from_euler('xyz', [rx, ry, rz], degrees=True) 
    rmat = r.as_matrix()
    T = np.eye(4)
    T[:3, :3] = rmat
    T[:3, 3] = tvec
    return T

def matrix_to_list(matrix: np.ndarray) -> list:
    return matrix.tolist()

class HansRobotWrapper:
    def __init__(self, ip, port=10003):
        self.client = CPSClient()
        self.boxID = 0
        self.rbtID = 0
        self.ip = ip
        self.port = port
        self.connect()

    def connect(self):
        print(f"Connecting to Hans Robot at {self.ip}:{self.port}...")
        ret = self.client.HRIF_Connect(self.boxID, self.ip, self.port)
        if ret != 0:
            raise ConnectionError(f"无法连接到大族机器人, 错误码: {ret}")
        print("Hans Robot Connected.")

    def disconnect(self):
        self.client.HRIF_DisConnect(self.boxID)

    def move_j(self, target_joints_deg, vel=20, acc=100, radius=0, wait=True):
        """
        发送关节运动指令
        :param wait: 是否阻塞等待直到运动结束，默认为 True
        """
        dummy_tcp_points = [0.0]*6
        tcp_name = "TCP"
        ucs_name = "Base"
        is_joint = 1
        is_seek = 0
        io_bit = 0
        io_state = 0
        cmd_id = "1"
        
        # 1. 发送运动指令   异步非阻塞！！！！！！！！！！！！！
        ret = self.client.HRIF_MoveJ(
            self.boxID, self.rbtID, dummy_tcp_points, target_joints_deg,
            tcp_name, ucs_name, vel, acc, radius, is_joint, is_seek,
            io_bit, io_state, cmd_id
        )
        
        if ret != 0:
            raise RuntimeError(f"运动指令发送失败, 错误码: {ret}")
        
        # 2. 调用 SDK 自带的阻塞等待函数
        if wait:
            print("正在等待机械臂到位...")
            self.client.waitMoveDone(self.boxID, self.rbtID)
            print("机械臂已到位")
        time.sleep(0.5)

    def get_tcp_pose_matrix(self):
        res_list = []
        ret2 = self.client.HRIF_ReadActPos(self.boxID, self.rbtID, res_list)
        tcp_pose = res_list[6:12]
        tcp_pose = [float(x) for x in tcp_pose]
        return hans_pose_to_matrix(tcp_pose)

def detect_chessboard_pose(img, pattern_size, square_len, K, D):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray, pattern_size, 
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ret:
        return False, None, None, img
    
    corners2 = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), 
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )
    
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
    objp *= square_len
    
    success, rvec, tvec = cv2.solvePnP(
        objp, 
        corners2, 
        K, 
        D,
        flags=cv2.SOLVEPNP_ITERATIVE)
    
    vis = img.copy()
    cv2.drawChessboardCorners(vis, pattern_size, corners2, ret)
    if success:
        cv2.drawFrameAxes(vis, K, D, rvec, tvec, 0.1)
        
    return success, rvec, tvec, vis


def run_calibration_collection(robot, camera, loop_index=1):
    """
    执行标定数据采集
    :param robot: 已经连接好的机器人实例
    :param camera: 已经初始化好的相机实例
    :param loop_index: 循环索引
    """
    if not os.path.exists(TRAJECTORY_FILE):
        print(f"错误: 找不到轨迹文件 {TRAJECTORY_FILE}")
        return

    print(f"\n========== 第 {loop_index} 次执行 ==========")
    target_joints_array_deg = np.load(TRAJECTORY_FILE)
    print(f"共加载 {len(target_joints_array_deg)} 个点位")

    # 获取相机内参 (只需获取一次，但为了兼容之前的逻辑放在这里也行)
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

        color_frame, _ = camera.get_frames()
        img, _ = camera.get_images_from_frames(color_frame, None)
        
        if img is None:
            print("图像获取失败")
            continue

        ret, rvec, tvec, vis_img = detect_chessboard_pose(img, PATTERN_SIZE, SQUARE_LEN, K, D)
        
        filename = capture_dir / f"{loop_index}_{i+1:03d}_{timestamp}.png"
        cv2.imwrite(str(filename), vis_img)

        if ret:
            print("√ 棋盘格检测成功")
            collected_data["robot_pose"].append(matrix_to_list(tcp_matrix))
            collected_data["target_rvecs"].append(rvec.reshape(3).tolist())
            collected_data["target_tvecs"].append(tvec.reshape(3).tolist())
            success_count += 1
        else:
            print("× 未检测到棋盘格")
        time.sleep(WAIT_TIME)

    # 保存数据
    if success_count > 0:
        with open(save_path, 'w') as f:
            yaml.dump(collected_data, f)
        print(f"\n第 {loop_index} 次采集完成！有效数据: {success_count} 组。")
    else:
        print("\n采集失败，没有收集到有效数据。")
    
if __name__ == "__main__":
    robot = None
    camera = None
    
    try:
        # 1. 在循环外初始化资源
        print("初始化硬件连接...")
        robot = HansRobotWrapper(ROBOT_IP, ROBOT_PORT)
        
        camera = OrbbecCamera(
            color_width=2560, 
            color_height=1440, 
            color_fps=30,
            enable_depth=False,
            enable_alignment=False
        )
        time.sleep(2) # 等待相机预热

        # 2. 循环执行任务，传入相同的 robot 和 camera 实例
        for loop_idx in range(1, LOOP_COUNT + 1):
            try:
                run_calibration_collection(robot, camera, loop_index=loop_idx)
                time.sleep(1) # 批次间稍微暂停
            except Exception as e:
                print(f"第 {loop_idx} 次任务执行出现严重错误: {e}")
                # 视情况决定是否 break

    except Exception as e:
        print(f"全局初始化或运行时错误: {e}")
        
    finally:
        print("\n正在断开连接...")
        robot.disconnect()
        print("机器人已断开。")
        camera.stop()
        print("相机已停止。")
        print("程序结束。")