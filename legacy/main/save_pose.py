import time
import sys
import numpy as np
from utils.CPS import CPSClient

# --- 机器人连接配置 (来自您的代码) ---
cps = CPSClient()
IP = "192.168.156.2" 
Port = 10003
# 为保存的轨迹文件名添加时间戳，方便区分多次采集
timestamp = time.strftime("%Y%m%d_%H%M%S")
tra_joints_path = f'./hand_eye_calibration/trajectory/trajectory_joints_{timestamp}.npy'
tra_car_path = f'./hand_eye_calibration/trajectory/trajectory_cartesian_{timestamp}.npy'

# --- 存储采集到的点 ---
# 我们需要同时保存关节和空间坐标，以便后续精确复现
joint_points_list = []
cartesian_points_list = []

# 1. 连接机器人
print(f"正在尝试连接到 {IP}:{Port}...")
ret_connect = cps.HRIF_Connect(0, IP, Port)
print(f"HRIF_Connect 返回值: {ret_connect}")

if ret_connect != 0:
    print(f"连接失败 (错误码: {ret_connect})！请检查 IP、网络或机器人控制器状态。")
    sys.exit()

print("连接成功！")

try:
    # # 2. 使能机器人
    # print("正在使能机器人...")
    # ret_enable = cps.HRIF_GrpEnable(0, 0)
    # print(f"HRIF_GrpEnable 返回: {ret_enable}")
    # if ret_enable != 0:
    #     raise Exception("机器人使能失败！请检查是否急停或有错误。")
    # time.sleep(2) # 等待使能完成

    # 3. 开始采集
    print("\n" + "="*30)
    input("请确保机器人已使能且姿态稳定，按 [Enter] 键开始拖动示教...")

    # HRIF_GrpOpenFreeDriver: 打开自由驱动
    # ret_free_drive = cps.HRIF_GrpOpenFreeDriver(0, 0)
    # if ret_free_drive != 0:
    #     # raise Exception("开启自由驱动模式失败！")
    #     print("wrong!!!!!! 开启自由驱动模式失败！ ")
    
    print("\n自由驱动已开启。")
    print("请拖动机器人到您想要的轨迹点...")

    while True:
        key_input = input(f"已采集 {len(joint_points_list)} 个点。按 [Enter] 记录当前点 | 输入 [q] 完成并保存: ")
        
        if key_input.lower() == 'q':
            break

        # HRIF_ReadActPos: 读取当前实际位置信息
        pos_result = []
        ret_read = cps.HRIF_ReadActPos(0, 0, pos_result)
        
        if ret_read == 0 and len(pos_result) >= 12:
            # 根据PDF 3.6.1: [0-5]是关节, [6-11]是空间(笛卡尔)
            current_joint_pose = [float(j) for j in pos_result[0:6]]
            current_cartesian_pose = [float(p) for p in pos_result[6:12]]
            
            # 存入列表
            joint_points_list.append(current_joint_pose)
            cartesian_points_list.append(current_cartesian_pose)
            
            print(f"点 {len(joint_points_list)} 记录成功 (X,Y,Z): [{current_cartesian_pose[0]:.1f}, {current_cartesian_pose[1]:.1f}, {current_cartesian_pose[2]:.1f}]")
        else:
            print(f"读取位置失败，错误码: {ret_read}")

    print("采集结束。")

except Exception as e:
    print(f"\n发生错误: {e}")

finally:
    # 4. 无论如何都关闭自由驱动、去使能并断开
    # print("正在关闭自由驱动模式...")
    # cps.HRIF_GrpCloseFreeDriver(0, 0)
    
    # print("正在去使能机器人...")
    # cps.HRIF_GrpDisable(0, 0)
    
    print("正在断开连接...")
    cps.HRIF_DisConnect(0)

# 5. 将采集到的数据保存为 .npy 文件
if len(joint_points_list) > 0:
    # 转换为 Numpy 数组
    np_joints = np.array(joint_points_list)
    np_cartesian = np.array(cartesian_points_list)
    
    # 保存文件
    np.save(tra_joints_path, np_joints)
    np.save(tra_car_path, np_cartesian)
    
    print("\n" + "="*30)
    print("轨迹保存成功！")
    print(f"关节数据已保存到:{tra_joints_path} ")
    print(f"空间数据已保存到: {tra_car_path}  ")
    print("="*30)
else:
    print("\n未采集任何点，未保存文件。")

print("采集脚本执行完毕。")


"""
1.执行save_pose.py脚本，连接机器人，开启自由拖动模式，拖动机器人到各个目标位置，按回车键记录点，输入q结束采集。
2.执行cps_hand_eye_calibration.py脚本，会进行轨迹复现，可以录制多段在第3步取平均。
3.执行rot_trans_avg_eye_to_hand.py脚本，计算最终结果
"""
