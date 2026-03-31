# Franka 机械臂手眼标定

基于 OpenEmbodiedAgent (OEA) 框架的手眼标定程序，使用 Franka 机械臂替代原有 Hans 机械臂。

## 硬件要求

- Franka 机械臂（FCI 接口）
- Orbbec 相机（RGB-D）
- 棋盘格标定板（11x8 内角点，方块尺寸 10mm）

## 依赖

```bash
conda activate oea_rekep
pip install franky-control pyorbbecsdk
```

## 标定流程

### Step 1: 轨迹示教

采集机械臂运动轨迹，用于标定数据采集。

```bash
cd /home/fanfan/proj/oea
python hand_eye_calibration/save_pose_franky.py
```

**操作步骤：**
1. 确保 Franka 机器人已上电且 FCI 已启用
2. 运行脚本，自动连接机器人
3. 将机器人切换到 **Free-Drive 模式**
4. 手动拖动机械臂到目标位置，按 **Enter** 记录点位
5. 拖动到下一个位置，按 **Enter** 继续记录
6. 完成采集后输入 **q** 保存并退出

**输出文件：**
- `trajectory_joints_YYYYMMDD_HHMMSS.npy` - 关节角度 [度]
- `trajectory_cartesian_YYYYMMDD_HHMMSS.npy` - 笛卡尔位姿 [mm, 度]

> **提示**：建议采集 10-20 个点位，覆盖不同空间位置和姿态

---

### Step 2: 标定数据采集

沿示教轨迹运动，同步采集棋盘格图像和机械臂位姿。

```bash
python hand_eye_calibration/franky_calibration.py
```

**配置项（文件顶部）：**
```python
FRANKA_IP = "10.90.90.1"                    # Franka 机器人 IP
TRAJECTORY_FILE = 'hand_eye_calibration/trajectory/trajectory_joints.npy'  # 示教轨迹
LOOP_COUNT = 3                              # 重复采集次数
PATTERN_SIZE = (11, 8)                      # 棋盘格内角点数
SQUARE_LEN = 0.01                           # 方块尺寸 [m]
```

**操作步骤：**
1. 放置棋盘格标定板在相机视野内
2. 运行脚本，自动连接机器人和相机
3. 机械臂沿轨迹运动，自动采集数据
4. 可重复执行多次（数据取平均）

**输出文件：**
- `calib_data/h_e_calib_data_YYYYMMDD_HHMMSS_{loop}.yaml` - 每轮采集的标定数据
- `captures/{loop}_{idx}.png` - 棋盘格检测可视化图像

---

### Step 3: 计算标定结果

计算相机到机器人基座的外参变换矩阵。

标定结果在 Step 2 中自动计算并保存。如需单独运行：

```bash
# 编辑脚本中的路径和时间戳后
python hand_eye_calibration/rot_trans_avg_eye_to_hand.py
```

---

## 输出结果

标定结果保存至：
```
/home/fanfan/proj/oea/oea-rekep-real-plugin/runtime/real_calibration/orbbec_config/orbbec_calibration.json
```

**JSON 格式：**
```json
{
  "device_info": {
    "serial_number": "",
    "name": "Orbbec Camera",
    "firmware_version": "",
    "product_line": "Orbbec",
    "device_id": ""
  },
  "timestamp": "20260330_153000",
  "color_intrinsics": {
    "width": 1280,
    "height": 720,
    "fx": 912.345,
    "fy": 912.345,
    "cx": 640.123,
    "cy": 360.123,
    "model": "distortion.brown_conrady",
    "coeffs": [0, 0, 0, 0, 0, 0, 0, 0],
    "distortion_model": "distortion.brown_conrady"
  },
  "depth_intrinsics": {
    "width": 640,
    "height": 576,
    "fx": 456.123,
    "fy": 456.123,
    "cx": 320.123,
    "cy": 288.123,
    "model": "distortion.brown_conrady",
    "coeffs": [0, 0, 0, 0, 0, 0, 0, 0],
    "distortion_model": "distortion.brown_conrady"
  },
  "depth_scale": 0.001,
  "extrinsics": {
    "rotation": [0.999, -0.017, 0.003, ...],
    "translation": [-0.009, 0.556, 1.064],
    "transform_matrix": [
      [0.999, -0.017, 0.003, -0.009],
      [-0.017, -0.999, 0.012, 0.556],
      [0.003, -0.012, -1.000, 1.064],
      [0, 0, 0, 1]
    ]
  },
  "stream_config": {
    "color": {"width": 1280, "height": 720, "format": "MJPG", "fps": 30},
    "depth": {"width": 640, "height": 576, "format": "Y16", "fps": 30}
  }
}
```

---

## 文件结构

```
hand_eye_calibration/
├── frank_robot_wrapper.py        # Franka 机器人封装类
├── franky_calibration.py         # Franka 标定主程序
├── save_pose_franky.py           # Franka 轨迹示教程序
├── orbbec_camera_v6.py           # Orbbec 相机封装（不变）
├── rot_trans_avg_eye_to_hand.py  # 标定算法（不变）
├── trajectory/                   # 示教轨迹存储
├── calib_data/                   # 标定数据存储
├── captures/                     # 图像采集存储
└── hand_eye_calib_output/        # 标定结果输出
```

---

## 快速参考

| 操作 | 命令 |
|------|------|
| 示教轨迹 | `python save_pose_franky.py` |
| 标定采集 | `python franky_calibration.py` |
| Franka IP | `10.90.90.1`（可配置） |
| 相机分辨率 | 1280x720 (color) / 640x576 (depth) |
| 棋盘格 | 11x8 内角点, 10mm 方块 |

---

## 故障排查

**连接失败：**
- 检查 Franka 机器人 FCI 是否启用
- 确认 IP 地址正确（默认 10.90.90.1）
- 检查网络连接

**棋盘格检测失败：**
- 确保标定板在相机视野内
- 调整光照避免反光
- 检查 PATTERN_SIZE 和 SQUARE_LEN 配置

**标定结果异常：**
- 增加 LOOP_COUNT 多采集几组数据
- 确保轨迹覆盖不同空间位置
