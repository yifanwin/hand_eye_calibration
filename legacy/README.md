# Legacy implementations

这里仅记录旧实现的来源，不参与安装、导入或测试。

- `main`（提交 `9b7ea2e`）：Hans/CPS 的 `save_pose.py`、`cps_hand_eye_calibration.py` 和 `rot_trans_avg_eye_to_hand.py`。
- `franka-rekep`（提交 `eea2dc4`）：`frank_robot_wrapper.py`、Franka 采集/计算/单体脚本和 `orbbec_camera_v6.py`。

迁移取舍：

- CPS/HRIF 调用和 mm/degree 转换重构到 `HansCPSAdapter`。
- `franky` 状态与运动调用重构到 `FrankaRobotAdapter`，公共单位改为 rad/m。
- Orbbec profile、帧解码和设备标定读取重构到精简 `OrbbecAdapter`。
- 棋盘检测和 Park 求解迁移到独立 detector/solver。
- 逐轮标定后平均、深度内参估算、彩色内参 fallback、硬编码输出路径、相机内的点云/录像/世界变换职责废弃。
- 历史生成数据、轨迹和输出不迁移；需要追溯时从上述 Git 提交读取。
