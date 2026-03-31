## 手眼标定


### 硬件支持
机械臂：华沿、franka、dobot

相机：orbbec、realsense

### 环境配置
```
pip install pyorbbecsdk2
```

linux系统需要配置
```bash
git clone https://github.com/orbbec/OrbbecSDK_v2.git
cd scripts/env_setup
  sudo chmod +x ./install_udev_rules.sh
  sudo ./install_udev_rules.sh
  sudo udevadm control --reload && sudo udevadm trigger
```


## 🤖 手眼标定流程 (Hand-Eye Calibration)

本模块位于 `cps_hand_eye_collect` 目录下，用于完成机械臂与相机的 Eye-to-Hand 标定。

### 第一步：采集位姿点

执行脚本：

```bash
python hand_eye_calibration/save_pose.py
```

**操作指南：**

1.  连接机器人，开启**自由拖动模式**。
2.  拖动机器人末端到各个不同的目标位置。
3.  按 **Enter (回车)** 键记录当前点。
4.  输入 **q** 结束采集并保存。

### 第二步：轨迹复现

执行脚本：

```bash
python hand_eye_calibration/cps_hand_eye_calibration.py
```

**说明：**

  * 机器人会自动进行轨迹复现。
  * 建议：可以录制多段轨迹（重复步骤一和二），以便在第三步中取平均值，提高标定精度。

### 第三步：计算最终结果

执行脚本：

```bash
python hand_eye_calibration/rot_trans_avg_eye_to_hand.py
```