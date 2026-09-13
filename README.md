# Hand–Eye Calibration

统一的 Eye-to-Hand 标定工具。当前标准组合是 **Franka Research 3 + Intel RealSense D435**，同时保留 **Hans CPS/HRIF + Orbbec**。核心采集、求解、验证和导出代码不依赖具体硬件 SDK；新增设备通过 Adapter 和 YAML 配置接入。

## 坐标与单位

- 变换统一写作 `T_A_B @ p_B = p_A`。
- `camera` 指彩色光学坐标系。
- 机器人状态为 `T_base_ee`，棋盘检测为 `T_camera_target`，最终结果为 `T_base_camera`。
- Eye-to-Hand 约束：`T_base_camera @ T_camera_target = T_base_ee @ T_ee_target`。
- 关节角为 rad，平移为 m，所有位姿为 `float64` 4×4 SE(3)。

## 安装

```bash
pip install -e '.[test]'
# FR3
pip install -e '.[franka,realsense]'
```

Orbbec 使用现场安装的 `pyorbbecsdk` v2。Hans 使用厂商提供的 CPS Python 模块，不随本仓库分发；其模块路径由 `cps_module` 配置。核心包采用 SDK 懒加载，因此未安装硬件依赖时仍可导入、求解和运行离线测试。

## 标定流程

### 1. 示教轨迹

先在机器人侧开启 freedrive，再记录关节位置：

```bash
handeye teach --config configs/fr3_d435.yaml \
  --output data/trajectories/fr3.yaml
```

轨迹使用版本化 YAML，waypoint 只保存 rad，不读取旧的 degree `.npy`。

### 2. 多轮采集

```bash
handeye collect --config configs/fr3_d435.yaml \
  --trajectory data/trajectories/fr3.yaml \
  --session data/sessions/fr3_d435_001 --rounds 3
```

每轮产生独立 `run_id`，有效 observation 逐行追加到同一个 `observations.jsonl`。中断不会破坏已经落盘的数据。再次指定同一 session 时，会先核对机器人身份、相机序列号、实际 stream profiles 和设备标定，防止混入不同硬件数据。

### 3. 联合求解和验证

```bash
handeye solve --config configs/fr3_d435.yaml \
  --session data/sessions/fr3_d435_001
```

所有 run 的 observation 只调用一次 OpenCV Park 手眼求解，不计算或平均“每轮结果”。硬错误阻止求解；质量超阈值写入 `validation.status=warning`，但保留结果供现场检查。

### 4. 规范导出

```bash
handeye export --session data/sessions/fr3_d435_001 \
  --output data/exports/fr3_d435_001.json
```

导出中 `T_base_camera` 和设备原始 `T_color_depth` 是两个独立字段。工具不输出旧 Rekep/OEA 的歧义 `extrinsics` 格式。

## Session 格式

```text
data/sessions/<session_id>/
├── session.yaml          # 配置快照、设备标定、run summaries
├── observations.jsonl    # append-only observations
├── images/               # 原图与检测调试图
└── result.json           # 联合求解和验证报告
```

每条 observation 包含 `run_id`、时间戳、`joint_positions_rad`、`T_base_ee`、`T_camera_target`、重投影误差和可选图像路径。相机标定包含实际 color/depth intrinsics、distortion model、`depth_scale_m_per_unit`、active profiles 和 `T_color_depth`；不存在缩放估算或彩色内参回退。

## 配置与扩展

参考 `configs/fr3_d435.yaml` 和 `configs/hans_orbbec.yaml`。硬件和 detector 使用完整 class path：

```yaml
robot:
  factory: hand_eye_calibration.adapters.robots.franka:FrankaRobotAdapter
  options: {host: 172.16.0.2}
```

新增机器人、相机或靶标实现对应接口并在新配置中引用即可；核心不维护设备 registry。

- `RobotAdapter`：阻塞关节运动和 `RobotState` 读取。
- `CameraAdapter`：帧采集及设备工厂标定读取。
- `TargetDetector`：从 BGR 图像和标准投影模型生成 `T_camera_target`。
- `Collector`：编排采集，不求解。
- `CalibrationSession`：持久化 session，不访问 SDK。
- `HandEyeSolver`：纯联合求解。
- `Validator`：输入门槛、运动覆盖、重投影和固定靶标一致性。
- `Exporter`：只序列化规范结果。

## 默认质量规则

硬失败：有效 observation 少于 10、SE(3)/schema 非法、设备或 profile 混用、最大相对旋转小于 15°、旋转轴秩小于 2。

质量警告：重投影 median/p95 超过 1/2 px，固定靶标 translation/rotation RMS 超过 5 mm/0.5°，机器人平移/旋转跨度小于 0.1 m/30°。阈值均可在 YAML 中修改；v1 不自动删除异常点。

## 测试

```bash
pytest -q
```

离线测试覆盖 SE(3)、session round-trip、多 run 追加、synthetic Eye-to-Hand 精确恢复、验证门槛、规范导出、动态 class-path 装载及 Adapter 单位转换。FR3+D435 和 Hans+Orbbec 的真实连接、三轮采集与现场精度仍需在对应硬件在线时执行。

## Legacy

旧的脚本式实现不再是可执行入口，也不兼容旧 YAML/NPY/旧 Rekep JSON。其来源和取舍记录在 `legacy/README.md`，完整历史仍保留在 `main` 与 `franka-rekep` 的 Git 提交中。

