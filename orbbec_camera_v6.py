# -*- coding: utf-8 -*-
"""
orbbec_camera_v6.py    v4.4.0

用于 Orbbec 相机（使用 pyorbbecsdk）的封装类。
管理相机的初始化、配置、帧同步、对齐以及数据提取。
保留了原版所有功能，仅增加了适配 4K 分辨率的逻辑 (MJPG支持 + 可选深度流)。
"""

import pyorbbecsdk as obs
from pyorbbecsdk import OBSensorType, OBStreamType, OBFormat, OBFrameAggregateOutputMode
import numpy as np
import cv2
import time
import os
from typing import Tuple, Optional, Dict, Any
import json

class OrbbecCamera:
    """
    封装 Orbbec 相机操作的类。
    """
    def __init__(self, 
                 color_width: int = 1280, 
                 color_height: int = 720, 
                 color_fps: int = 30,
                 depth_width: int = 640, 
                 depth_height: int = 576, 
                 depth_fps: int = 30,
                 enable_alignment: bool = True,
                 enable_depth: bool = True,      # [新增] 是否开启深度流 (4K必须关)
                 depth_scale: float = 0.001,
                 
                 extrinsics: Optional[np.ndarray] = None,
                 video_writer: Optional[cv2.VideoWriter] = None,
                 record_path: Optional[str] = None): 
        """
        初始化相机管线、配置流和对齐过滤器。
        """
        
        print("正在初始化 Orbbec 相机...")
        self.config = obs.Config()
        self.pipeline = obs.Pipeline()
        self.enable_depth = enable_depth
        self.enable_alignment = enable_alignment
        
        # [4K适配] 如果禁用了深度，强制禁用对齐
        if not self.enable_depth and self.enable_alignment:
            print("警告: 已禁用深度流 (enable_depth=False)，自动禁用对齐。")
            self.enable_alignment = False
            
        self.align_filter = None
        self.depth_scale = depth_scale # 毫米转米
        self.extrinsics = extrinsics # 存储外参
        self.pipeline_started = False # (新增) 状态标志
        self.video_writer = video_writer # (新增) 视频写入器
        self.record_path = record_path   # (新增) 录制文件路径

        if self.extrinsics is not None:
            print(f"已加载相机外参 (Extrinsics) T_base_camera, shape: {self.extrinsics.shape}")

        self.color_profile = None
        self.depth_profile = None
        self.color_intrinsics = {}
        self.depth_intrinsics = {}

        # --- 1. 配置彩色流 (修改：4K适配优先尝试 MJPG) ---
        color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        
        # 尝试获取 MJPG (对于 4K 是必须的)
        try:
            self.color_profile = color_profile_list.get_video_stream_profile(
                color_width, color_height, OBFormat.MJPG, color_fps
            )
        except Exception:
            self.color_profile = None

        # 如果 MJPG 失败，回退到 RGB
        if self.color_profile is None:
            try:
                self.color_profile = color_profile_list.get_video_stream_profile(
                    color_width, color_height, OBFormat.RGB, color_fps
                )
            except Exception:
                self.color_profile = None

        # 兜底默认流
        if self.color_profile is None:
            print(f"警告: 找不到指定彩色流 ({color_width}x{color_height}@{color_fps}fps)。使用默认流。")
            self.color_profile = color_profile_list.get_default_video_stream_profile()
        
        self.config.enable_stream(self.color_profile)
        # (修改) 始终从硬件读取内参
        self.color_intrinsics = self._extract_intrinsics(self.color_profile)
        
        print(f"彩色流已配置: {self.color_profile}")
        print(f"彩色相机内参 (字典): {{'width': {self.color_intrinsics['width']}, 'height': {self.color_intrinsics['height']}, 'fx': {self.color_intrinsics['fx']}, 'fy': {self.color_intrinsics['fy']}, 'cx': {self.color_intrinsics['cx']}, 'cy': {self.color_intrinsics['cy']}}}")
        # print("彩色相机内参 (3x3 矩阵 K):")
        # print(self.get_color_intrinsics_matrix())
        # print(f"彩色相机畸变系数 (Distortion): {self.get_color_distortion_coeffs()}")


        # --- 2. 配置深度流 (修改：支持禁用) ---
        if self.enable_depth:
            try:
                depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                self.depth_profile = depth_profile_list.get_video_stream_profile(
                    depth_width, depth_height, OBFormat.Y16, depth_fps
                )
                if self.depth_profile is None:
                    print(f"警告: 找不到指定的深度流 (Y16, {depth_width}x{depth_height}@{depth_fps}fps)。尝试默认流...")
                    self.depth_profile = depth_profile_list.get_default_video_stream_profile()

                self.config.enable_stream(self.depth_profile)
                # (修改) 始终从硬件读取内参
                self.depth_intrinsics = self._extract_intrinsics(self.depth_profile)
                
                print(f"深度流已配置: {self.depth_profile}")
                print(f"深度相机内参 (字典): {{'width': {self.depth_intrinsics['width']}, 'height': {self.depth_intrinsics['height']}, 'fx': {self.depth_intrinsics['fx']}, 'fy': {self.depth_intrinsics['fy']}, 'cx': {self.depth_intrinsics['cx']}, 'cy': {self.depth_intrinsics['cy']}}}")
                # print("深度相机内参 (3x3 矩阵 K):")
                # print(self.get_depth_intrinsics_matrix())
                # print(f"深度相机畸变系数 (Distortion): {self.get_depth_distortion_coeffs()}")
            except Exception as e:
                print(f"配置深度流失败: {e}")
                self.enable_depth = False
        else:
            print("提示: 深度流已禁用 (enable_depth=False)。")

        # --- 3. 配置对齐和同步 ---
        if self.enable_alignment:
            self.align_filter = obs.AlignFilter(OBStreamType.COLOR_STREAM)
            print("已创建 AlignFilter (软件 D2C 到彩色流)。")
            self.config.set_frame_aggregate_output_mode(obs.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        
        # --- 4. 启动管线 ---
        self.pipeline.start(self.config)
        self.pipeline_started = True # (新增)
        if self.enable_alignment:
            print(f"Orbbec 相机初始化成功 (聚合模式: FULL_FRAME_REQUIRE)。")
        else:
            print(f"Orbbec 相机初始化成功 (未启用对齐)。")
        
        print("等待相机稳定启动...")
        time.sleep(1.5) # 等待相机稳定启动
        

    def _extract_intrinsics(self, profile) -> Dict[str, Any]:
        """
        (修改) 辅助函数：
        始终从相机硬件 Profile 获取内参、畸变和分辨率。
        """
        
        # 1. 获取内参 (fx, fy, cx, cy)
        params = profile.get_intrinsic()
        
        # 2. 获取畸变对象
        distortion_obj = profile.get_distortion()
        
        # 3. 从畸变对象构建 Numpy 数组
        distortion_coeffs = np.array([
            distortion_obj.k1, distortion_obj.k2, distortion_obj.p1, distortion_obj.p2, 
            distortion_obj.k3, distortion_obj.k4, distortion_obj.k5, distortion_obj.k6
        ], dtype=np.float32)

        # 4. 构建字典
        return {
            "width": params.width,
            "height": params.height,
            "fx": params.fx,
            "fy": params.fy,
            "cx": params.cx,
            "cy": params.cy,
            "distortion": distortion_coeffs
        }

    def get_color_intrinsics(self) -> Dict[str, Any]:
        """获取彩色相机的完整内参字典 (包含畸变)。"""
        return self.color_intrinsics

    def get_depth_intrinsics(self) -> Dict[str, Any]:
        """获取深度相机的完整内参字典 (包含畸变)。"""
        return self.depth_intrinsics

    def get_color_intrinsics_matrix(self) -> np.ndarray:
        """以 3x3 Numpy 矩阵形式获取彩色相机内参 K。"""
        intr = self.color_intrinsics
        k_matrix = np.array([
            [intr['fx'], 0,          intr['cx']],
            [0,          intr['fy'], intr['cy']],
            [0,          0,          1]
        ], dtype=np.float32)
        return k_matrix

    def get_depth_intrinsics_matrix(self) -> np.ndarray:
        """以 3x3 Numpy 矩阵形式获取深度相机内参 K。"""
        intr = self.depth_intrinsics
        k_matrix = np.array([
            [intr['fx'], 0,          intr['cx']],
            [0,          intr['fy'], intr['cy']],
            [0,          0,          1]
        ], dtype=np.float32)
        return k_matrix

    def get_color_distortion_coeffs(self) -> np.ndarray:
        """以 Numpy 数组形式获取彩色相机畸变系数。"""
        return self.color_intrinsics.get("distortion", np.zeros(8, dtype=np.float32))

    def get_depth_distortion_coeffs(self) -> np.ndarray:
        """以 Numpy 数组形式获取深度相机畸变系数。"""
        return self.depth_intrinsics.get("distortion", np.zeros(8, dtype=np.float32))

    def get_extrinsics(self) -> Optional[np.ndarray]:
        """获取相机外参（T_base_camera） 4x4 矩阵。"""
        return self.extrinsics
        
    def get_rotation_matrix(self) -> Optional[np.ndarray]:
        """从 4x4 外参矩阵中获取 3x3 旋转矩阵 (R)。"""
        if self.extrinsics is not None and self.extrinsics.shape == (4, 4):
            return self.extrinsics[0:3, 0:3]
        elif self.extrinsics is not None:
             print(f"警告: get_rotation_matrix() 失败, 外参矩阵形状不是 (4, 4)，而是 {self.extrinsics.shape}")
        return None

    def get_translation_vector(self) -> Optional[np.ndarray]:
        """从 4x4 外参矩阵中获取 3x1 平移向量 (t)。"""
        if self.extrinsics is not None and self.extrinsics.shape == (4, 4):
            return self.extrinsics[0:3, 3]
        elif self.extrinsics is not None:
             print(f"警告: get_translation_vector() 失败, 外参矩阵形状不是 (4, 4)，而是 {self.extrinsics.shape}")
        return None
    
    def save_intrinsics(self, save_dir: str):
        """保存相机内参到 JSON"""
        os.makedirs(save_dir, exist_ok=True)

        # 辅助函数：处理字典副本，避免修改 self 中的原始 numpy 数据
        def _process_dict(intr_dict):
            d = intr_dict.copy()
            if 'distortion' in d and isinstance(d['distortion'], np.ndarray):
                d['distortion'] = d['distortion'].tolist()
            return d

        data = {
            "color_intrinsics": _process_dict(self.color_intrinsics),
            "depth_intrinsics": _process_dict(self.depth_intrinsics),
            "description": "Orbbec Camera Intrinsics & Distortion Coefficients"
        }

        json_path = os.path.join(save_dir, "camera_intrinsics.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[System] 相机内参已保存至: {json_path}")
        
    def get_color_image(self):
        """获取彩色图像帧"""
        try:
            # [4K适配] 增加超时时间
            frames = self.pipeline.wait_for_frames(2000)
            if not frames:
                return None
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None

            # 获取彩色图像数据
            color_data = np.asanyarray(color_frame.get_data())
            color_format = color_frame.get_format()

            # 如果是MJPG格式，需要解码
            if color_format == OBFormat.MJPG: 
                color_bgr = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
            else:
                # RGB格式直接reshape
                width = color_frame.get_width()
                height = color_frame.get_height()
                color_bgr = color_data.reshape((height, width, 3))
                # 如果是RGB格式，转换为BGR
                if color_format == OBFormat.RGB:
                    color_bgr = cv2.cvtColor(color_bgr, cv2.COLOR_RGB2BGR)

            return color_bgr
        except Exception as e:
            print(f"获取彩色图像失败: {e}")
            return None
        

    def get_frames(self) -> Tuple[Optional[obs.ColorFrame], Optional[obs.DepthFrame]]:
        """
        (修改) 获取一组帧（可能已对齐）。
        增加内部重试逻辑以解决超时和 segfault 问题。
        """
        raw_frameset = None
        
        # (新增) 内部重试循环
        max_retry = 20
        # [4K适配] 增加超时时间到 2000ms
        timeout_ms = 2000
        
        for attempt in range(max_retry):
            raw_frameset = self.pipeline.wait_for_frames(timeout_ms) 
            if raw_frameset:
                break # 成功获取，跳出循环
            print(f"警告: 丢失帧 (wait_for_frames 返回为空), 重试 {attempt+1}/{max_retry}...")
            
        if not raw_frameset:
            print(f"警告: {max_retry} 次尝试后，wait_for_frames 始终返回为空。")
            raise RuntimeError("相机获取帧失败: wait_for_frames 返回为空。")
        
        # 1. 如果只开启了 RGB，直接返回 RGB，深度为 None
        if not self.enable_depth:
            color_frame = raw_frameset.get_color_frame()
            if color_frame:
                return color_frame, None
            else:
                raise RuntimeError("未获取到 Color 帧。")

        # 2. 如果未对齐 (宽松模式)
        if not self.enable_alignment:
            # --- A. 不需要对齐 ---
            color_frame = raw_frameset.get_color_frame()
            depth_frame = raw_frameset.get_depth_frame()
            # [4K适配] 只要有 Color 就可以，深度丢失不报错
            if color_frame:
                return color_frame, depth_frame
            else:
                print("警告: 原始帧集中缺少 Color 帧。")
                raise RuntimeError("相机获取帧失败: 缺少 Color 帧。")

        # --- B. 需要对齐 ---
        if self.align_filter is None:
             raise RuntimeError("对齐已启用 (enable_alignment=True) 但 align_filter 未初始化。")
             
        processed_frameset = self.align_filter.process(raw_frameset)
        if not processed_frameset:
            print("警告: 对齐失败 (align_filter.process)。")
            # 这种情况也可能由丢失的帧引起，因此也重试（尽管上面的循环应该已经捕获了）
            raise RuntimeError("相机获取帧失败: 对齐过程返回为空。")
        
        processed_frameset = processed_frameset.as_frame_set()
        color_frame = processed_frameset.get_color_frame()
        depth_frame = processed_frameset.get_depth_frame()
        
        if color_frame and depth_frame:
            # 成功获取并对齐
            return color_frame, depth_frame
        else:
             print("警告: 对齐帧集中缺少 Color或Depth 帧。")
             raise RuntimeError("相机获取帧失败: 对齐帧集不完整。")

    def get_images_from_frames(self, color_frame: obs.ColorFrame, depth_frame: Optional[obs.DepthFrame]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        从 ColorFrame 和 DepthFrame 解码为 Numpy 数组。
        """
        # 1. 处理彩色图像
        if color_frame is None:
            raise RuntimeError("get_images_from_frames: color_frame is None")

        color_data_buffer = color_frame.get_data()
        color_format = color_frame.get_format()
        
        if color_format == OBFormat.MJPG:
            color_data = np.frombuffer(color_data_buffer, dtype=np.uint8)
            color_image_bgr = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
        elif color_format == OBFormat.RGB:
             color_data = np.frombuffer(color_data_buffer, dtype=np.uint8)
             color_image_bgr = color_data.reshape((color_frame.get_height(), color_frame.get_width(), 3))
        else:
            raise RuntimeError(f"不支持的彩色图像格式: {color_format}")
            
        # color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. 处理深度图像 (原始脚本格式为 Y16 / uint16)
        depth_image_mm = None
        if depth_frame is not None:
            depth_data = depth_frame.get_data()
            if depth_data is not None:
                depth_image_mm = np.frombuffer(depth_data, dtype=np.uint16).reshape(
                    (depth_frame.get_height(), depth_frame.get_width())
                )
        
        return color_image_bgr , depth_image_mm

    def create_organized_point_cloud(self, depth_image_mm: np.ndarray, intrinsics: dict) -> np.ndarray:
        """
        (核心功能) 使用Numpy向量化操作，从深度图和内参创建 (H, W, 3) 的有序点云。
        """
        height, width = depth_image_mm.shape
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']

        v_map, u_map = np.indices((height, width)) 
        depth_in_meters = depth_image_mm.astype(np.float32) * self.depth_scale
        x = (u_map - cx) * depth_in_meters / fx
        y = (v_map - cy) * depth_in_meters / fy
        z = depth_in_meters
        pcd_organized = np.stack((x, y, z), axis=-1)
        
        return pcd_organized

    def transform_point_cloud_to_world(self, pcd_camera: np.ndarray) -> np.ndarray:
        """
        (新增) 将相机坐标系下的点云 (H, W, 3) 转换到世界坐标系 (Base 坐标系)。
        """
        if self.extrinsics is None:
            print("警告: transform_point_cloud_to_world() 失败, 未设置外参 (extrinsics)。")
            return pcd_camera
        
        if self.extrinsics.shape != (4, 4):
            print(f"警告: transform_point_cloud_to_world() 失败, 外参矩阵形状不是 (4, 4)，而是 {self.extrinsics.shape}")
            return pcd_camera

        T_base_camera = self.extrinsics
        height, width, _ = pcd_camera.shape
        
        points_camera_flat = pcd_camera.reshape(-1, 3)
        ones = np.ones((points_camera_flat.shape[0], 1), dtype=points_camera_flat.dtype)
        points_camera_homogeneous = np.hstack((points_camera_flat, ones))
        
        result_homogeneous_T = T_base_camera @ points_camera_homogeneous.T
        points_world_flat = result_homogeneous_T.T[:, :3]
        pcd_world = points_world_flat.reshape(height, width, 3)
        
        return pcd_world

    def get_aligned_point_cloud(self, aligned_depth_image_mm: np.ndarray, to_world_frame: bool = False) -> np.ndarray:
        """
        快捷方式：使用【彩色相机内参】从【已对齐的深度图】创建点云。
        """
        if not self.enable_alignment:
            print("警告: 对齐(alignment)未启用，但您请求了 'aligned' 点云。")
        
        pcd_camera = self.create_organized_point_cloud(aligned_depth_image_mm, self.color_intrinsics)
        
        if to_world_frame:
            return self.transform_point_cloud_to_world(pcd_camera)
        else:
            return pcd_camera

    def get_unaligned_point_cloud(self, unaligned_depth_image_mm: np.ndarray, to_world_frame: bool = False) -> np.ndarray:
        """
        快捷方式：使用【深度相机内参】从【原始(未对齐)深度图】创建点云。
        """
        pcd_camera = self.create_organized_point_cloud(unaligned_depth_image_mm, self.depth_intrinsics)

        if to_world_frame:
            return self.transform_point_cloud_to_world(pcd_camera)
        else:
            return pcd_camera
        
    def start_recording(self, filepath: str, fourcc_str: str = 'mp4v'):
        """
        (新增) 启动视频录制。
        这将使用在 __init__ 中配置的彩色流的参数 (FPS, Width, Height)。

        :param filepath: 保存的 .mp4 文件路径。
        :param fourcc_str: 视频编解码器, 默认为 'mp4v' (适用于 .mp4)。
        """
        if self.video_writer is not None:
            print("警告: 录制已经在进行中。请先调用 stop_recording()。")
            return

        if not self.pipeline_started or self.color_profile is None:
            print("错误: 必须在相机管线启动 (pipeline started) 并且 color_profile 有效后才能开始录制。")
            return

        fps = self.color_profile.get_fps()
        width = self.color_profile.get_width()
        height = self.color_profile.get_height()

        # 确保尺寸有效
        if width == 0 or height == 0 or fps == 0:
            print(f"错误: 无法获取有效的彩色流参数 (FPS: {fps}, W: {width}, H: {height})。无法开始录制。")
            return

        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        self.video_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        
        if not self.video_writer.isOpened():
            print(f"错误: 无法打开 VideoWriter，路径: {filepath}")
            self.video_writer = None
            return
        
        self.record_path = filepath
        print(f"录制已开始，将保存到: {filepath} (FPS: {fps}, Res: {width}x{height})")

    def write_video_frame(self, bgr_image: np.ndarray):
        """
        (新增) 将一个 BGR 图像帧写入视频文件。
        
        注意：图像尺寸必须与录制器初始化时的尺寸 (color_width, color_height) 完全一致。
        """
        if self.video_writer is not None and self.video_writer.isOpened():
            # (新增) 增加一个尺寸检查
            expected_shape = (self.color_intrinsics['height'], self.color_intrinsics['width'], 3)
            if bgr_image.shape != expected_shape:
                print(f"警告: 写入视频帧的尺寸 ({bgr_image.shape}) 与预期 ({expected_shape}) 不符。正在尝试缩放...")
                try:
                    bgr_image = cv2.resize(bgr_image, (expected_shape[1], expected_shape[0]))
                except Exception as e:
                    print(f"错误: 自动缩放帧失败: {e}")
                    return
            self.video_writer.write(bgr_image)

    def stop_recording(self):
        """
        (新增) 停止视频录制并释放文件。
        """
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            if self.record_path:
                print(f"录制已停止。视频已保存到: {self.record_path}")
                self.record_path = None

    def stop(self):
        """停止相机管线。"""
        # (新增) 增加检查，防止重复停止
        if self.pipeline_started:
            self.pipeline_started = False
            
            # (新增) 停止录制 (如果正在进行)
            if self.video_writer is not None:
                self.stop_recording()
                
            self.pipeline.stop()
            print("Orbbec 相机管线已停止。")

    def __del__(self):
        """析构函数，确保相机停止。"""
        # (修改) 检查 pipeline 是否存在且已启动
        if hasattr(self, 'pipeline_started') and self.pipeline_started:
            try:
                # (新增) 确保录制器被释放
                if hasattr(self, 'video_writer') and self.video_writer is not None:
                    self.stop_recording()
                    
                self.stop()
            except Exception as e:
                # 在垃圾回收期间抑制错误
                pass


if __name__ == "__main__":
    print("--- [OrbbecCamera Test] ---")
    
    # 1. 模拟一个外参矩阵 (T_base_camera)
    mock_T_base_cam = np.array([
        [1.0, 0.0, 0.0, 0.5], # 沿X轴平移 0.5m
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.3], # 沿Z轴平移 0.3m
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)

    camera = None
    
    # --- (新增) 录制测试的参数 ---
    record_filename = "orbbec_test_record.mp4"
    record_duration_sec = 5.0 # 录制 5 秒
    # --------------------------

    try:
        # 2. 初始化相机
        print("\n[Test] 正在初始化相机...")
        camera = OrbbecCamera(
            color_width=1280, 
            color_height=720, 
            color_fps=30,
            depth_width=640, 
            depth_height=576,
            enable_alignment=True, # 启用对齐
            extrinsics=mock_T_base_cam
        )
        print("\n[Test] 相机初始化完成。")
        
        # [新增] 保存内参测试
        camera.save_intrinsics("./output_data")
        
        print("[Test] 等待 1.0 秒让相机预热...")
        time.sleep(1.0)

        # 3. 测试参数获取 (Getters)
        print("\n--- [Test] 测试参数获取 ---")
        print("彩色相机 K 矩阵:\n", camera.get_color_intrinsics_matrix())
        print("深度相机 K 矩阵:\n", camera.get_depth_intrinsics_matrix())
        print("外参 (T_base_camera):\n", camera.get_extrinsics())

        # 4. (修改) 测试帧获取、点云、和视频录制
        print(f"\n--- [Test] 测试录制 (持续 {record_duration_sec} 秒) ---")
        
        # 4.1. (新增) 开始录制
        camera.start_recording(record_filename)
        
        # 4.2. (新增) 循环获取帧并写入
        start_time = time.time()
        frame_count = 0
        
        # 保持最后一帧的引用，用于点云测试
        last_bgr_img = None
        last_depth_img_mm = None 
        
        while (time.time() - start_time) < record_duration_sec:
            color_frame, depth_frame = camera.get_frames()
            
            # (注意: get_images_from_frames 返回 BGR 格式)
            bgr_img, depth_img_mm = camera.get_images_from_frames(color_frame, depth_frame)
            
            # (新增) 写入视频
            camera.write_video_frame(bgr_img)
            
            # (新增) 存储最后一帧
            last_bgr_img = bgr_img
            last_depth_img_mm = depth_img_mm
            frame_count += 1
        
        print(f"\n[Test] {record_duration_sec} 秒内共录制 {frame_count} 帧。")
        
        # 4.3. (新增) 测试点云生成 (使用录制循环中的最后一帧)
        print("\n--- [Test] 测试点云 (使用最后一帧) ---")
        if last_depth_img_mm is not None:
            pcd_world = camera.get_aligned_point_cloud(last_depth_img_mm, to_world_frame=True)
            print(f"生成的世界坐标系点云 Shape: {pcd_world.shape}")
            
            valid_points_mask = pcd_world[:, :, 2] > 0
            if np.any(valid_points_mask):
                sample_point = pcd_world[valid_points_mask][0]
                print(f"抽样一个有效世界坐标点 (X,Y,Z): {sample_point}")
            else:
                print("未在点云中找到有效深度点。")
        else:
             print("警告: 未能捕获到任何帧，跳过点云测试。")
        
        print("\n--- [Test] 测试成功 ---")

    except Exception as e:
        print(f"\n--- [Test] 测试失败 ---")
        import traceback
        traceback.print_exc()
    
    finally:
        # 5. 清理 (将自动停止录制)
        if camera is not None:
            print("\n[Test] 正在停止相机 (将自动保存录制)...")
            camera.stop() # stop() 方法现在会自动调用 stop_recording()
        else:
            print("\n[Test] 相机未初始化，无需清理。")