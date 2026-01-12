#!/usr/bin/env python3
"""
CTE (Cross-Track Error) 估算调参工具

功能:
1. 加载图片或使用摄像头
2. 交互式调整 HSV 颜色阈值
3. 可视化边缘检测 / 中心线追踪结果
4. 测试不同的 CTE 估算方法
5. 导出配置

用法:
    python test_cte_tuner.py                          # 使用默认测试图片
    python test_cte_tuner.py --image path/to/img.jpg  # 指定图片
    python test_cte_tuner.py --dir real_road_data/    # 图片目录 (n/p 切换)
    python test_cte_tuner.py --camera                 # 使用摄像头

按键:
    q/ESC: 退出
    s: 保存当前配置
    1: 切换到边缘检测模式
    2: 切换到中心线追踪模式
    3: 切换到 Canny 边缘检测模式
    n: 下一张图片 (目录模式)
    p: 上一张图片 (目录模式)
    r: 重置参数
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from cte_estimator import VisualCTEEstimator


# 使用统一的 VisualCTEEstimator，添加返回 mask 的包装方法
class CTEEstimator:
    """CTE 估算器（用于调参工具）
    
    包装 VisualCTEEstimator，提供返回 mask 的接口以兼容调参工具。
    注意：centerline_tracking 现在使用和 color_edge_detection 相同的 HSV 阈值，
    但计算整个颜色区域的中心点而不是边缘。
    """

    def __init__(self, method: str = "canny_edges", max_cte: float = 3.0):
        # 使用统一的 VisualCTEEstimator
        self._estimator = VisualCTEEstimator(
            method=method,
            image_width=320,  # 默认值，实际会从图像获取
            image_height=240,
            max_cte=max_cte,
            track_lower=(0, 100, 100),
            track_upper=(180, 255, 255),
        )
        self.method = method
        self.max_cte = max_cte
        
        # 为了向后兼容，保留这些属性
        self.track_lower = self._estimator.track_lower.copy()
        self.track_upper = self._estimator.track_upper.copy()
        self.edge_lower = self.track_lower
        self.edge_upper = self.track_upper
        self.centerline_lower = self.track_lower
        self.centerline_upper = self.track_upper
        
        # debug_image 和 last_debug_image 都指向同一个
        self.last_debug_image = None

    def set_edge_thresholds(
        self, h_low: int, h_high: int, s_low: int, s_high: int, v_low: int, v_high: int
    ):
        """设置 HSV 阈值（同时用于 color_edge_detection 和 centerline_tracking）"""
        self._estimator.track_lower = np.array([h_low, s_low, v_low])
        self._estimator.track_upper = np.array([h_high, s_high, v_high])
        # 保持向后兼容
        self.track_lower = self._estimator.track_lower.copy()
        self.track_upper = self._estimator.track_upper.copy()
        self.edge_lower = self.track_lower
        self.edge_upper = self.track_upper
        self.centerline_lower = self.track_lower
        self.centerline_upper = self.track_upper

    def set_centerline_thresholds(
        self, h_low: int, h_high: int, s_low: int, s_high: int, v_low: int, v_high: int
    ):
        """设置中心线阈值（现在和边缘检测使用相同的阈值）"""
        # 现在 centerline_tracking 使用和 color_edge_detection 相同的阈值
        self.set_edge_thresholds(h_low, h_high, s_low, s_high, v_low, v_high)

    def estimate(
        self, frame_bgr: np.ndarray
    ) -> Tuple[float, float, Optional[np.ndarray]]:
        """估算 CTE，返回 (cte, confidence, mask)"""
        # 更新图像尺寸和方法
        h, w = frame_bgr.shape[:2]
        self._estimator.image_width = w
        self._estimator.image_height = h
        self._estimator.image_center = w // 2
        self._estimator.method = self.method
        
        # 调用统一的估计器
        cte, confidence = self._estimator.estimate(frame_bgr)
        
        # 获取 mask 和 debug image
        mask = self._estimator.last_mask_image
        self.last_debug_image = self._estimator.last_debug_image
        
        return float(cte), float(confidence), mask


class CTETuner:
    """CTE 调参工具"""

    def __init__(self):
        self.estimator = CTEEstimator()
        self.current_image: Optional[np.ndarray] = None
        self.image_paths: List[str] = []
        self.current_image_idx: int = 0
        self.camera = None

        # HSV 参数（统一用于 color_edge_detection 和 centerline_tracking）
        # 初始值从 estimator 的默认值同步
        self.h_low = int(self.estimator.track_lower[0])
        self.h_high = int(self.estimator.track_upper[0])
        self.s_low = int(self.estimator.track_lower[1])
        self.s_high = int(self.estimator.track_upper[1])
        self.v_low = int(self.estimator.track_lower[2])
        self.v_high = int(self.estimator.track_upper[2])

        # 窗口名称
        self.win_main = "CTE Tuner"
        self.win_mask = "Mask"
        self.win_hsv = "HSV Controls"

    def load_image(self, path: str) -> bool:
        """加载图片"""
        img = cv2.imread(path)
        if img is None:
            print(f"无法加载图片: {path}")
            return False
        self.current_image = img
        return True

    def load_directory(self, dir_path: str) -> bool:
        """加载目录中的所有图片"""
        path = Path(dir_path)
        extensions = (".jpg", ".jpeg", ".png", ".bmp")
        self.image_paths = sorted(
            [str(f) for f in path.iterdir() if f.suffix.lower() in extensions]
        )
        if not self.image_paths:
            print(f"目录中没有找到图片: {dir_path}")
            return False
        self.current_image_idx = 0
        return self.load_image(self.image_paths[0])

    def init_camera(self, camera_id: int = 0) -> bool:
        """初始化摄像头"""
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            print(f"无法打开摄像头: {camera_id}")
            return False
        print(f"摄像头已打开: {camera_id}")
        return True

    def get_frame(self) -> Optional[np.ndarray]:
        """获取当前帧"""
        if self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                return frame
            return None
        return self.current_image

    def next_image(self):
        """下一张图片"""
        if self.image_paths:
            self.current_image_idx = (self.current_image_idx + 1) % len(self.image_paths)
            self.load_image(self.image_paths[self.current_image_idx])
            print(
                f"图片: {self.current_image_idx + 1}/{len(self.image_paths)} - {os.path.basename(self.image_paths[self.current_image_idx])}"
            )

    def prev_image(self):
        """上一张图片"""
        if self.image_paths:
            self.current_image_idx = (self.current_image_idx - 1) % len(self.image_paths)
            self.load_image(self.image_paths[self.current_image_idx])
            print(
                f"图片: {self.current_image_idx + 1}/{len(self.image_paths)} - {os.path.basename(self.image_paths[self.current_image_idx])}"
            )

    def _on_trackbar(self, val):
        """Trackbar 回调 (空实现，实际读取在主循环)"""
        pass

    def _create_trackbars(self):
        """创建 HSV 调节滑块"""
        cv2.namedWindow(self.win_hsv, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_hsv, 400, 300)

        cv2.createTrackbar("H Low", self.win_hsv, self.h_low, 180, self._on_trackbar)
        cv2.createTrackbar("H High", self.win_hsv, self.h_high, 180, self._on_trackbar)
        cv2.createTrackbar("S Low", self.win_hsv, self.s_low, 255, self._on_trackbar)
        cv2.createTrackbar("S High", self.win_hsv, self.s_high, 255, self._on_trackbar)
        cv2.createTrackbar("V Low", self.win_hsv, self.v_low, 255, self._on_trackbar)
        cv2.createTrackbar("V High", self.win_hsv, self.v_high, 255, self._on_trackbar)

    def _read_trackbars(self):
        """读取滑块值"""
        self.h_low = cv2.getTrackbarPos("H Low", self.win_hsv)
        self.h_high = cv2.getTrackbarPos("H High", self.win_hsv)
        self.s_low = cv2.getTrackbarPos("S Low", self.win_hsv)
        self.s_high = cv2.getTrackbarPos("S High", self.win_hsv)
        self.v_low = cv2.getTrackbarPos("V Low", self.win_hsv)
        self.v_high = cv2.getTrackbarPos("V High", self.win_hsv)

    def _update_estimator_thresholds(self):
        """更新估算器阈值"""
        # color_edge_detection 和 centerline_tracking 现在使用相同的阈值
        if self.estimator.method in ["color_edge_detection", "centerline_tracking", "edge_detection"]:
            self.estimator.set_edge_thresholds(
                self.h_low, self.h_high, self.s_low, self.s_high, self.v_low, self.v_high
            )

    def _set_trackbar_values(self):
        """设置滑块到当前值"""
        cv2.setTrackbarPos("H Low", self.win_hsv, self.h_low)
        cv2.setTrackbarPos("H High", self.win_hsv, self.h_high)
        cv2.setTrackbarPos("S Low", self.win_hsv, self.s_low)
        cv2.setTrackbarPos("S High", self.win_hsv, self.s_high)
        cv2.setTrackbarPos("V Low", self.win_hsv, self.v_low)
        cv2.setTrackbarPos("V High", self.win_hsv, self.v_high)

    def _load_preset(self, preset: str):
        """加载预设参数"""
        presets = {
            "white": (0, 180, 0, 30, 200, 255),  # 白色边线
            "yellow": (20, 35, 100, 255, 100, 255),  # 黄色中心线
            "orange": (10, 25, 100, 255, 100, 255),  # 橙色线
            "black": (0, 180, 0, 255, 0, 50),  # 黑色
        }
        if preset in presets:
            self.h_low, self.h_high, self.s_low, self.s_high, self.v_low, self.v_high = (
                presets[preset]
            )
            self._set_trackbar_values()
            print(f"已加载预设: {preset}")

    def export_config(self):
        """导出当前配置到文件"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cte_config_{timestamp}.py"
        
        config_content = f'''# CTE 估算器配置
# 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# 方法: {self.estimator.method}

# HSV 参数:
# H: [{self.h_low}, {self.h_high}]
# S: [{self.s_low}, {self.s_high}]
# V: [{self.v_low}, {self.v_high}]

CTE_CONFIG = {{
    "method": "{self.estimator.method}",
    "max_cte": {self.estimator.max_cte},
    # HSV 阈值 - 用于 color_edge_detection 和 centerline_tracking
    # (centerline_tracking 现在使用和 color_edge_detection 相同的阈值)
    "track_lower": {tuple(self.estimator.track_lower.tolist())},
    "track_upper": {tuple(self.estimator.track_upper.tolist())},
}}

# 用于 real_car_env.py:
# from real_car_env import VisualCTEEstimator
# cte_estimator = VisualCTEEstimator(**CTE_CONFIG)
'''
        
        # 保存到文件
        with open(filename, 'w') as f:
            f.write(config_content)
        
        print("\n" + "=" * 60)
        print(f"配置已保存到: {filename}")
        print("=" * 60)
        print(config_content)
        print("=" * 60 + "\n")

    def run(self):
        """运行调参工具"""
        cv2.namedWindow(self.win_main, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.win_mask, cv2.WINDOW_NORMAL)
        self._create_trackbars()
        
        # 初始化时同步 trackbar 值到 estimator
        self._update_estimator_thresholds()

        print("\n按键说明:")
        print("  q/ESC: 退出")
        print("  s: 保存/导出当前配置")
        print("  1: Canny边缘检测模式 (canny_edges)")
        print("  2: 颜色边缘检测模式 (color_edge_detection)")
        print("  3: 中心线追踪模式 (centerline_tracking)")
        print("    注意: centerline_tracking 使用和 color_edge_detection 相同的 HSV 阈值")
        print("  n: 下一张图片")
        print("  p: 上一张图片")
        print("  w: 加载白色预设")
        print("  y: 加载黄色预设")
        print("  o: 加载橙色预设")
        print(f"\n当前方法: {self.estimator.method}\n")

        while True:
            frame = self.get_frame()
            if frame is None:
                print("无法获取图像")
                break

            # 读取滑块值并更新估算器
            self._read_trackbars()
            # 只有 color_edge_detection 和 centerline_tracking 需要 HSV 参数
            if self.estimator.method in ["color_edge_detection", "centerline_tracking", "edge_detection"]:
                self._update_estimator_thresholds()

            # 估算 CTE
            cte, confidence, mask = self.estimator.estimate(frame)

            # 创建显示图像
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]

            # 添加信息文本
            info_texts = [
                f"Method: {self.estimator.method}",
                f"CTE: {cte:+.2f} (pos=right, neg=left)",
                f"Confidence: {confidence:.2f}",
                f"H: [{self.h_low}, {self.h_high}]",
                f"S: [{self.s_low}, {self.s_high}]",
                f"V: [{self.v_low}, {self.v_high}]",
            ]

            for i, text in enumerate(info_texts):
                cv2.putText(
                    display_frame,
                    text,
                    (10, 25 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            # 如果有图片列表，显示当前索引
            if self.image_paths:
                img_info = f"Image: {self.current_image_idx + 1}/{len(self.image_paths)}"
                cv2.putText(
                    display_frame,
                    img_info,
                    (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

            # 合并显示: 原图 + 调试图
            if self.estimator.last_debug_image is not None:
                debug_resized = cv2.resize(
                    self.estimator.last_debug_image,
                    (w, self.estimator.last_debug_image.shape[0] * w // self.estimator.last_debug_image.shape[1]),
                )
                combined = np.vstack([display_frame, debug_resized])
            else:
                combined = display_frame

            cv2.imshow(self.win_main, combined)

            # 显示 mask
            if mask is not None:
                mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cv2.imshow(self.win_mask, mask_colored)

            # 处理按键
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q") or key == 27:  # q 或 ESC
                break
            elif key == ord("s"):
                self.export_config()
            elif key == ord("1"):
                self.estimator.method = "canny_edges"
                print("切换到: canny_edges")
                # Canny 方法不需要 HSV 参数
            elif key == ord("2"):
                self.estimator.method = "color_edge_detection"
                print("切换到: color_edge_detection")
                # 同步当前 trackbar 值到 estimator（不自动加载预设）
                self._update_estimator_thresholds()
            elif key == ord("3"):
                self.estimator.method = "centerline_tracking"
                print("切换到: centerline_tracking")
                # centerline_tracking 使用和 color_edge_detection 相同的 HSV 阈值
                # 同步当前 trackbar 值到 estimator（不自动加载预设）
                self._update_estimator_thresholds()
            elif key == ord("n"):
                self.next_image()
            elif key == ord("p"):
                self.prev_image()
            elif key == ord("w"):
                self._load_preset("white")
            elif key == ord("y"):
                self._load_preset("yellow")
            elif key == ord("o"):
                self._load_preset("orange")

        # 清理
        cv2.destroyAllWindows()
        if self.camera is not None:
            self.camera.release()


def main():
    parser = argparse.ArgumentParser(
        description="CTE 估算调参工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test_cte_tuner.py --image real_road_data/pic0.jpg
  python test_cte_tuner.py --dir real_road_data/
  python test_cte_tuner.py --camera
        """,
    )

    parser.add_argument("--image", "-i", type=str, help="输入图片路径")
    parser.add_argument("--dir", "-d", type=str, help="图片目录路径")
    parser.add_argument("--camera", "-c", action="store_true", help="使用摄像头")
    parser.add_argument("--camera-id", type=int, default=0, help="摄像头 ID (默认: 0)")

    args = parser.parse_args()

    tuner = CTETuner()

    # 确定输入源
    if args.camera:
        if not tuner.init_camera(args.camera_id):
            sys.exit(1)
    elif args.dir:
        if not tuner.load_directory(args.dir):
            sys.exit(1)
    elif args.image:
        if not tuner.load_image(args.image):
            sys.exit(1)
    else:
        # 默认尝试加载 real_road_data 目录
        default_dirs = [
            "real_road_data",
            "data/road",
        ]
        loaded = False
        for dir_path in default_dirs:
            if os.path.isdir(dir_path):
                if tuner.load_directory(dir_path):
                    print(f"已加载目录: {dir_path} ({len(tuner.image_paths)} 张图片)")
                    print("按 n 下一张, p 上一张")
                    loaded = True
                    break
        if not loaded:
            print("请指定图片路径: --image 或 --dir 或 --camera")
            print("示例: python test_cte_tuner.py --dir real_road_data/")
            sys.exit(1)

    tuner.run()


if __name__ == "__main__":
    main()

