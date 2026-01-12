#!/usr/bin/env python3
"""
Real-to-Sim Image Enhancement Script

将实际摄像头图像增强为与模拟环境图像更接近的风格。

主要处理:
1. 去除粉红/红色色偏 (白平衡校正)
2. 增强车道线颜色饱和度
3. 调整亮度和对比度
4. 修正镜头暗角
5. 可选的几何畸变校正

Usage:
    python enhance_real_to_sim.py --input real_road_data/pic0.jpg --output enhanced_output.jpg
    python enhance_real_to_sim.py --input real_road_data/ --output enhanced_output/ --batch
    python enhance_real_to_sim.py --interactive  # 交互式调参模式
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class RealToSimEnhancer:
    """Real-to-Sim image enhancement class."""

    def __init__(
        self,
        # 白平衡参数 - 减少红色/粉色色偏，保持自然色调
        wb_red_gain: float = 0.92,      # 红色通道增益 (<1 减少红色)
        wb_green_gain: float = 0.98,    # 绿色通道增益
        wb_blue_gain: float = 0.95,     # 蓝色通道增益
        
        # 径向颜色校正参数 - 解决边缘偏粉、中心正常的问题
        # 公式: gain = 1 / (1 + coef_r2 * r^2 + coef_r4 * r^4), r是归一化距离[0,1]
        radial_color_correction: bool = True,
        radial_red_coef_r2: float = 0.35,    # 红色r^2系数 (正值=边缘减少红色)
        radial_red_coef_r4: float = 0.25,    # 红色r^4系数 (正值=角落更强校正)
        radial_green_coef_r2: float = 0.0,   # 绿色r^2系数
        radial_green_coef_r4: float = 0.0,   # 绿色r^4系数
        radial_blue_coef_r2: float = -0.1,   # 蓝色r^2系数 (负值=边缘略增加蓝色)
        radial_blue_coef_r4: float = 0.0,    # 蓝色r^4系数
        
        # 色相/饱和度参数
        saturation_boost: float = 1.15,   # 整体饱和度增强 (保守)
        orange_saturation_boost: float = 1.5,  # 橙色车道线饱和度额外增强
        
        # 亮度/对比度参数
        brightness_offset: float = 12.0,  # 亮度偏移 (正值增亮)
        contrast_factor: float = 1.15,    # 对比度因子 (>1 增加对比度)
        gamma: float = 0.93,              # Gamma校正 (<1 提亮阴影，不要太激进)
        
        # 镜头暗角校正参数
        vignette_correction: bool = True,
        vignette_strength: float = 0.35,  # 暗角校正强度 (适中)
        
        # 色彩偏移校正 (Lab空间) - 全局色调微调
        lab_a_offset: float = -3.0,       # a通道偏移 (减少红/粉)
        lab_b_offset: float = 2.0,        # b通道偏移 (轻微增加暖黄色调)
    ):
        self.wb_red_gain = wb_red_gain
        self.wb_green_gain = wb_green_gain
        self.wb_blue_gain = wb_blue_gain
        
        # 径向颜色校正参数
        self.radial_color_correction = radial_color_correction
        self.radial_red_coef_r2 = radial_red_coef_r2
        self.radial_red_coef_r4 = radial_red_coef_r4
        self.radial_green_coef_r2 = radial_green_coef_r2
        self.radial_green_coef_r4 = radial_green_coef_r4
        self.radial_blue_coef_r2 = radial_blue_coef_r2
        self.radial_blue_coef_r4 = radial_blue_coef_r4
        
        self.saturation_boost = saturation_boost
        self.orange_saturation_boost = orange_saturation_boost
        
        self.brightness_offset = brightness_offset
        self.contrast_factor = contrast_factor
        self.gamma = gamma
        
        self.vignette_correction = vignette_correction
        self.vignette_strength = vignette_strength
        
        self.lab_a_offset = lab_a_offset
        self.lab_b_offset = lab_b_offset
        
        # 缓存
        self._vignette_map_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._radius_map_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    def _build_vignette_correction_map(self, h: int, w: int) -> np.ndarray:
        """构建暗角校正map (中心亮边缘暗的反向)."""
        cache_key = (h, w)
        if cache_key in self._vignette_map_cache:
            return self._vignette_map_cache[cache_key]
        
        cx, cy = w / 2.0, h / 2.0
        y, x = np.ogrid[:h, :w]
        
        # 计算到中心的归一化距离
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        dist_norm = dist / max_dist  # [0, 1]
        
        # 使用二次函数创建增益曲线 (边缘增益更大)
        # gain = 1 + strength * r^2
        gain = 1.0 + self.vignette_strength * (dist_norm ** 2)
        gain = gain.astype(np.float32)
        
        self._vignette_map_cache[cache_key] = gain
        return gain

    def _build_radius_maps(self, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        """构建归一化半径的r^2和r^4 maps，用于径向颜色校正."""
        cache_key = (h, w)
        if cache_key in self._radius_map_cache:
            return self._radius_map_cache[cache_key]
        
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        
        ys, xs = np.indices((h, w), dtype=np.float32)
        x = xs - cx
        y = ys - cy
        r2 = x * x + y * y
        
        # 归一化到[0, 1]，基于图像对角线距离
        max_r2 = cx * cx + cy * cy
        r2_norm = r2 / max_r2
        r4_norm = r2_norm * r2_norm
        
        self._radius_map_cache[cache_key] = (r2_norm, r4_norm)
        return r2_norm, r4_norm

    def apply_radial_color_correction(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        应用径向颜色校正 - 解决边缘偏粉、中心正常的镜头色差问题.
        
        原理: 对每个颜色通道应用基于距离的增益
        gain = 1 / (1 + coef_r2 * r^2 + coef_r4 * r^4)
        
        - 正系数: 边缘减少该颜色 (适用于边缘偏红/粉的情况)
        - 负系数: 边缘增加该颜色
        - 中心(r=0): gain = 1, 保持原色
        - 边缘(r=1): gain 由系数决定
        """
        if not self.radial_color_correction:
            return img_bgr
        
        h, w = img_bgr.shape[:2]
        r2, r4 = self._build_radius_maps(h, w)
        
        # 计算每个通道的增益
        # 红色通道增益 (降低边缘红色)
        gain_r = 1.0 / (1.0 + self.radial_red_coef_r2 * r2 + self.radial_red_coef_r4 * r4)
        # 绿色通道增益
        gain_g = 1.0 / (1.0 + self.radial_green_coef_r2 * r2 + self.radial_green_coef_r4 * r4)
        # 蓝色通道增益 (可能需要在边缘增加一点蓝色来中和粉色)
        gain_b = 1.0 / (1.0 + self.radial_blue_coef_r2 * r2 + self.radial_blue_coef_r4 * r4)
        
        # 限制增益范围，避免极端值
        gain_r = np.clip(gain_r, 0.3, 3.0)
        gain_g = np.clip(gain_g, 0.3, 3.0)
        gain_b = np.clip(gain_b, 0.3, 3.0)
        
        # 应用增益
        img_f = img_bgr.astype(np.float32)
        b_ch, g_ch, r_ch = cv2.split(img_f)
        
        r_corr = np.clip(r_ch * gain_r, 0, 255)
        g_corr = np.clip(g_ch * gain_g, 0, 255)
        b_corr = np.clip(b_ch * gain_b, 0, 255)
        
        corrected = cv2.merge([b_corr, g_corr, r_corr])
        return corrected.astype(np.uint8)

    def apply_white_balance(self, img_bgr: np.ndarray) -> np.ndarray:
        """应用白平衡校正 - 减少粉红色偏."""
        img_f = img_bgr.astype(np.float32)
        
        # BGR顺序
        img_f[:, :, 0] *= self.wb_blue_gain   # B
        img_f[:, :, 1] *= self.wb_green_gain  # G
        img_f[:, :, 2] *= self.wb_red_gain    # R
        
        return np.clip(img_f, 0, 255).astype(np.uint8)

    def apply_lab_color_correction(self, img_bgr: np.ndarray) -> np.ndarray:
        """在Lab色彩空间中校正色偏."""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        
        # L: 亮度 [0, 255]
        # a: 绿-红 [-128, 127] (映射到0-255)
        # b: 蓝-黄 [-128, 127] (映射到0-255)
        
        lab[:, :, 1] = np.clip(lab[:, :, 1] + self.lab_a_offset, 0, 255)  # a通道
        lab[:, :, 2] = np.clip(lab[:, :, 2] + self.lab_b_offset, 0, 255)  # b通道
        
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)

    def apply_saturation_boost(self, img_bgr: np.ndarray) -> np.ndarray:
        """增强饱和度,特别是橙色车道线."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # 整体饱和度增强
        hsv[:, :, 1] *= self.saturation_boost
        
        # 针对橙色区域额外增强 (H: 10-25 范围)
        h_channel = hsv[:, :, 0]
        orange_mask = ((h_channel >= 8) & (h_channel <= 25)).astype(np.float32)
        
        # 对橙色区域额外增强饱和度
        extra_boost = (self.orange_saturation_boost - 1.0) * self.saturation_boost
        hsv[:, :, 1] += orange_mask * hsv[:, :, 1] * extra_boost
        
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def apply_brightness_contrast(self, img_bgr: np.ndarray) -> np.ndarray:
        """调整亮度和对比度."""
        img_f = img_bgr.astype(np.float32)
        
        # 对比度: 以128为中心缩放
        img_f = (img_f - 128.0) * self.contrast_factor + 128.0
        
        # 亮度偏移
        img_f += self.brightness_offset
        
        return np.clip(img_f, 0, 255).astype(np.uint8)

    def apply_gamma_correction(self, img_bgr: np.ndarray) -> np.ndarray:
        """应用Gamma校正."""
        if abs(self.gamma - 1.0) < 0.01:
            return img_bgr
        
        # 创建查找表
        inv_gamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                          for i in range(256)]).astype(np.uint8)
        
        return cv2.LUT(img_bgr, table)

    def apply_vignette_correction(self, img_bgr: np.ndarray) -> np.ndarray:
        """校正镜头暗角 (边缘提亮)."""
        if not self.vignette_correction:
            return img_bgr
        
        h, w = img_bgr.shape[:2]
        gain_map = self._build_vignette_correction_map(h, w)
        
        img_f = img_bgr.astype(np.float32)
        
        # 对每个通道应用增益
        for c in range(3):
            img_f[:, :, c] *= gain_map
        
        return np.clip(img_f, 0, 255).astype(np.uint8)

    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        完整的图像增强流程.
        
        处理顺序:
        1. 径向颜色校正 (解决边缘偏粉、中心正常的镜头色差)
        2. 暗角校正 (先处理亮度不均)
        3. 白平衡校正 (RGB通道增益 - 全局微调)
        4. Lab色彩空间校正 (精细色偏调整)
        5. 饱和度增强 (突出车道线)
        6. 亮度/对比度调整
        7. Gamma校正
        
        Args:
            img_bgr: 输入BGR图像 (uint8)
            
        Returns:
            增强后的BGR图像 (uint8)
        """
        result = img_bgr.copy()
        
        # 1. 径向颜色校正 (关键步骤: 解决边缘偏粉的问题)
        result = self.apply_radial_color_correction(result)
        
        # 2. 暗角校正
        result = self.apply_vignette_correction(result)
        
        # 3. 白平衡 (全局微调)
        result = self.apply_white_balance(result)
        
        # 4. Lab色彩空间校正
        result = self.apply_lab_color_correction(result)
        
        # 5. 饱和度增强
        result = self.apply_saturation_boost(result)
        
        # 6. 亮度/对比度
        result = self.apply_brightness_contrast(result)
        
        # 7. Gamma校正
        result = self.apply_gamma_correction(result)
        
        return result


def apply_geometric_undistort(
    img_bgr: np.ndarray,
    *,
    calib_w: int = 640,
    calib_h: int = 480,
) -> np.ndarray:
    """
    应用JetRacer相机的几何畸变校正.
    
    使用预先标定的相机参数进行畸变校正.
    """
    # JetRacer相机标定参数 (从real_obs_preprocess.py获取)
    jetracer_mtx = np.array(
        [
            [402.60350228, 0.0, 263.30000918],
            [0.0, 537.76023089, 278.24728515],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    
    dist_coeffs = np.array(
        [[-0.31085325, -0.11558236, 0.00249467, -0.00088277, 0.51442531]], 
        dtype=np.float32
    )
    
    h, w = img_bgr.shape[:2]
    
    # 缩放相机矩阵
    sx = float(w) / float(calib_w)
    sy = float(h) / float(calib_h)
    mtx = jetracer_mtx.copy()
    mtx[0, 0] *= sx  # fx
    mtx[1, 1] *= sy  # fy
    mtx[0, 2] *= sx  # cx
    mtx[1, 2] *= sy  # cy
    
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img_bgr, mtx, dist_coeffs, None, newcameramtx)
    
    # 裁剪ROI
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[y : y + rh, x : x + rw]
    
    return undistorted


def create_comparison_image(
    original: np.ndarray, 
    enhanced: np.ndarray,
    sim_reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """创建对比图像 (原图 | 增强后 | 参考)."""
    # 确保尺寸一致
    h, w = original.shape[:2]
    enhanced_resized = cv2.resize(enhanced, (w, h))
    
    images = [original, enhanced_resized]
    labels = ["Original", "Enhanced"]
    
    if sim_reference is not None:
        sim_resized = cv2.resize(sim_reference, (w, h))
        images.append(sim_resized)
        labels.append("Sim Reference")
    
    # 添加标签
    labeled_images = []
    for img, label in zip(images, labels):
        img_with_label = img.copy()
        cv2.putText(
            img_with_label, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )
        labeled_images.append(img_with_label)
    
    return np.hstack(labeled_images)


def interactive_tuning(image_paths: list, sim_reference_path: Optional[str] = None):
    """交互式参数调优模式."""
    print("\n=== 交互式参数调优模式 ===")
    print("按键说明:")
    print("  q/ESC: 退出")
    print("  s: 保存当前参数")
    print("  n: 下一张图片")
    print("  p: 上一张图片")
    print("  r: 重置参数")
    print("  1-9: 选择要调节的参数")
    print("  +/-: 增加/减少参数值")
    print()
    
    # 加载参考图像
    sim_ref = None
    if sim_reference_path and os.path.exists(sim_reference_path):
        sim_ref = cv2.imread(sim_reference_path)
    
    # 默认参数
    default_params = {
        "wb_red_gain": 0.92,
        "wb_green_gain": 0.98,
        "wb_blue_gain": 0.95,
        "radial_red_coef_r2": 0.35,
        "radial_red_coef_r4": 0.25,
        "radial_blue_coef_r2": -0.1,
        "saturation_boost": 1.15,
        "orange_saturation_boost": 1.5,
        "brightness_offset": 12.0,
        "contrast_factor": 1.15,
        "gamma": 0.93,
        "vignette_strength": 0.35,
        "lab_a_offset": -3.0,
        "lab_b_offset": 2.0,
    }
    
    params = default_params.copy()
    param_names = list(params.keys())
    param_steps = {
        "wb_red_gain": 0.02,
        "wb_green_gain": 0.02,
        "wb_blue_gain": 0.02,
        "radial_red_coef_r2": 0.05,
        "radial_red_coef_r4": 0.05,
        "radial_blue_coef_r2": 0.05,
        "saturation_boost": 0.05,
        "orange_saturation_boost": 0.1,
        "brightness_offset": 3.0,
        "contrast_factor": 0.05,
        "gamma": 0.02,
        "vignette_strength": 0.05,
        "lab_a_offset": 1.0,
        "lab_b_offset": 1.0,
    }
    
    current_param_idx = 0
    current_image_idx = 0
    
    cv2.namedWindow("Real-to-Sim Enhancement", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-to-Sim Enhancement", 1400, 500)
    
    while True:
        # 加载当前图片
        img_path = image_paths[current_image_idx]
        original = cv2.imread(img_path)
        if original is None:
            print(f"无法加载图片: {img_path}")
            current_image_idx = (current_image_idx + 1) % len(image_paths)
            continue
        
        # 创建增强器
        enhancer = RealToSimEnhancer(**params)
        enhanced = enhancer.enhance(original)
        
        # 创建对比图
        comparison = create_comparison_image(original, enhanced, sim_ref)
        
        # 添加参数信息
        info_img = np.zeros((200, comparison.shape[1], 3), dtype=np.uint8)
        y_pos = 20
        for i, name in enumerate(param_names):
            marker = ">>>" if i == current_param_idx else "   "
            text = f"{marker} [{i+1}] {name}: {params[name]:.2f}"
            color = (0, 255, 255) if i == current_param_idx else (200, 200, 200)
            cv2.putText(info_img, text, (10 + (i // 6) * 400, 20 + (i % 6) * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # 显示文件名
        cv2.putText(info_img, f"File: {os.path.basename(img_path)} ({current_image_idx+1}/{len(image_paths)})",
                    (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        
        display = np.vstack([comparison, info_img])
        cv2.imshow("Real-to-Sim Enhancement", display)
        
        key = cv2.waitKey(50) & 0xFF
        
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord('s'):
            # 保存参数
            print("\n当前参数:")
            for name, value in params.items():
                print(f"    {name}={value:.2f},")
            print()
        elif key == ord('n'):
            current_image_idx = (current_image_idx + 1) % len(image_paths)
        elif key == ord('p'):
            current_image_idx = (current_image_idx - 1) % len(image_paths)
        elif key == ord('r'):
            params = default_params.copy()
            print("参数已重置")
        elif ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(param_names):
                current_param_idx = idx
        elif key == ord('+') or key == ord('='):
            name = param_names[current_param_idx]
            params[name] += param_steps[name]
        elif key == ord('-') or key == ord('_'):
            name = param_names[current_param_idx]
            params[name] -= param_steps[name]
    
    cv2.destroyAllWindows()
    return params


def process_single_image(
    input_path: str,
    output_path: str,
    enhancer: RealToSimEnhancer,
    *,
    apply_undistort: bool = False,
    save_comparison: bool = False,
) -> None:
    """处理单张图片."""
    img = cv2.imread(input_path)
    if img is None:
        print(f"错误: 无法读取图片 {input_path}")
        return
    
    # 可选: 几何畸变校正
    if apply_undistort:
        img = apply_geometric_undistort(img)
    
    # 应用增强
    enhanced = enhancer.enhance(img)
    
    # 保存结果
    cv2.imwrite(output_path, enhanced)
    print(f"已保存: {output_path}")
    
    # 可选: 保存对比图
    if save_comparison:
        comparison = create_comparison_image(img, enhanced)
        comp_path = output_path.replace('.', '_comparison.')
        cv2.imwrite(comp_path, comparison)
        print(f"对比图已保存: {comp_path}")


def process_batch(
    input_dir: str,
    output_dir: str,
    enhancer: RealToSimEnhancer,
    *,
    apply_undistort: bool = False,
    extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
) -> None:
    """批量处理目录中的图片."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in extensions]
    
    print(f"找到 {len(image_files)} 张图片")
    
    for i, img_file in enumerate(image_files):
        output_file = output_path / img_file.name
        process_single_image(
            str(img_file), str(output_file), enhancer,
            apply_undistort=apply_undistort,
        )
        print(f"进度: {i+1}/{len(image_files)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-to-Sim Image Enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--input", "-i", type=str, help="输入图片路径或目录")
    parser.add_argument("--output", "-o", type=str, help="输出图片路径或目录")
    parser.add_argument("--batch", "-b", action="store_true", help="批量处理模式")
    parser.add_argument("--interactive", action="store_true", help="交互式调参模式")
    parser.add_argument("--undistort", "-u", action="store_true", help="应用几何畸变校正")
    parser.add_argument("--comparison", "-c", action="store_true", help="保存对比图")
    parser.add_argument("--sim-ref", type=str, help="模拟器参考图像路径 (用于对比)")
    
    # 增强参数
    parser.add_argument("--wb-red", type=float, default=0.92, help="白平衡-红色增益")
    parser.add_argument("--wb-green", type=float, default=0.98, help="白平衡-绿色增益")
    parser.add_argument("--wb-blue", type=float, default=0.95, help="白平衡-蓝色增益")
    
    # 径向颜色校正参数 (解决边缘偏粉问题)
    parser.add_argument("--radial-red-r2", type=float, default=0.35, help="径向红色r^2系数 (正值=边缘减少红色)")
    parser.add_argument("--radial-red-r4", type=float, default=0.25, help="径向红色r^4系数 (正值=角落更强校正)")
    parser.add_argument("--radial-blue-r2", type=float, default=-0.1, help="径向蓝色r^2系数 (负值=边缘增加蓝色)")
    
    parser.add_argument("--saturation", type=float, default=1.15, help="饱和度增强")
    parser.add_argument("--brightness", type=float, default=12.0, help="亮度偏移")
    parser.add_argument("--contrast", type=float, default=1.15, help="对比度因子")
    parser.add_argument("--gamma", type=float, default=0.93, help="Gamma校正")
    parser.add_argument("--vignette", type=float, default=0.35, help="暗角校正强度")
    parser.add_argument("--lab-a", type=float, default=-3.0, help="Lab a通道偏移")
    parser.add_argument("--lab-b", type=float, default=2.0, help="Lab b通道偏移")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 交互式模式
    if args.interactive:
        # 查找所有图片
        if args.input:
            input_path = Path(args.input)
            if input_path.is_dir():
                image_paths = sorted([str(f) for f in input_path.glob("*.jpg")] + 
                                     [str(f) for f in input_path.glob("*.png")])
            else:
                image_paths = [str(input_path)]
        else:
            # 默认使用real_road_data目录
            image_paths = sorted([str(f) for f in Path("real_road_data").glob("*.jpg")])
        
        if not image_paths:
            print("错误: 未找到图片文件")
            return
        
        final_params = interactive_tuning(image_paths, args.sim_ref)
        print("\n最终参数:")
        for name, value in final_params.items():
            print(f"    {name}={value:.2f},")
        return
    
    # 检查输入输出
    if not args.input or not args.output:
        print("错误: 请指定输入和输出路径 (--input, --output)")
        print("或使用 --interactive 进入交互式模式")
        return
    
    # 创建增强器
    enhancer = RealToSimEnhancer(
        wb_red_gain=args.wb_red,
        wb_green_gain=args.wb_green,
        wb_blue_gain=args.wb_blue,
        radial_red_coef_r2=args.radial_red_r2,
        radial_red_coef_r4=args.radial_red_r4,
        radial_blue_coef_r2=args.radial_blue_r2,
        saturation_boost=args.saturation,
        brightness_offset=args.brightness,
        contrast_factor=args.contrast,
        gamma=args.gamma,
        vignette_strength=args.vignette,
        lab_a_offset=args.lab_a,
        lab_b_offset=args.lab_b,
    )
    
    # 批量或单张处理
    if args.batch:
        process_batch(
            args.input, args.output, enhancer,
            apply_undistort=args.undistort,
        )
    else:
        process_single_image(
            args.input, args.output, enhancer,
            apply_undistort=args.undistort,
            save_comparison=args.comparison,
        )


if __name__ == "__main__":
    main()

