#!/usr/bin/env python3
"""
CTE Estimator 调试脚本

对比 test_cte_tuner.py 中的 CTEEstimator 和 real_car_env.py 中的
CTEEstimator/VisualCTEEstimator 的结果，显示 mask 和 debug 图像的差异。

用法:
    python debug_cte_estimator.py --image path/to/img.jpg
    python debug_cte_estimator.py --dir real_road_data/
    python debug_cte_estimator.py --camera
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 导入统一的实现
from cte_estimator import CTEEstimator, VisualCTEEstimator
from test_cte_tuner import CTEEstimator as TunerCTEEstimator


def compare_estimators(
    frame_bgr: np.ndarray,
    method: str = "color_edge_detection",
    track_lower: tuple = (0, 100, 100),
    track_upper: tuple = (180, 255, 255),
    max_cte: float = 3.0,
):
    """对比两个估计器的结果"""
    
    print(f"\n{'='*60}")
    print(f"方法: {method}")
    print(f"HSV 阈值: lower={track_lower}, upper={track_upper}")
    print(f"{'='*60}\n")
    
    # 1. real_car_env.py 中的 CTEEstimator (基础类，无可视化)
    base_estimator = CTEEstimator(
        method=method,
        image_width=frame_bgr.shape[1],
        image_height=frame_bgr.shape[0],
        max_cte=max_cte,
        track_lower=track_lower,
        track_upper=track_upper,
    )
    base_cte, base_conf = base_estimator.estimate(frame_bgr)
    print(f"[CTEEstimator (base)]")
    print(f"  CTE: {base_cte:.3f}")
    print(f"  Confidence: {base_conf:.3f}")
    print(f"  Has mask: False (base class doesn't store mask)")
    
    # 2. real_car_env.py 中的 VisualCTEEstimator (有可视化)
    visual_estimator = VisualCTEEstimator(
        method=method,
        image_width=frame_bgr.shape[1],
        image_height=frame_bgr.shape[0],
        max_cte=max_cte,
        track_lower=track_lower,
        track_upper=track_upper,
    )
    visual_cte, visual_conf = visual_estimator.estimate(frame_bgr)
    visual_mask = visual_estimator.last_mask_image
    visual_debug = visual_estimator.last_debug_image
    print(f"\n[VisualCTEEstimator]")
    print(f"  CTE: {visual_cte:.3f}")
    print(f"  Confidence: {visual_conf:.3f}")
    print(f"  Has mask: {visual_mask is not None}")
    if visual_mask is not None:
        print(f"  Mask shape: {visual_mask.shape}, dtype: {visual_mask.dtype}")
        print(f"  Mask non-zero pixels: {np.count_nonzero(visual_mask)}")
    print(f"  Has debug image: {visual_debug is not None}")
    
    # 3. test_cte_tuner.py 中的 CTEEstimator (返回 mask)
    tuner_estimator = TunerCTEEstimator(
        method=method,
        max_cte=max_cte,
    )
    tuner_estimator.track_lower = np.array(track_lower)
    tuner_estimator.track_upper = np.array(track_upper)
    tuner_cte, tuner_conf, tuner_mask = tuner_estimator.estimate(frame_bgr)
    tuner_debug = tuner_estimator.last_debug_image
    print(f"\n[TunerCTEEstimator (test_cte_tuner.py)]")
    print(f"  CTE: {tuner_cte:.3f}")
    print(f"  Confidence: {tuner_conf:.3f}")
    print(f"  Has mask: {tuner_mask is not None}")
    if tuner_mask is not None:
        print(f"  Mask shape: {tuner_mask.shape}, dtype: {tuner_mask.dtype}")
        print(f"  Mask non-zero pixels: {np.count_nonzero(tuner_mask)}")
    print(f"  Has debug image: {tuner_debug is not None}")
    
    # 对比结果
    print(f"\n{'='*60}")
    print("对比结果:")
    print(f"{'='*60}")
    print(f"CTE 差异:")
    print(f"  Base vs Visual: {abs(base_cte - visual_cte):.6f}")
    print(f"  Base vs Tuner: {abs(base_cte - tuner_cte):.6f}")
    print(f"  Visual vs Tuner: {abs(visual_cte - tuner_cte):.6f}")
    print(f"\nConfidence 差异:")
    print(f"  Base vs Visual: {abs(base_conf - visual_conf):.6f}")
    print(f"  Base vs Tuner: {abs(base_conf - tuner_conf):.6f}")
    print(f"  Visual vs Tuner: {abs(visual_conf - tuner_conf):.6f}")
    
    if visual_mask is not None and tuner_mask is not None:
        # 检查 mask 是否相同
        if visual_mask.shape == tuner_mask.shape:
            mask_diff = np.abs(visual_mask.astype(np.int32) - tuner_mask.astype(np.int32))
            diff_pixels = np.count_nonzero(mask_diff)
            print(f"\nMask 差异:")
            print(f"  Shape 相同: True")
            print(f"  不同像素数: {diff_pixels} / {visual_mask.size}")
            print(f"  差异百分比: {diff_pixels / visual_mask.size * 100:.2f}%")
            if diff_pixels > 0:
                print(f"  最大差异值: {mask_diff.max()}")
        else:
            print(f"\nMask 差异:")
            print(f"  Shape 不同: Visual {visual_mask.shape} vs Tuner {tuner_mask.shape}")
    
    # 可视化对比
    return {
        "frame": frame_bgr,
        "base": {"cte": base_cte, "conf": base_conf},
        "visual": {
            "cte": visual_cte,
            "conf": visual_conf,
            "mask": visual_mask,
            "debug": visual_debug,
        },
        "tuner": {
            "cte": tuner_cte,
            "conf": tuner_conf,
            "mask": tuner_mask,
            "debug": tuner_debug,
        },
    }


def visualize_comparison(results: dict, save_path: Optional[str] = None):
    """可视化对比结果"""
    frame = results["frame"]
    visual = results["visual"]
    tuner = results["tuner"]
    
    # 计算需要的子图数量
    n_plots = 1  # 原始图像
    if visual["mask"] is not None:
        n_plots += 1
    if tuner["mask"] is not None:
        n_plots += 1
    if visual["debug"] is not None:
        n_plots += 1
    if tuner["debug"] is not None:
        n_plots += 1
    
    fig, axes = plt.subplots(2, max(3, (n_plots + 1) // 2), figsize=(20, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    
    # 1. 原始图像
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    axes[plot_idx].imshow(frame_rgb)
    axes[plot_idx].set_title("Original Frame")
    axes[plot_idx].axis("off")
    plot_idx += 1
    
    # 2. VisualCTEEstimator mask
    if visual["mask"] is not None:
        axes[plot_idx].imshow(visual["mask"], cmap="gray")
        axes[plot_idx].set_title(
            f"VisualCTEEstimator Mask\nCTE: {visual['cte']:.3f}, Conf: {visual['conf']:.3f}"
        )
        axes[plot_idx].axis("off")
        plot_idx += 1
    
    # 3. TunerCTEEstimator mask
    if tuner["mask"] is not None:
        axes[plot_idx].imshow(tuner["mask"], cmap="gray")
        axes[plot_idx].set_title(
            f"TunerCTEEstimator Mask\nCTE: {tuner['cte']:.3f}, Conf: {tuner['conf']:.3f}"
        )
        axes[plot_idx].axis("off")
        plot_idx += 1
    
    # 4. Mask 差异（如果有）
    if visual["mask"] is not None and tuner["mask"] is not None:
        if visual["mask"].shape == tuner["mask"].shape:
            mask_diff = np.abs(
                visual["mask"].astype(np.int32) - tuner["mask"].astype(np.int32)
            )
            axes[plot_idx].imshow(mask_diff, cmap="hot")
            axes[plot_idx].set_title("Mask Difference (Visual - Tuner)")
            axes[plot_idx].axis("off")
            plot_idx += 1
    
    # 5. VisualCTEEstimator debug
    if visual["debug"] is not None:
        debug_rgb = cv2.cvtColor(visual["debug"], cv2.COLOR_BGR2RGB)
        axes[plot_idx].imshow(debug_rgb)
        axes[plot_idx].set_title("VisualCTEEstimator Debug")
        axes[plot_idx].axis("off")
        plot_idx += 1
    
    # 6. TunerCTEEstimator debug
    if tuner["debug"] is not None:
        debug_rgb = cv2.cvtColor(tuner["debug"], cv2.COLOR_BGR2RGB)
        axes[plot_idx].imshow(debug_rgb)
        axes[plot_idx].set_title("TunerCTEEstimator Debug")
        axes[plot_idx].axis("off")
        plot_idx += 1
    
    # 隐藏多余的子图
    for i in range(plot_idx, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n对比图像已保存到: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="CTE Estimator 调试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", "-i", type=str, help="输入图片路径")
    parser.add_argument("--dir", "-d", type=str, help="图片目录路径")
    parser.add_argument("--camera", "-c", action="store_true", help="使用摄像头")
    parser.add_argument("--camera-id", type=int, default=0, help="摄像头 ID (默认: 0)")
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default="color_edge_detection",
        choices=["canny_edges", "color_edge_detection", "centerline_tracking"],
        help="CTE 估算方法",
    )
    parser.add_argument(
        "--h-low", type=int, default=0, help="HSV H 下限 (默认: 0)"
    )
    parser.add_argument(
        "--h-high", type=int, default=180, help="HSV H 上限 (默认: 180)"
    )
    parser.add_argument(
        "--s-low", type=int, default=100, help="HSV S 下限 (默认: 0)"
    )
    parser.add_argument(
        "--s-high", type=int, default=255, help="HSV S 上限 (默认: 30)"
    )
    parser.add_argument(
        "--v-low", type=int, default=100, help="HSV V 下限 (默认: 200)"
    )
    parser.add_argument(
        "--v-high", type=int, default=255, help="HSV V 上限 (默认: 255)"
    )
    parser.add_argument(
        "--max-cte", type=float, default=3.0, help="最大 CTE 值 (默认: 3.0)"
    )
    parser.add_argument(
        "--save", "-s", type=str, help="保存对比图像到文件"
    )
    
    args = parser.parse_args()
    
    # 准备 HSV 阈值
    track_lower = (args.h_low, args.s_low, args.v_low)
    track_upper = (args.h_high, args.s_high, args.v_high)
    
    # 加载图像
    frame_bgr = None
    
    if args.camera:
        camera = cv2.VideoCapture(args.camera_id)
        if not camera.isOpened():
            print(f"无法打开摄像头: {args.camera_id}")
            sys.exit(1)
        ret, frame_bgr = camera.read()
        camera.release()
        if not ret:
            print("无法从摄像头读取图像")
            sys.exit(1)
    elif args.dir:
        path = Path(args.dir)
        extensions = (".jpg", ".jpeg", ".png", ".bmp")
        image_paths = sorted(
            [f for f in path.iterdir() if f.suffix.lower() in extensions]
        )
        if not image_paths:
            print(f"目录中没有找到图片: {args.dir}")
            sys.exit(1)
        frame_bgr = cv2.imread(str(image_paths[0]))
        print(f"加载图片: {image_paths[0]}")
    elif args.image:
        frame_bgr = cv2.imread(args.image)
        if frame_bgr is None:
            print(f"无法加载图片: {args.image}")
            sys.exit(1)
    else:
        # 默认尝试加载 real_road_data 目录
        default_dirs = ["real_road_data", "data/road"]
        loaded = False
        for dir_path in default_dirs:
            if Path(dir_path).is_dir():
                path = Path(dir_path)
                extensions = (".jpg", ".jpeg", ".png", ".bmp")
                image_paths = sorted(
                    [f for f in path.iterdir() if f.suffix.lower() in extensions]
                )
                if image_paths:
                    frame_bgr = cv2.imread(str(image_paths[0]))
                    print(f"加载图片: {image_paths[0]}")
                    loaded = True
                    break
        if not loaded:
            print("请指定图片路径: --image 或 --dir 或 --camera")
            print("示例: python debug_cte_estimator.py --image real_road_data/pic0.jpg")
            sys.exit(1)
    
    if frame_bgr is None:
        print("无法获取图像")
        sys.exit(1)
    
    print(f"图像尺寸: {frame_bgr.shape}")
    
    # 对比估计器
    results = compare_estimators(
        frame_bgr,
        method=args.method,
        track_lower=track_lower,
        track_upper=track_upper,
        max_cte=args.max_cte,
    )
    
    # 可视化
    visualize_comparison(results, save_path=args.save)


if __name__ == "__main__":
    main()

