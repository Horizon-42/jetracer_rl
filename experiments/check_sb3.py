import os
import cv2
import csv
import numpy as np
import argparse
from stable_baselines3 import PPO

def preprocess_image(img_path, width, height):
    # 读取图片
    frame_bgr = cv2.imread(img_path)
    if frame_bgr is None:
        return None
    
    # 1. Resize
    img = cv2.resize(frame_bgr, (width, height))
    # 2. BGR 转 RGB (SB3 默认是在 RGB 图像上训练的)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 3. HWC -> CHW
    img = img.transpose((2, 0, 1))
    # 4. 归一化 [0, 255] -> [0, 1] float32
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .zip model")
    parser.add_argument("--folder", type=str, required=True, help="Path to image folder")
    parser.add_argument("--output", type=str, default="result_sb3.csv", help="Output CSV filename")
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--height", type=int, default=84)
    args = parser.parse_args()

    print(f"Loading SB3 Model: {args.model}")
    # 强制 CPU 运行以减少与 ONNX CPU 推理时的浮点差异
    model = PPO.load(args.model, device="cpu")

    image_files = sorted([f for f in os.listdir(args.folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"Processing {len(image_files)} images...")
    
    with open(args.output, mode='w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['filename', 'throttle', 'steering'])

        for img_name in image_files:
            img_path = os.path.join(args.folder, img_name)
            
            obs = preprocess_image(img_path, args.width, args.height)
            if obs is None:
                continue

            # deterministic=True 确保输出的是均值（Mean），而不是采样
            action, _ = model.predict(obs, deterministic=True)
            act = action.flatten()
            
            # 写入一行
            writer.writerow([img_name, f"{act[0]:.8f}", f"{act[1]:.8f}"])

    print(f"Done! Results saved to {args.output}")

if __name__ == "__main__":
    main()