import os
import cv2
import csv
import numpy as np
import argparse
import onnxruntime as ort

def preprocess_image(img_path, width, height):
    # 保持与 SB3 脚本完全一致的预处理逻辑
    frame_bgr = cv2.imread(img_path)
    if frame_bgr is None:
        return None
    
    img = cv2.resize(frame_bgr, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    
    # ONNX 需要增加 Batch 维度 -> (1, 3, H, W)
    return img[np.newaxis, ...]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .onnx model")
    parser.add_argument("--folder", type=str, required=True, help="Path to image folder")
    parser.add_argument("--output", type=str, default="result_onnx.csv", help="Output CSV filename")
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--height", type=int, default=84)
    args = parser.parse_args()

    print(f"Loading ONNX Model: {args.model}")
    # 使用 CPU Provider 以消除 GPU 带来的非确定性浮点误差
    session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    image_files = sorted([f for f in os.listdir(args.folder) if f.endswith(('.jpg', '.jpeg', '.png'))])

    print(f"Processing {len(image_files)} images...")

    with open(args.output, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'throttle', 'steering'])

        for img_name in image_files:
            img_path = os.path.join(args.folder, img_name)
            
            input_tensor = preprocess_image(img_path, args.width, args.height)
            if input_tensor is None:
                continue

            # ONNX 推理
            outputs = session.run(None, {input_name: input_tensor})
            
            # 假设输出已经是 Mean (需要在导出时设置)
            raw_action = outputs[0].flatten()
            
            writer.writerow([img_name, f"{raw_action[0]:.8f}", f"{raw_action[1]:.8f}"])

    print(f"Done! Results saved to {args.output}")

if __name__ == "__main__":
    main()