import argparse
import time
import sys
import numpy as np
import cv2
import onnxruntime as ort

# --- 硬件控制类 ---
class JetRacerActuator:
    def __init__(self):
        try:
            from jetracer.nvidia_racecar import NvidiaRacecar
            self._car = NvidiaRacecar()
        except ImportError:
            print("Error: 'jetracer' library not found. Running in dummy mode.")
            self._car = None

    def apply(self, throttle: float, steering: float):
        if self._car:
            # 限制范围，保护硬件
            self._car.steering = float(np.clip(steering, -1.0, 1.0))
            self._car.throttle = float(np.clip(throttle, -0.5, 1.0))
        else:
            print(f"Mock Action -> Throttle: {throttle:.2f}, Steering: {steering:.2f}")

    def stop(self):
        if self._car:
            self._car.throttle = 0.0

# --- 图像预处理 (内联以减少依赖) ---
def preprocess_image(frame_bgr, width, height):
    """
    将 OpenCV 读取的 BGR 图像转换为模型需要的格式 (1, 3, H, W) float32 [0,1]
    """
    # 1. Resize
    img = cv2.resize(frame_bgr, (width, height))
    # 2. BGR 转 RGB (SB3 模型通常在 RGB 上训练)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 3. HWC -> CHW (3, H, W)
    img = img.transpose((2, 0, 1))
    # 4. 归一化到 [0, 1] 并增加 Batch 维度 -> (1, 3, H, W)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img[np.newaxis, ...]

# --- 主程序 ---
def main():
    parser = argparse.ArgumentParser(description="Run ONNX policy on JetRacer")
    parser.add_argument("--model", type=str, required=True, help="Path to .onnx model file")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--obs-width", type=int, default=84)
    parser.add_argument("--obs-height", type=int, default=84)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    # 1. 初始化 ONNX Runtime
    print(f"Loading ONNX model: {args.model}...")
    try:
        # 优先尝试 CUDA (GPU)，失败则回退到 CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(args.model, providers=providers)
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")
        sys.exit(1)

    # 获取输入节点的名称 (通常是 "input" 或 "observation")
    input_name = session.get_inputs()[0].name
    print(f"Model loaded. Input name: {input_name}, Device: {ort.get_device()}")

    # 2. 初始化硬件
    actuator = JetRacerActuator()
    
    # 3. 打开摄像头
    # Jetson Nano CSI 摄像头通常需要 GStreamer 字符串，USB 摄像头直接用 index
    gst_str = (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
        f"nvvidconv ! video/x-raw, width={320}, height={240}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )
    
    # 尝试打开 CSI 摄像头，失败则尝试 USB
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("CSI Camera not found, trying USB camera...")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("Failed to open any camera.")
            sys.exit(1)

    dt = 1.0 / args.fps
    print("Starting control loop. Press Ctrl+C to stop.")

    try:
        while True:
            start_time = time.time()

            # 读取帧
            ret, frame = cap.read()
            print(frame.shape)
            if not ret:
                print("Failed to read frame")
                continue

            # 预处理
            input_tensor = preprocess_image(frame, args.obs_width, args.obs_height)

            # 推理 (Inference)
            # run 返回一个列表，第一个元素通常是 Action
            outputs = session.run(None, {input_name: input_tensor})
            raw_action = outputs[0] # Shape usually (1, 2)

            # 解析动作 (假设输出是 [throttle, steering] 或 [steering, throttle])
            # 注意: SB3 PPO 默认连续动作输出通常是高斯分布的均值，无需再做随机采样
            action_vals = raw_action.flatten()
            throttle = action_vals[0]
            steering = action_vals[1]

            print(f"Action: throttle={throttle:.2f}, steering={steering:.2f}")

            # 执行
            actuator.apply(throttle, steering)

            # 循环频率控制
            process_time = time.time() - start_time
            if process_time < dt:
                time.sleep(dt - process_time)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        actuator.stop()
        cap.release()

if __name__ == "__main__":
    main()