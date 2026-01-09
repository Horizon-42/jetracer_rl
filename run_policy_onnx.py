import argparse
import time
import sys
import numpy as np
import cv2
import onnxruntime as ort

# --- 1. 硬件控制类 (保持不变) ---
class JetRacerActuator:
    def __init__(self):
        try:
            from jetracer.nvidia_racecar import NvidiaRacecar
            self._car = NvidiaRacecar()
        except ImportError:
            print("Warning: 'jetracer' library not found. Running in Mock/Dummy mode.")
            self._car = None

    def apply(self, throttle: float, steering: float):
        if self._car:
            self._car.steering = float(np.clip(steering, -1.0, 1.0))
            self._car.throttle = float(np.clip(throttle, -0.5, 1.0))
        else:
            # 仅打印，防止刷屏
            pass 

    def stop(self):
        if self._car:
            self._car.throttle = 0.0

# --- 2. 图像预处理 ---
def preprocess_image(frame_bgr, model_width, model_height):
    """
    输入: 从 jetcam 读取的 BGR 图像 (尺寸由 cam_width/cam_height 决定)
    输出: 模型需要的 (1, 3, 84, 84) float32
    """
    # 1. Resize 到模型输入大小 (例如 84x84)
    img = cv2.resize(frame_bgr, (model_width, model_height))
    # 2. BGR 转 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 3. HWC -> CHW
    img = img.transpose((2, 0, 1))
    # 4. 归一化并增加 Batch 维度
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img[np.newaxis, ...]

# --- 3. 核心主程序 ---
def main():
    parser = argparse.ArgumentParser(description="Run ONNX policy using JetCam")
    parser.add_argument("--model", type=str, required=True, help="Path to .onnx model")
    
    # 摄像头设置
    parser.add_argument("--cam-type", type=str, default="csi", choices=["csi", "usb"], help="Camera type: csi or usb")
    parser.add_argument("--cam-idx", type=int, default=0, help="Camera index (for USB typically 0 or 1)")
    parser.add_argument("--cam-width", type=int, default=320, help="Camera capture width")
    parser.add_argument("--cam-height", type=int, default=240, help="Camera capture height")
    
    # 模型设置
    parser.add_argument("--obs-width", type=int, default=84, help="Model input width")
    parser.add_argument("--obs-height", type=int, default=84, help="Model input height")
    parser.add_argument("--fps", type=float, default=20.0, help="Control loop FPS")
    
    args = parser.parse_args()

    # --- A. 初始化 ONNX Runtime ---
    print(f"Loading ONNX model: {args.model}...")
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(args.model, providers=providers)
    except Exception as e:
        print(f"Error loading ONNX: {e}")
        sys.exit(1)
    
    input_name = session.get_inputs()[0].name
    print(f"Model loaded. Device: {ort.get_device()}")

    # --- B. 初始化 JetCam (核心修改) ---
    print(f"Initializing {args.cam_type.upper()} Camera...")
    try:
        if args.cam_type == "csi":
            from jetcam.csi_camera import CSICamera
            # CSI 摄像头通常 capture_width 设为 1280x720 比较稳定，
            # width/height 是 jetcam 帮你 resize 后的输出大小
            camera = CSICamera(width=args.cam_width, height=args.cam_height, capture_width=320, capture_height=240, capture_fps=20)
        else:
            print("CSI Camera not found, trying USB camera...")
            from jetcam.usb_camera import USBCamera
            # USB 摄像头直接设置 width/height 即可
            camera = USBCamera(width=args.cam_width, height=args.cam_height, capture_device=args.cam_idx)
        
        # 预读取一帧以启动摄像头
        camera.read() 
        print(f"Camera started. Output resolution: {args.cam_width}x{args.cam_height}")
        
    except Exception as e:
        print(f"Failed to open camera: {e}")
        sys.exit(1)

    # --- C. 初始化小车 ---
    actuator = JetRacerActuator()
    dt = 1.0 / args.fps
    
    print("Running... Press Ctrl+C to stop.")
    
    try:
        while True:
            start_time = time.time()

            # 1. 获取图像 (JetCam 内部已经在后台线程持续读取，这里获取的是最新帧)
            frame = camera.read()
            
            if frame is None:
                print("No frame received")
                continue

            # 2. 预处理 (Resize 224 -> 84, BGR->RGB, Norm)
            input_tensor = preprocess_image(frame, args.obs_width, args.obs_height)

            # 3. 推理
            outputs = session.run(None, {input_name: input_tensor})
            raw_action = outputs[0].flatten() # [throttle, steering]

            throttle = raw_action[0]
            steering = raw_action[1]

            # 4. 执行动作
            actuator.apply(throttle, steering)
            
            # (可选) 显示简略日志
            # print(f"\rThr: {throttle:.2f} Str: {steering:.2f}", end="")

            # 5. 频率控制
            process_time = time.time() - start_time
            if process_time < dt:
                time.sleep(dt - process_time)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        actuator.stop()
        # JetCam 析构时通常会自动释放资源，但手动释放是个好习惯
        if hasattr(camera, 'cap'):
            camera.cap.release()

if __name__ == "__main__":
    main()