import argparse
import time
import sys
import atexit
import numpy as np
import cv2
import onnxruntime as ort

# --- 1. 硬件控制类 ---
class JetRacerActuator:
    def __init__(self):
        self._car = None
        try:
            from jetracer.nvidia_racecar import NvidiaRacecar
            self._car = NvidiaRacecar()
        except ImportError:
            print("Warning: 'jetracer' not found. Mock mode.")

    def apply(self, throttle: float, steering: float):
        if self._car:
            self._car.steering = float(np.clip(steering, -1.0, 1.0))
            self._car.throttle = float(np.clip(throttle, -0.5, 1.0))

    def stop(self):
        if self._car:
            self._car.throttle = 0.0

# --- 2. 图像预处理 ---
def preprocess_image(frame_bgr, model_width, model_height):
    img = cv2.resize(frame_bgr, (model_width, model_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img[np.newaxis, ...]

# --- 3. 全局清理函数 ---
camera_resource = None
actuator_resource = None

def cleanup_resources():
    """在脚本退出时（无论正常还是异常）强制执行"""
    print("\n[Cleanup] Releasing resources...")
    
    # 1. 停车
    if actuator_resource:
        try:
            actuator_resource.stop()
            print("- Car stopped")
        except: pass

    # 2. 释放相机 (最重要的一步)
    if camera_resource:
        try:
            # 停止 jetcam 的后台线程
            if hasattr(camera_resource, 'running'):
                camera_resource.running = False
            
            # 释放 OpenCV 句柄
            if hasattr(camera_resource, 'cap'):
                if camera_resource.cap is not None:
                    camera_resource.cap.release()
            print("- Camera released")
        except Exception as e:
            print(f"- Camera release error: {e}")

# 注册清理函数
atexit.register(cleanup_resources)

# --- 4. 主程序 ---
def main():
    global camera_resource, actuator_resource

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--cam-type", type=str, default="csi", choices=["csi", "usb"])
    parser.add_argument("--cam-idx", type=int, default=0)
    parser.add_argument("--cam-width", type=int, default=224)
    parser.add_argument("--cam-height", type=int, default=224)
    parser.add_argument("--obs-width", type=int, default=84)
    parser.add_argument("--obs-height", type=int, default=84)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    # 初始化模型
    print(f"Loading ONNX: {args.model}...")
    try:
        sess = ort.InferenceSession(args.model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
    except Exception as e:
        print(f"Error loading ONNX: {e}")
        return

    # 初始化小车
    actuator_resource = JetRacerActuator()

    # 初始化相机
    print(f"Initializing {args.cam_type.upper()} Camera...")
    try:
        if args.cam_type == "csi":
            from jetcam.csi_camera import CSICamera
            # 显式指定 capture_device 0，防止默认值导致的不确定性
            camera_resource = CSICamera(width=args.cam_width, height=args.cam_height, capture_width=1280, capture_height=720, capture_fps=30)
        else:
            from jetcam.usb_camera import USBCamera
            camera_resource = USBCamera(width=args.cam_width, height=args.cam_height, capture_device=args.cam_idx)
        
        # 尝试读取第一帧，如果这里失败，会直接触发 cleanup
        camera_resource.read()
        print("Camera ready.")
        
    except Exception as e:
        print(f"Failed to open camera: {e}")
        # 这里不需要手动调用 cleanup，sys.exit 会触发 atexit
        sys.exit(1)

    dt = 1.0 / args.fps
    print("Running... Ctrl+C to stop.")

    try:
        while True:
            t0 = time.time()
            
            # 这里的 read() 是非阻塞的，获取最新帧
            frame = camera_resource.read()
            if frame is None:
                print("No frame")
                continue

            # 推理
            obs = preprocess_image(frame, args.obs_width, args.obs_height)
            action = sess.run(None, {input_name: obs})[0].flatten()
            
            # 执行
            actuator_resource.apply(action[0], action[1])

            # 延时
            t_process = time.time() - t0
            if t_process < dt:
                time.sleep(dt - t_process)

    except KeyboardInterrupt:
        print("\nStopping by user...")
        # 退出时会自动调用 cleanup_resources()

if __name__ == "__main__":
    main()