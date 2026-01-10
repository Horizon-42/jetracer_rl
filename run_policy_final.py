import argparse
import time
import sys
import atexit
import numpy as np
import cv2
import onnxruntime as ort

# --- 1. Robust Camera Class (Replaces JetCam) ---
class ReliableCSICamera:
    def __init__(self, capture_width=1280, capture_height=720, out_width=224, out_height=224, fps=120):
        # This pipeline mimics exactly what worked in your nvgstcapture log (Mode 5)
        # We capture at 1280x720 @ 120fps, then downscale to model input size (e.g., 224x224)
        self.gst_str = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
            f"format=NV12, framerate={fps}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width={out_width}, height={out_height}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! appsink drop=true sync=false"
        )
        
        print(f"Opening Camera with pipeline:\n{self.gst_str}")
        self.cap = cv2.VideoCapture(self.gst_str, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open CSI Camera via OpenCV")
            
    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

# --- 2. Hardware Control (JetRacer) ---
class JetRacerActuator:
    def __init__(self):
        self._car = None
        try:
            from jetracer.nvidia_racecar import NvidiaRacecar
            self._car = NvidiaRacecar()
        except ImportError:
            print("Warning: 'jetracer' not found. Running in Mock mode.")

    def apply(self, throttle: float, steering: float):
        if self._car:
            self._car.steering = float(np.clip(steering, -1.0, 1.0))
            self._car.throttle = float(np.clip(throttle, -0.5, 1.0))

    def stop(self):
        if self._car:
            self._car.throttle = 0.0

# --- 3. Image Preprocessing ---
def preprocess_image(frame_bgr, model_width, model_height):
    # Resize (if camera output isn't already exact match)
    if frame_bgr.shape[0] != model_height or frame_bgr.shape[1] != model_width:
        img = cv2.resize(frame_bgr, (model_width, model_height))
    else:
        img = frame_bgr
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img[np.newaxis, ...]

# --- 4. Global Cleanup ---
camera = None
actuator = None

def cleanup():
    print("\n[Cleanup] Stopping car and releasing camera...")
    if actuator: actuator.stop()
    if camera: camera.release()

atexit.register(cleanup)

# --- 5. Main Loop ---
def main():
    global camera, actuator
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    # Camera settings to feed into the pipeline
    parser.add_argument("--cam-width", type=int, default=224, help="Resize inside GStreamer to this width")
    parser.add_argument("--cam-height", type=int, default=224, help="Resize inside GStreamer to this height")
    # Model input settings
    parser.add_argument("--obs-width", type=int, default=84)
    parser.add_argument("--obs-height", type=int, default=84)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    # 1. Load Model
    print(f"Loading Model: {args.model}")
    try:
        sess = ort.InferenceSession(args.model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
    except Exception as e:
        print(f"Error loading ONNX: {e}")
        return

    # 2. Init Car
    actuator = JetRacerActuator()

    # 3. Init Camera (Direct OpenCV)
    try:
        # We output 224x224 from GStreamer directly to save Python CPU usage
        camera = ReliableCSICamera(out_width=args.cam_width, out_height=args.cam_height)
        
        # Warmup: Read 5 frames to let auto-exposure settle
        print("Warming up camera...")
        for _ in range(5):
            camera.read()
            time.sleep(0.1)
        print("Camera Ready!")
        
    except Exception as e:
        print(f"Camera Error: {e}")
        sys.exit(1)

    dt = 1.0 / args.fps
    print("Running Policy... Press Ctrl+C to stop.")

    try:
        while True:
            t0 = time.time()
            
            # Read Frame
            frame = camera.read()
            if frame is None:
                print("Lost frame from camera!")
                continue

            # Preprocess
            obs = preprocess_image(frame, args.obs_width, args.obs_height)
            
            # Inference
            outputs = sess.run(None, {input_name: obs})
            action = outputs[0].flatten() # [throttle, steering]
            
            # Act
            actuator.apply(action[0], action[1])

            # Timing
            t_process = time.time() - t0
            if t_process < dt:
                time.sleep(dt - t_process)

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()