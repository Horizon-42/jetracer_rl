import argparse
import time
import sys
import atexit
import numpy as np
import cv2
import onnxruntime as ort

# --- 1. Hardware Control Class ---
class JetRacerActuator:
    def __init__(self, throttle_gain: float = 1.0, steering_gain: float = 1.0, steering_offset: float = 0.0):
        """
        Initialize JetRacer actuator.
        
        Args:
            throttle_gain: Scale factor for throttle (output = gain * throttle)
            steering_gain: Scale factor for steering (output = gain * steering + offset)
            steering_offset: Offset for steering to correct mechanical bias
        """
        self._car = None
        try:
            from jetracer.nvidia_racecar import NvidiaRacecar
            self._car = NvidiaRacecar()
            self._car.throttle_gain = throttle_gain
            self._car.steering_gain = steering_gain
            self._car.steering_offset = steering_offset
        except ImportError:
            print("Warning: 'jetracer' not found. Mock mode.")

    def apply(self, throttle: float, steering: float):
        if self._car:
            # Clip to valid range [-1.0, 1.0], gain/offset applied by NvidiaRacecar
            self._car.throttle = float(np.clip(throttle, -1.0, 1.0))
            self._car.steering = float(np.clip(steering, -1.0, 1.0))

    def stop(self):
        if self._car:
            self._car.throttle = 0.0

# --- 2. Image Preprocessing ---
def preprocess_image(frame_bgr, model_width, model_height):
    img = cv2.resize(frame_bgr, (model_width, model_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img[np.newaxis, ...]

# --- 3. Global Cleanup Function ---
camera_resource = None
actuator_resource = None

def cleanup_resources():
    """Cleanup resources on exit (normal or exception)"""
    print("\n[Cleanup] Releasing resources...")
    
    # Stop the car
    if actuator_resource:
        try:
            actuator_resource.stop()
            print("- Car stopped")
        except: pass

    # Release camera (most important step)
    if camera_resource:
        try:
            # Stop jetcam background thread
            if hasattr(camera_resource, 'running'):
                camera_resource.running = False
            
            # Release OpenCV handle
            if hasattr(camera_resource, 'cap'):
                if camera_resource.cap is not None:
                    camera_resource.cap.release()
            print("- Camera released")
        except Exception as e:
            print(f"- Camera release error: {e}")

# Register cleanup function
atexit.register(cleanup_resources)

# --- 4. Main Program ---
def main():
    global camera_resource, actuator_resource

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--cam-type", type=str, default="csi", choices=["csi", "usb"])
    parser.add_argument("--cam-idx", type=int, default=0)
    parser.add_argument("--cam-width", type=int, default=320)
    parser.add_argument("--cam-height", type=int, default=240)
    parser.add_argument("--obs-width", type=int, default=84)
    parser.add_argument("--obs-height", type=int, default=84)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--throttle-gain", type=float, default=1.0, help="Throttle gain: output = gain * throttle (default: 1.0)")
    parser.add_argument("--steering-gain", type=float, default=1.0, help="Steering gain: output = gain * steering + offset (default: 1.0)")
    parser.add_argument("--steering-offset", type=float, default=0.0, help="Steering offset to correct mechanical bias (default: 0.0)")
    args = parser.parse_args()

    # Initialize model
    print(f"Loading ONNX: {args.model}...")
    try:
        sess = ort.InferenceSession(args.model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
    except Exception as e:
        print(f"Error loading ONNX: {e}")
        return

    # Initialize car actuator
    actuator_resource = JetRacerActuator(
        throttle_gain=args.throttle_gain,
        steering_gain=args.steering_gain,
        steering_offset=args.steering_offset
    )
    print(f"Actuator configured: throttle_gain={args.throttle_gain}, steering_gain={args.steering_gain}, steering_offset={args.steering_offset}")

    # Initialize camera
    print(f"Initializing {args.cam_type.upper()} Camera...")
    try:
        if args.cam_type == "csi":
            from jetcam.csi_camera import CSICamera
            camera_resource = CSICamera(width=args.cam_width, height=args.cam_height, capture_width=1280, capture_height=720, capture_fps=30)
        else:
            from jetcam.usb_camera import USBCamera
            camera_resource = USBCamera(width=args.cam_width, height=args.cam_height, capture_device=args.cam_idx)
        
        # Try to read first frame, if this fails, cleanup will be triggered
        camera_resource.read()
        print("Camera ready.")
        
    except Exception as e:
        print(f"Failed to open camera: {e}")
        # No need to manually call cleanup, sys.exit will trigger atexit
        sys.exit(1)

    dt = 1.0 / args.fps
    print("Running... Ctrl+C to stop.")

    try:
        while True:
            t0 = time.time()
            
            # read() is non-blocking, gets the latest frame
            frame = camera_resource.read()
            if frame is None:
                print("No frame")
                continue

            # Inference
            obs = preprocess_image(frame, args.obs_width, args.obs_height)
            action = sess.run(None, {input_name: obs})[0].flatten()
            
            # Execute action
            actuator_resource.apply(action[0], action[1])

            # Maintain target FPS
            t_process = time.time() - t0
            if t_process < dt:
                time.sleep(dt - t_process)

    except KeyboardInterrupt:
        print("\nStopping by user...")
        # cleanup_resources() will be called automatically on exit

if __name__ == "__main__":
    main()