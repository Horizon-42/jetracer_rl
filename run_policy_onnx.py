import argparse
from ast import arg
import time
import sys
import atexit
import signal
import threading
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

    def apply(self, throttle: float, steering: float, log: bool = False):
        if self._car:
            # Clip to valid range [-1.0, 1.0], gain/offset applied by NvidiaRacecar
            clipped_throttle = float(np.clip(throttle, -1.0, 1.0))
            clipped_steering = float(np.clip(steering, -1.0, 1.0))
            self._car.throttle = clipped_throttle
            self._car.steering = clipped_steering
            if log:
                print(f"[Actuator] Applied: throttle={clipped_throttle:.4f}, steering={clipped_steering:.4f} (raw: throttle={throttle:.4f}, steering={steering:.4f})")
        elif log:
            print(f"[Actuator] Mock mode - Would apply: throttle={throttle:.4f}, steering={steering:.4f}")

    def stop(self):
        if self._car:
            self._car.throttle = 0.0

# --- 2. Image Preprocessing ---
def _get_perspective_transform_matrix(cam_width: int, cam_height: int):
    """Get perspective transform matrix for bird's-eye view transformation.
    
    This matches the perspective transform used in training (donkey_rl.obs_preprocess.ObsPreprocess).
    The source and destination points are designed for 320x240 images.
    Source points are scaled based on actual camera dimensions, but output size is fixed to (320, 240)
    to match training configuration.
    
    Args:
        cam_width: Camera image width
        cam_height: Camera image height
    
    Returns:
        Tuple of (perspective_transform_matrix, output_size)
        - perspective_transform_matrix: 3x3 numpy array
        - output_size: Fixed tuple (320, 240) to match training
    """
    # Original design size for 320x240 image (as in ObsPreprocess)
    design_width, design_height = 320, 240
    
    # Destination points (fixed output size, matches training)
    # These define where source points map to in the transformed image
    dst_pts = np.array([
        [10, 10],    # Top-left
        [310, 10],   # Top-right
        [310, 230],  # Bottom-right
        [10, 230]    # Bottom-left
    ], dtype=np.float32)
    
    # Source points define the region of interest in the original image
    # These are designed for 320x240, so scale if camera size differs
    src_pts = np.array([
        [75, 154],   # Top-left
        [242, 154],  # Top-right
        [319, 238],  # Bottom-right
        [0, 238]     # Bottom-left
    ], dtype=np.float32)
    
    # Scale source points if camera dimensions differ from design size
    if cam_width != design_width or cam_height != design_height:
        scale_x = cam_width / design_width
        scale_y = cam_height / design_height
        src_pts = src_pts * np.array([scale_x, scale_y], dtype=np.float32)
    
    # Compute perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Output size is always (320, 240) to match training configuration
    return matrix, (design_width, design_height)


def preprocess_image(frame_bgr, model_width, model_height, obs_mode="raw", perspective_matrix=None, perspective_size=None):
    """
    Preprocess camera frame for model inference.
    
    This matches the preprocessing pipeline used in training (donkey_rl.obs_preprocess.ObsPreprocess):
    1. Apply perspective transformation (if needed by obs_mode)
    2. Combine images based on obs_mode (raw, perspective, or mix)
    3. Resize to target dimensions
    4. Convert BGR to RGB
    5. Convert HWC to CHW format (channels-first)
    6. Normalize to [0, 1] range
    
    Args:
        frame_bgr: Input frame in BGR format (HWC uint8)
        model_width: Target width for model input
        model_height: Target height for model input
        obs_mode: Observation mode ('raw', 'perspective', 'mix')
            - 'raw': Use only raw camera image
            - 'perspective': Use only perspective-transformed (bird's-eye view) image
            - 'mix': Stack raw + perspective vertically, then resize to target dimensions
        perspective_matrix: Precomputed perspective transform matrix (3x3)
        perspective_size: Output size for perspective transform (width, height)
    
    Returns:
        Preprocessed image ready for model inference (1CHW float32 in [0, 1])
    """
    # Ensure frame is in correct format (uint8)
    if frame_bgr.dtype != np.uint8:
        frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)
    
    # Step 1: Apply perspective transformation (if needed)
    transformed = None
    if obs_mode in ("perspective", "mix") and perspective_matrix is not None and perspective_size is not None:
        transformed = cv2.warpPerspective(
            frame_bgr,
            perspective_matrix,
            perspective_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),  # Black padding
        )
    
    # Step 2: Combine images based on obs_mode
    if obs_mode == "raw":
        to_resize = frame_bgr
    elif obs_mode == "perspective":
        to_resize = transformed
    elif obs_mode == "mix":
        # Stack raw + perspective vertically
        # First resize raw to match perspective size for proper stacking
        raw_resized = cv2.resize(frame_bgr, perspective_size, interpolation=cv2.INTER_AREA)
        # Stack vertically: [raw on top, perspective on bottom]
        to_resize = np.vstack([raw_resized, transformed])
    else:
        # Fallback to raw
        to_resize = frame_bgr
    
    # Step 3: Resize to target dimensions (using INTER_AREA for downsampling)
    img = cv2.resize(to_resize, (model_width, model_height), interpolation=cv2.INTER_AREA)
    
    # Step 4: Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Step 5: Convert HWC to CHW format (channels-first)
    img = img.transpose((2, 0, 1))
    
    # Step 6: Normalize to [0, 1] range and add batch dimension
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img[np.newaxis, ...]

# --- 3. Global Cleanup Function ---
camera_resource = None
actuator_resource = None
shutdown_flag = threading.Event()
_cleanup_done = False

def cleanup_resources():
    global _cleanup_done
    if _cleanup_done:
        return  # Avoid double cleanup
    _cleanup_done = True
    """Cleanup resources on exit (normal or exception)"""
    print("\n[Cleanup] Releasing resources...")
    
    # Stop the car first
    if actuator_resource:
        try:
            actuator_resource.stop()
            print("- Car stopped")
        except Exception as e:
            print(f"- Car stop error: {e}")

    # Release camera (most important step - needs careful cleanup)
    if camera_resource:
        try:
            # Stop jetcam background thread
            if hasattr(camera_resource, 'running'):
                camera_resource.running = False
            
            # Wait longer for thread to notice the flag and finish
            # This is critical for preventing segfault
            time.sleep(0.3)
            
            # Try to join background thread if it exists
            if hasattr(camera_resource, 'thread') and camera_resource.thread is not None:
                try:
                    if camera_resource.thread.is_alive():
                        camera_resource.thread.join(timeout=0.5)
                except:
                    pass
            
            # Release OpenCV handle if it exists
            if hasattr(camera_resource, 'cap'):
                if camera_resource.cap is not None:
                    try:
                        camera_resource.cap.release()
                    except:
                        pass
            
            # For CSICamera, try to access and clean up the camera object
            # Some jetcam versions store the actual camera in different attributes
            if hasattr(camera_resource, 'camera'):
                try:
                    if camera_resource.camera is not None:
                        # Try to release the camera if it has a release method
                        if hasattr(camera_resource.camera, 'release'):
                            camera_resource.camera.release()
                        elif hasattr(camera_resource.camera, 'stop'):
                            camera_resource.camera.stop()
                except:
                    pass
            
            # Additional wait to ensure GStreamer pipeline is fully closed
            time.sleep(0.2)
            
            print("- Camera released")
        except Exception as e:
            print(f"- Camera release error: {e}")
            import traceback
            traceback.print_exc()

# Register cleanup function
atexit.register(cleanup_resources)

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print("\n[Signal] Interrupt received, shutting down gracefully...")
    shutdown_flag.set()

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- 4. Main Program ---
def main():
    global camera_resource, actuator_resource

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--cam-type", type=str, default="csi", choices=["csi", "usb"])
    parser.add_argument("--cam-idx", type=int, default=0)
    parser.add_argument("--cam-width", type=int, default=320)
    parser.add_argument("--cam-height", type=int, default=240)
    parser.add_argument("--obs-width", type=int, default=84, help="Model input width (should match training)")
    parser.add_argument("--obs-height", type=int, default=84, help="Model input height (should match training)")
    parser.add_argument(
        "--perspective-transform",
        action="store_false",
        default=True,
        help="Enable perspective transformation (bird's-eye view) preprocessing. Deprecated: use --obs-mode instead.",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="mix",
        choices=["auto", "raw", "perspective", "mix"],
        help="Observation mode: 'auto' (use --perspective-transform flag), 'raw' (original image only), "
             "'perspective' (bird's-eye view only), 'mix' (stack raw+perspective vertically, compress to 84x84).",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Target FPS for control loop")
    parser.add_argument("--throttle-gain", type=float, default=0.5, help="Throttle gain: output = gain * throttle (default: 1.0)")
    parser.add_argument("--throttle-boost", type=float, default=0.5, help="Throttle boost: output = gain *(boost+throttle)")
    parser.add_argument("--throttle-scale", type=float, default=0.02, help="Throttle  scale: output = gain *(boost+scale*throttle)")
    
    parser.add_argument("--steering-gain", type=float, default=0.5, help="Steering gain: output = gain * steering + offset (default: 1.0)")
    parser.add_argument("--steering-offset", type=float, default=0.4, help="Steering offset to correct mechanical bias (default: 0.0)")
    parser.add_argument("--log-interval", type=int, default=10, help="Print log every N frames (default: 10, 0 to disable)")
    args = parser.parse_args()

    # Initialize model
    print(f"Loading ONNX: {args.model}...")
    try:
        sess = ort.InferenceSession(args.model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        output_name = sess.get_outputs()[0].name if len(sess.get_outputs()) > 0 else "unknown"
        output_shape = sess.get_outputs()[0].shape if len(sess.get_outputs()) > 0 else "unknown"
        print(f"Model loaded successfully:")
        print(f"  - Input: name={input_name}, shape={input_shape}, dtype={sess.get_inputs()[0].type}")
        print(f"  - Output: name={output_name}, shape={output_shape}, dtype={sess.get_outputs()[0].type}")
    except Exception as e:
        print(f"Error loading ONNX: {e}")
        return

    # Determine obs_mode
    obs_mode = args.obs_mode.lower().strip()
    if obs_mode == "auto":
        # Backward compatibility: use perspective_transform flag
        obs_mode = "perspective" if args.perspective_transform else "raw"
    
    # Initialize perspective transform (if needed by obs_mode)
    perspective_matrix = None
    perspective_size = None
    if obs_mode in ("perspective", "mix"):
        perspective_matrix, perspective_size = _get_perspective_transform_matrix(
            args.cam_width, args.cam_height
        )
        print(f"Observation mode: {obs_mode}")
        print(f"  - Perspective transform: matrix shape={perspective_matrix.shape}, output size={perspective_size}")
    else:
        print(f"Observation mode: {obs_mode} (no perspective transform)")

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
    print(f"Log interval: {args.log_interval} frames (0 = disabled)")

    frame_count = 0
    try:
        while not shutdown_flag.is_set():
            t0 = time.time()
            
            # Check shutdown flag before blocking operations
            if shutdown_flag.is_set():
                break
            
            # read() is non-blocking, gets the latest frame
            try:
                frame = camera_resource.read()
            except Exception as e:
                if not shutdown_flag.is_set():
                    print(f"Camera read error: {e}")
                break
            
            if frame is None:
                if shutdown_flag.is_set():
                    break
                print("No frame")
                continue

            # Preprocess and inference
            obs = preprocess_image(
                frame,
                args.obs_width,
                args.obs_height,
                obs_mode=obs_mode,
                perspective_matrix=perspective_matrix,
                perspective_size=perspective_size,
            )
            action_raw = sess.run(None, {input_name: obs})[0].flatten()

            # add extra scale throttle
            # action_raw[0] = action_raw[0] * 5

            # clip steering to -1.0 to 1.0
            action_steering = np.clip(action_raw[1], -1.0, 1.0)
            # clip throttle to 0.0 to 1.0
            action_input_throttle = np.clip(action_raw[0], 0.0, 1.0)
            # add throttle boost
            action_throttle = 0
            if action_input_throttle > 0:
                action_throttle += args.throttle_boost
                action_throttle += action_input_throttle*args.throttle_scale
            
            # Log model output
            should_log = args.log_interval > 0 and (frame_count % args.log_interval == 0)
            if should_log:
                print(f"[Frame {frame_count}] Model raw output: action={action_raw}, shape={action_raw.shape}, dtype={action_raw.dtype}")
                print(f"  - Action values: throttle={action_throttle:.6f}, steering={action_steering:.6f}")
            
            # Execute action
            actuator_resource.apply(action_throttle, action_steering, log=should_log)
            
            frame_count += 1

            # Maintain target FPS
            t_process = time.time() - t0
            if t_process < dt:
                time.sleep(dt - t_process)

    except KeyboardInterrupt:
        print("\nStopping by user...")
        shutdown_flag.set()
    finally:
        # Ensure cleanup is called before exit
        # This helps prevent segfault by cleaning up resources before Python exits
        try:
            cleanup_resources()
        except Exception as e:
            print(f"Error during cleanup: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()