"""Real car Gym environment for JetRacer RL training.

This module provides a Gym environment for training RL policies directly
on a real JetRacer car. It estimates CTE (cross-track error) from camera images
using computer vision, enabling centerline-following reward functions.

Compatible with: Python 3.6+, gym 0.21.0, stable-baselines3 1.2

Key differences from simulation:
- CTE is estimated from camera images (not ground truth)
- Speed is estimated from throttle or wheel encoders
- No automatic reset on crash (requires manual intervention)

Usage:
    from real_car_env import RealJetRacerEnv
    
    env = RealJetRacerEnv(
        cte_estimator="edge_detection",  # or "centerline_tracking"
        max_cte=1.0,
    )
    obs = env.reset()
    
    for _ in range(1000):
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
"""

import time
from typing import Optional, Tuple, Dict, Any, Callable

import cv2
import gym
import numpy as np
from gym import spaces


class VisualCTEEstimator:
    """Estimate cross-track error (CTE) from camera images.
    
    This class provides multiple methods for estimating how far the car
    is from the track centerline based on camera images.
    """
    
    def __init__(
        self,
        method: str = "edge_detection",
        image_width: int = 320,
        image_height: int = 240,
        max_cte: float = 3.0,
        # Color thresholds for track detection (HSV)
        track_lower: Tuple[int, int, int] = (0, 0, 0),
        track_upper: Tuple[int, int, int] = (180, 50, 80),
        # Color thresholds for centerline detection (HSV)
        centerline_lower: Tuple[int, int, int] = (20, 100, 100),
        centerline_upper: Tuple[int, int, int] = (30, 255, 255),
    ):
        """Initialize the CTE estimator.
        
        Args:
            method: Estimation method ("edge_detection" or "centerline_tracking")
            image_width: Expected image width
            image_height: Expected image height
            max_cte: Maximum CTE value (for normalization)
            track_lower: HSV lower bound for track color
            track_upper: HSV upper bound for track color
            centerline_lower: HSV lower bound for centerline color
            centerline_upper: HSV upper bound for centerline color
        """
        self.method = method
        self.image_width = image_width
        self.image_height = image_height
        self.image_center = image_width // 2
        self.max_cte = max_cte
        
        self.track_lower = np.array(track_lower)
        self.track_upper = np.array(track_upper)
        self.centerline_lower = np.array(centerline_lower)
        self.centerline_upper = np.array(centerline_upper)
        
        # For visualization/debugging
        self.last_debug_image: Optional[np.ndarray] = None
    
    def estimate(self, frame_bgr: np.ndarray) -> Tuple[float, float]:
        """Estimate CTE from camera image.
        
        Args:
            frame_bgr: Camera image in BGR format (HWC uint8)
        
        Returns:
            Tuple of (cte, confidence):
            - cte: Cross-track error (positive = right of center)
            - confidence: Detection confidence (0.0 to 1.0)
        """
        if self.method == "edge_detection":
            return self._estimate_by_edges(frame_bgr)
        elif self.method == "centerline_tracking":
            return self._estimate_by_centerline(frame_bgr)
        else:
            return 0.0, 0.0
    
    def _estimate_by_edges(self, frame_bgr: np.ndarray) -> Tuple[float, float]:
        """Estimate CTE by detecting track edges.
        
        Detects left and right track boundaries, computes their midpoint,
        and returns the offset from image center as CTE.
        """
        h, w = frame_bgr.shape[:2]
        
        # Take lower portion of image (near the car)
        roi_start = int(h * 0.6)
        roi = frame_bgr[roi_start:, :]
        
        # Convert to grayscale and apply edge detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find edge points in bottom rows
        scan_row = edges.shape[0] - 10  # Near bottom
        if scan_row < 0:
            scan_row = edges.shape[0] // 2
        
        edge_pixels = np.where(edges[scan_row, :] > 0)[0]
        
        if len(edge_pixels) < 2:
            # Not enough edges detected
            return 0.0, 0.0
        
        # Assume leftmost and rightmost edges are track boundaries
        left_edge = edge_pixels[0]
        right_edge = edge_pixels[-1]
        
        # Compute lane center
        lane_center = (left_edge + right_edge) // 2
        
        # Compute CTE (normalized)
        pixel_offset = lane_center - (w // 2)
        cte = (pixel_offset / (w / 2)) * self.max_cte
        
        # Confidence based on edge separation
        edge_width = right_edge - left_edge
        expected_width = w * 0.5  # Expect track to be ~50% of image width
        confidence = min(1.0, edge_width / expected_width)
        
        # Debug visualization
        self.last_debug_image = roi.copy()
        cv2.line(self.last_debug_image, (left_edge, scan_row), (left_edge, scan_row - 20), (0, 255, 0), 2)
        cv2.line(self.last_debug_image, (right_edge, scan_row), (right_edge, scan_row - 20), (0, 255, 0), 2)
        cv2.line(self.last_debug_image, (lane_center, scan_row), (lane_center, scan_row - 30), (0, 0, 255), 2)
        
        return float(cte), float(confidence)
    
    def _estimate_by_centerline(self, frame_bgr: np.ndarray) -> Tuple[float, float]:
        """Estimate CTE by tracking a colored centerline.
        
        Detects a colored centerline (e.g., yellow tape) and returns
        its offset from image center as CTE.
        """
        h, w = frame_bgr.shape[:2]
        
        # Take lower portion of image
        roi_start = int(h * 0.5)
        roi = frame_bgr[roi_start:, :]
        
        # Convert to HSV and detect centerline color
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.centerline_lower, self.centerline_upper)
        
        # Find centroid of detected region
        moments = cv2.moments(mask)
        
        if moments["m00"] < 100:  # Not enough pixels detected
            return 0.0, 0.0
        
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        
        # Compute CTE
        pixel_offset = cx - (w // 2)
        cte = (pixel_offset / (w / 2)) * self.max_cte
        
        # Confidence based on detected area
        detected_area = moments["m00"]
        expected_area = w * (h - roi_start) * 0.05  # Expect ~5% of ROI
        confidence = min(1.0, detected_area / expected_area)
        
        # Debug visualization
        self.last_debug_image = roi.copy()
        cv2.circle(self.last_debug_image, (cx, cy), 10, (0, 0, 255), -1)
        cv2.line(self.last_debug_image, (w // 2, 0), (w // 2, roi.shape[0]), (255, 0, 0), 1)
        
        return float(cte), float(confidence)


class RealJetRacerEnv(gym.Env):
    """Gym environment for real JetRacer car (gym 0.21.0 compatible).
    
    This environment interfaces with a real JetRacer car and provides:
    - Camera observations (like simulation)
    - CTE estimation from camera images (replaces sim ground truth)
    - Speed estimation (from throttle or encoders)
    - Centerline-following rewards (like simulation)
    
    Note: Unlike simulation, the environment cannot automatically reset
    after crashes. You need to manually reposition the car.
    
    API: gym 0.21.0 (old API)
    - reset() returns: observation
    - step() returns: observation, reward, done, info
    """
    
    metadata = {"render.modes": ["human", "rgb_array"]}  # gym 0.21 format
    
    def __init__(
        self,
        # Camera settings
        cam_type: str = "csi",
        cam_width: int = 320,
        cam_height: int = 240,
        # Observation settings
        obs_width: int = 84,
        obs_height: int = 84,
        obs_mode: str = "perspective",
        # CTE estimation
        cte_estimator: str = "edge_detection",
        max_cte: float = 3.0,
        # Actuator settings
        throttle_gain: float = 0.5,
        steering_gain: float = 0.5,
        steering_offset: float = 0.0,
        # Episode settings
        max_episode_steps: int = 1000,
        # Reward settings (same as simulation)
        w_center: float = 1.0,
        w_speed: float = 1.0,
        alive_bonus: float = 0.03,
        offtrack_penalty: float = 50.0,
        # Safety
        emergency_stop_cte: float = 2.5,
        render_mode: Optional[str] = None,
    ):
        """Initialize the real car environment.
        
        Args:
            cam_type: Camera type ("csi" or "usb")
            cam_width: Camera capture width
            cam_height: Camera capture height
            obs_width: Observation width for policy
            obs_height: Observation height for policy
            obs_mode: Observation mode ("raw", "perspective", "mix")
            cte_estimator: CTE estimation method ("edge_detection" or "centerline_tracking")
            max_cte: Maximum CTE before episode termination
            throttle_gain: Throttle scaling factor
            steering_gain: Steering scaling factor
            steering_offset: Steering offset for mechanical bias
            max_episode_steps: Maximum steps per episode
            w_center: Centerline reward weight
            w_speed: Speed reward weight
            alive_bonus: Per-step survival bonus
            offtrack_penalty: Penalty for going off-track
            emergency_stop_cte: CTE threshold for emergency stop
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.cam_type = cam_type
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.obs_mode = obs_mode
        self.max_cte = max_cte
        self.throttle_gain = throttle_gain
        self.steering_gain = steering_gain
        self.steering_offset = steering_offset
        self.max_episode_steps = max_episode_steps
        self.emergency_stop_cte = emergency_stop_cte
        self.render_mode = render_mode
        
        # Reward settings
        self.w_center = w_center
        self.w_speed = w_speed
        self.alive_bonus = alive_bonus
        self.offtrack_penalty = offtrack_penalty
        
        # Initialize CTE estimator
        self.cte_estimator = VisualCTEEstimator(
            method=cte_estimator,
            image_width=cam_width,
            image_height=cam_height,
            max_cte=max_cte,
        )
        
        # Initialize perspective transform (same as simulation)
        self._init_perspective_transform()
        
        # Action space: [throttle, steering]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Observation space: CHW float32 [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, obs_height, obs_width),
            dtype=np.float32,
        )
        
        # State
        self._camera = None
        self._car = None
        self._step_count = 0
        self._last_throttle = 0.0
        self._last_steering = 0.0
        self._last_cte = 0.0
        self._last_speed = 0.0
        self._last_confidence = 0.0
        
    def _init_perspective_transform(self):
        """Initialize perspective transform matrix (same as training)."""
        src_pts = np.array([
            [75, 154], [242, 154], [319, 238], [0, 238]
        ], dtype=np.float32)
        dst_pts = np.array([
            [10, 10], [310, 10], [310, 230], [10, 230]
        ], dtype=np.float32)
        
        # Scale for camera size
        scale_x = self.cam_width / 320
        scale_y = self.cam_height / 240
        src_pts = src_pts * np.array([scale_x, scale_y], dtype=np.float32)
        
        self.perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.perspective_size = (320, 240)
    
    def _init_hardware(self):
        """Initialize camera and car actuator."""
        # Initialize camera
        if self._camera is None:
            try:
                if self.cam_type == "csi":
                    from jetcam.csi_camera import CSICamera
                    self._camera = CSICamera(
                        width=self.cam_width,
                        height=self.cam_height,
                        capture_width=1280,
                        capture_height=720,
                        capture_fps=30,
                    )
                else:
                    from jetcam.usb_camera import USBCamera
                    self._camera = USBCamera(
                        width=self.cam_width,
                        height=self.cam_height,
                    )
                self._camera.read()  # Warm up
                print(f"[RealJetRacerEnv] Camera initialized: {self.cam_type}")
            except Exception as e:
                print(f"[RealJetRacerEnv] Camera init failed: {e}")
                self._camera = None
        
        # Initialize car
        if self._car is None:
            try:
                from jetracer.nvidia_racecar import NvidiaRacecar
                self._car = NvidiaRacecar()
                self._car.throttle_gain = self.throttle_gain
                self._car.steering_gain = self.steering_gain
                self._car.steering_offset = self.steering_offset
                print(f"[RealJetRacerEnv] Car initialized")
            except Exception as e:
                print(f"[RealJetRacerEnv] Car init failed (mock mode): {e}")
                self._car = None
    
    def _get_observation(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Process camera frame into observation (matches training preprocessing)."""
        if self.obs_mode == "raw":
            to_resize = frame_bgr
        elif self.obs_mode == "perspective":
            to_resize = cv2.warpPerspective(
                frame_bgr,
                self.perspective_matrix,
                self.perspective_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        elif self.obs_mode == "mix":
            # Stack raw + perspective vertically
            raw_resized = cv2.resize(frame_bgr, self.perspective_size, interpolation=cv2.INTER_AREA)
            transformed = cv2.warpPerspective(
                frame_bgr,
                self.perspective_matrix,
                self.perspective_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            to_resize = np.vstack([raw_resized, transformed])
        else:
            to_resize = frame_bgr
        
        # Resize to observation size
        resized = cv2.resize(to_resize, (self.obs_width, self.obs_height), interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to CHW float32 [0, 1]
        obs = rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
        
        return obs
    
    def _compute_reward(self, cte: float, speed: float, confidence: float) -> Tuple[float, bool]:
        """Compute reward (same logic as training centerline rewards).
        
        Returns:
            Tuple of (reward, terminated)
        """
        abs_cte = abs(cte)
        
        # Check if off-track
        if abs_cte > self.max_cte:
            return -self.offtrack_penalty, True
        
        # Centerline score: 1.0 at center, 0.0 at edge
        center_score = max(0.0, 1.0 - (abs_cte / self.max_cte))
        
        # Reward components
        reward = 0.0
        reward += self.alive_bonus
        reward += self.w_center * center_score
        reward += self.w_speed * speed
        
        # Reduce reward if detection confidence is low
        if confidence < 0.5:
            reward *= confidence
        
        return float(reward), False
    
    def _apply_action(self, throttle: float, steering: float):
        """Apply action to car actuator."""
        if self._car is not None:
            self._car.throttle = float(np.clip(throttle, -1.0, 1.0))
            self._car.steering = float(np.clip(steering, -1.0, 1.0))
        
        self._last_throttle = throttle
        self._last_steering = steering
    
    def _emergency_stop(self):
        """Emergency stop the car."""
        if self._car is not None:
            self._car.throttle = 0.0
        self._last_throttle = 0.0
        print("[RealJetRacerEnv] EMERGENCY STOP!")
    
    def reset(self) -> np.ndarray:
        """Reset the environment.
        
        Note: For real car, this just stops the car and waits for user
        to reposition it. You may want to add a countdown or button press.
        
        Returns:
            observation: Initial observation (gym 0.21 API)
        """
        # Initialize hardware if not done
        self._init_hardware()
        
        # Stop the car
        self._apply_action(0.0, 0.0)
        
        # Reset state
        self._step_count = 0
        self._last_cte = 0.0
        self._last_speed = 0.0
        
        # Wait for user to position car (optional)
        print("[RealJetRacerEnv] Reset: Please position the car on the track.")
        print("[RealJetRacerEnv] Press Enter to start episode...")
        try:
            input()  # Wait for user
        except Exception:
            pass
        
        # Get initial observation
        if self._camera is not None:
            frame = self._camera.read()
        else:
            frame = np.zeros((self.cam_height, self.cam_width, 3), dtype=np.uint8)
        
        obs = self._get_observation(frame)
        
        # Estimate initial CTE (store for info access)
        cte, confidence = self.cte_estimator.estimate(frame)
        self._last_cte = cte
        self._last_confidence = confidence
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step.
        
        Returns:
            obs, reward, done, info (gym 0.21 API - 4 values)
        """
        self._step_count += 1
        
        # Parse action
        throttle = float(action[0])
        steering = float(action[1])
        
        # Apply action
        self._apply_action(throttle, steering)
        
        # Small delay for action to take effect
        time.sleep(0.02)  # 50 Hz max
        
        # Get camera frame
        if self._camera is not None:
            frame = self._camera.read()
        else:
            frame = np.zeros((self.cam_height, self.cam_width, 3), dtype=np.uint8)
        
        # Get observation
        obs = self._get_observation(frame)
        
        # Estimate CTE
        cte, confidence = self.cte_estimator.estimate(frame)
        self._last_cte = cte
        self._last_confidence = confidence
        
        # Estimate speed (simple: use throttle as proxy)
        speed = max(0.0, throttle * 1.0)  # Adjust scaling as needed
        self._last_speed = speed
        
        # Compute reward
        reward, terminated = self._compute_reward(cte, speed, confidence)
        
        # Check for emergency stop
        if abs(cte) > self.emergency_stop_cte:
            self._emergency_stop()
            terminated = True
            reward = -self.offtrack_penalty
        
        # Check for max steps (truncated in new API, combined into done for old API)
        truncated = self._step_count >= self.max_episode_steps
        
        # Combine terminated and truncated into single done flag (gym 0.21 API)
        done = terminated or truncated
        
        # Stop car if episode ends
        if done:
            self._apply_action(0.0, 0.0)
        
        info = {
            "cte": cte,
            "speed": speed,
            "confidence": confidence,
            "step": self._step_count,
            "throttle": throttle,
            "steering": steering,
            "TimeLimit.truncated": truncated and not terminated,  # SB3 convention
        }
        
        return obs, reward, done, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self.cte_estimator.last_debug_image
        elif self.render_mode == "human":
            if self.cte_estimator.last_debug_image is not None:
                cv2.imshow("CTE Debug", self.cte_estimator.last_debug_image)
                cv2.waitKey(1)
    
    def close(self):
        """Clean up resources."""
        # Stop car
        if self._car is not None:
            self._car.throttle = 0.0
        
        # Close camera
        if self._camera is not None:
            if hasattr(self._camera, 'running'):
                self._camera.running = False
            time.sleep(0.2)
        
        cv2.destroyAllWindows()


# ============================================================================
# Training Script Example
# ============================================================================

if __name__ == "__main__":
    """Example: Train on real car (or test the environment)."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test mode (no training)")
    parser.add_argument("--cte-method", type=str, default="edge_detection",
                        choices=["edge_detection", "centerline_tracking"])
    parser.add_argument("--obs-mode", type=str, default="mix",
                        choices=["raw", "perspective", "mix"])
    args = parser.parse_args()
    
    env = RealJetRacerEnv(
        cte_estimator=args.cte_method,
        obs_mode=args.obs_mode,
        render_mode="human",
        max_episode_steps=500,
    )
    
    if args.test:
        # Test mode: manual control to verify CTE estimation
        print("=== Test Mode ===")
        print("This will display CTE estimation. Press Ctrl+C to stop.")
        
        obs = env.reset()  # gym 0.21 API: returns only obs
        
        try:
            while True:
                # Zero action (car stationary)
                action = np.array([0.0, 0.0], dtype=np.float32)
                obs, reward, done, info = env.step(action)  # gym 0.21 API: 4 values
                
                print("CTE: {:.3f}, Confidence: {:.2f}, Reward: {:.3f}".format(
                    info['cte'], info['confidence'], reward))
                env.render()
                
                if done:
                    obs = env.reset()
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            env.close()
    else:
        # Training mode
        print("=== Training Mode ===")
        print("This requires stable-baselines3. Run:")
        print("  python real_car_env.py --test  # to test CTE estimation first")
        
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
            
            vec_env = DummyVecEnv([lambda: env])
            
            model = PPO(
                "CnnPolicy",
                vec_env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=256,
                batch_size=64,
            )
            
            print("Starting real car training...")
            print("WARNING: The car will move! Ensure safe environment.")
            model.learn(total_timesteps=10000)
            model.save("real_car_policy.zip")
            
        except ImportError:
            print("stable-baselines3 not installed. Install with:")
            print("  pip install stable-baselines3==1.2.0")
        finally:
            env.close()

