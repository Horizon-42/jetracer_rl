"""Observation preprocessing wrapper for DonkeyCar environments.

This module provides observation preprocessing that transforms raw simulator
frames into the format expected by neural network policies. It handles:
- Resizing images to specified dimensions
- Perspective transformation (bird's-eye view)
- Domain randomization (brightness, contrast, noise, color jitter)
- Format conversion (HWC uint8 -> CHW float32 [0,1])

The preprocessing is designed to be compatible with real-world deployment
where similar transformations are applied to camera frames.
"""

from __future__ import annotations

from typing import Optional

import cv2
import gymnasium as gym
import numpy as np


def _apply_color_jitter(image: np.ndarray, jitter_strength: float) -> np.ndarray:
    """Apply color jitter augmentation in HSV color space.

    Color jitter randomly shifts hue, saturation, and value channels to
    simulate variations in lighting conditions and camera characteristics.
    This helps improve robustness to real-world lighting changes.

    Args:
        image: Input image in RGB format, float32 in [0, 1] range.
        jitter_strength: Strength of jittering (0.0 = no jitter, 1.0 = max jitter).

    Returns:
        Augmented image in RGB format, float32 in [0, 1] range.
    """
    if jitter_strength <= 0:
        return image

    # Convert RGB to HSV
    # Note: OpenCV uses H: 0-180, S: 0-255, V: 0-255
    hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

    # Shift Hue (0-180 in OpenCV)
    h_shift = float(np.random.uniform(-jitter_strength * 180, jitter_strength * 180))
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + h_shift, 0, 180)

    # Shift Saturation (0-255)
    s_shift = float(np.random.uniform(-jitter_strength * 255, jitter_strength * 255))
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + s_shift, 0, 255)

    # Shift Value (brightness in HSV) (0-255)
    v_shift = float(np.random.uniform(-jitter_strength * 255, jitter_strength * 255))
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + v_shift, 0, 255)

    # Convert back to RGB
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return rgb.astype(np.float32) / 255.0


def _apply_brightness_contrast(image: np.ndarray, brightness_range: float, contrast_range: float) -> np.ndarray:
    """Apply brightness and contrast augmentation.

    These augmentations simulate variations in exposure and camera gain settings.
    They are applied additively (brightness) and multiplicatively (contrast).

    Args:
        image: Input image in RGB format, float32 in [0, 1] range.
        brightness_range: Range for random brightness adjustment (-range to +range).
        contrast_range: Range for random contrast adjustment (1-range to 1+range).

    Returns:
        Augmented image in RGB format, float32 in [0, 1] range.
    """
    # Apply contrast: multiply around 1.0
    if contrast_range > 0:
        contrast_factor = 1.0 + float(np.random.uniform(-contrast_range, contrast_range))
        image = image * contrast_factor

    # Apply brightness: additive offset
    if brightness_range > 0:
        brightness_offset = float(np.random.uniform(-brightness_range, brightness_range))
        image = image + brightness_offset

    return image


def _apply_gaussian_noise(image: np.ndarray, noise_std: float) -> np.ndarray:
    """Apply additive Gaussian noise to simulate sensor noise.

    This augmentation helps the model become robust to camera noise and
    quantization artifacts that occur in real-world deployments.

    Args:
        image: Input image in RGB format, float32 in [0, 1] range.
        noise_std: Standard deviation of Gaussian noise to add.

    Returns:
        Augmented image in RGB format, float32 in [0, 1] range.
    """
    if noise_std <= 0:
        return image

    noise = np.random.normal(loc=0.0, scale=noise_std, size=image.shape).astype(np.float32)
    return image + noise


def _apply_domain_randomization(
    image: np.ndarray,
    *,
    brightness_range: float,
    contrast_range: float,
    noise_std: float,
    color_jitter_strength: float,
) -> np.ndarray:
    """Apply all domain randomization augmentations.

    This function applies multiple augmentation techniques to improve
    sim-to-real transfer. The order of operations matters:
    1. Color jitter (HSV space transformations)
    2. Contrast adjustment
    3. Brightness adjustment
    4. Gaussian noise

    Args:
        image: Input image in RGB format, float32 in [0, 1] range.
        brightness_range: Range for brightness adjustment.
        contrast_range: Range for contrast adjustment.
        noise_std: Standard deviation for Gaussian noise.
        color_jitter_strength: Strength of color jitter in HSV space.

    Returns:
        Augmented image in RGB format, float32 in [0, 1] range, clipped to [0, 1].
    """
    # Apply color jitter (in HSV space)
    if color_jitter_strength > 0:
        image = _apply_color_jitter(image, color_jitter_strength)

    # Apply contrast and brightness
    image = _apply_brightness_contrast(image, brightness_range, contrast_range)

    # Apply Gaussian noise
    if noise_std > 0:
        image = _apply_gaussian_noise(image, noise_std)

    # Ensure values stay in valid range
    return np.clip(image, 0.0, 1.0)


class ObsPreprocess(gym.ObservationWrapper):
    """Observation preprocessing wrapper for DonkeyCar environments.

    This wrapper transforms raw simulator frames into the format expected by
    neural network policies. It performs the following operations:

    1. Caches raw frames for debugging (accessible via last_raw_observation)
    2. Applies perspective transformation (if enabled) for bird's-eye view
    3. Resizes to target dimensions
    4. Applies domain randomization (if enabled) for sim-to-real transfer
    5. Converts to CHW format (channels-first) as float32 in [0, 1]

    Observation modes (obs_mode):
    - "raw": Use only the raw camera image (default when perspective_transform=False)
    - "perspective": Use only the perspective-transformed image
    - "mix": Stack raw + perspective images vertically, then resize to target dimensions

    Debug caches (for DebugObsDumpCallback):
    - last_raw_observation: Raw simulator frame (HWC uint8)
    - last_resized_observation: Resized frame (HWC uint8)
    - last_transformed_observation: Perspective-transformed frame (HWC uint8)

    Attributes:
        observation_space: Modified observation space (CHW float32 [0, 1])
        last_raw_observation: Cached raw observation from simulator
        last_resized_observation: Cached resized observation
        last_transformed_observation: Cached perspective-transformed observation
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        width: int = 84,
        height: int = 84,
        perspective_transform: bool = False,
        obs_mode: str = "auto",
        domain_rand: bool = False,
        aug_brightness: float = 0.25,
        aug_contrast: float = 0.25,
        aug_noise_std: float = 0.02,
        aug_color_jitter: float = 0.2,
    ):
        """Initialize the observation preprocessing wrapper.

        Args:
            env: The environment to wrap.
            width: Target width for resized observations (default: 84).
            height: Target height for resized observations (default: 84).
            perspective_transform: If True, apply perspective transformation to
                                   get bird's-eye view. Default: False. (Deprecated, use obs_mode)
            obs_mode: Observation mode. Options:
                - "auto": Use perspective_transform flag for backward compatibility
                - "raw": Use only raw camera image
                - "perspective": Use only perspective-transformed image
                - "mix": Stack raw + perspective vertically, resize to target dimensions
            domain_rand: If True, apply domain randomization augmentations.
                         Default: False.
            aug_brightness: Brightness augmentation range (applied if domain_rand=True).
                            Default: 0.25.
            aug_contrast: Contrast augmentation range (applied if domain_rand=True).
                          Default: 0.25.
            aug_noise_std: Gaussian noise standard deviation (applied if domain_rand=True).
                           Default: 0.02.
            aug_color_jitter: Color jitter strength in HSV space (applied if domain_rand=True).
                              Default: 0.2.

        Raises:
            AssertionError: If input observation is not RGB (3 channels).
            ValueError: If obs_mode is not recognized.
        """
        super().__init__(env)

        self._width = int(width)
        self._height = int(height)

        # Validate input observation space
        _h, _w, c = self.observation_space.shape
        assert c == 3, f"Expected RGB observation, got shape {self.observation_space.shape}"

        # Determine obs_mode
        obs_mode = str(obs_mode).lower().strip()
        if obs_mode == "auto":
            # Backward compatibility: use perspective_transform flag
            self._obs_mode = "perspective" if perspective_transform else "raw"
        elif obs_mode in ("raw", "perspective", "mix"):
            self._obs_mode = obs_mode
        else:
            raise ValueError(f"Unknown obs_mode: {obs_mode}. Supported modes: raw, perspective, mix, auto")

        # Update observation space to CHW format (channels-first)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self._height, self._width),
            dtype=np.float32,
        )

        # Debug caches for visualization/debugging
        self.last_raw_observation: Optional[np.ndarray] = None
        self.last_transformed_observation: Optional[np.ndarray] = None
        self.last_resized_observation: Optional[np.ndarray] = None

        # Domain randomization settings
        self._domain_rand = bool(domain_rand)
        self._aug_brightness = float(aug_brightness)
        self._aug_contrast = float(aug_contrast)
        self._aug_noise_std = float(aug_noise_std)
        self._aug_color_jitter = float(aug_color_jitter)

        # Perspective transformation settings
        # Source points define the region of interest in the original image
        # Destination points define where they should map to in the transformed image
        self._perspective_transform = self._obs_mode in ("perspective", "mix")
        self.perspective_src_pts = [(75, 154), (242, 154), (319, 238), (0, 238)]
        self.perspective_dst_pts = [(10, 10), (310, 10), (310, 230), (10, 230)]
        self.perspective_matrix = cv2.getPerspectiveTransform(
            np.array(self.perspective_src_pts, dtype=np.float32),
            np.array(self.perspective_dst_pts, dtype=np.float32),
        )
        self.perspective_image_size = (320, 240)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Process the raw observation from the environment.

        This method is called automatically by gym.Wrapper on each step.
        It applies all preprocessing transformations in sequence:
        1. Cache raw observation
        2. Apply perspective transform (if enabled by obs_mode)
        3. Combine images based on obs_mode (raw, perspective, or mix)
        4. Resize to target dimensions
        5. Apply domain randomization (if enabled)
        6. Convert to CHW float32 format

        Args:
            observation: Raw observation from the environment (HWC uint8).

        Returns:
            Processed observation (CHW float32 in [0, 1] range).
        """
        # Step 1: Cache raw observation and ensure uint8 format
        try:
            raw = np.asarray(observation)
            if raw.dtype != np.uint8:
                raw = np.clip(raw, 0, 255).astype(np.uint8)
            self.last_raw_observation = raw.copy()
        except Exception:
            self.last_raw_observation = None
            raw = np.asarray(observation)

        # Step 2: Apply perspective transformation (if needed)
        # This transforms the camera view to a bird's-eye view, which can
        # help with lane detection and path following tasks
        if self._perspective_transform:
            transformed = cv2.warpPerspective(
                raw,
                self.perspective_matrix,
                self.perspective_image_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),  # Black padding
            )
            self.last_transformed_observation = transformed.copy()
        else:
            transformed = None

        # Step 3: Combine images based on obs_mode
        if self._obs_mode == "raw":
            # Use only raw image
            to_resize = raw
        elif self._obs_mode == "perspective":
            # Use only perspective-transformed image
            to_resize = transformed
        elif self._obs_mode == "mix":
            # Stack raw + perspective vertically, then resize to target dimensions
            # This gives the network both the original view and bird's-eye view
            # Both images should have the same width for proper stacking
            raw_resized_w = cv2.resize(raw, (self.perspective_image_size[0], self.perspective_image_size[1]), 
                                        interpolation=cv2.INTER_AREA)
            # Stack vertically: [raw on top, perspective on bottom]
            to_resize = np.vstack([raw_resized_w, transformed])
        else:
            # Fallback to raw
            to_resize = raw

        # Step 4: Resize to target dimensions
        # Using INTER_AREA interpolation which is best for downsampling
        resized = cv2.resize(
            to_resize, (self._width, self._height), interpolation=cv2.INTER_AREA
        )

        # Cache resized observation for debugging
        try:
            self.last_resized_observation = np.asarray(resized).copy()
        except Exception:
            self.last_resized_observation = None

        # Step 5: Convert to float32 in [0, 1] range
        processed = resized.astype(np.float32) / 255.0

        # Step 6: Apply domain randomization (if enabled)
        # This should be done in float32 space for numerical precision
        if self._domain_rand:
            processed = _apply_domain_randomization(
                processed,
                brightness_range=self._aug_brightness,
                contrast_range=self._aug_contrast,
                noise_std=self._aug_noise_std,
                color_jitter_strength=self._aug_color_jitter,
            )

        # Step 7: Convert from HWC to CHW format (channels-first)
        # This is the format expected by PyTorch and most neural network frameworks
        chw = processed.transpose(2, 0, 1).astype(np.float32)

        return chw
