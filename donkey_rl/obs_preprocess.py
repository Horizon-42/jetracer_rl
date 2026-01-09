from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import cv2


class ObsPreprocess(gym.ObservationWrapper):
    """A-route sim-side observation preprocessing.

    We do NOT simulate camera artifacts in the simulator.
    Instead, we correct the real camera feed to match simulator-like images.

    This wrapper only:
    - caches raw simulator frames for debug
    - resizes to (width, height)
    - converts to CHW float32 in [0, 1]

    Debug caches (for DebugObsDumpCallback):
    - last_raw_observation: raw sim frame (HWC uint8)
    - last_resized_observation: resized frame (HWC uint8)
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        width: int = 84,
        height: int = 84,
        domain_rand: bool = False,
        aug_brightness: float = 0.25,
        aug_contrast: float = 0.25,
        aug_noise_std: float = 0.02,
        aug_color_jitter: float = 0.2,
    ):
        super().__init__(env)

        self._width = int(width)
        self._height = int(height)

        _h, _w, c = self.observation_space.shape
        assert c == 3, f"Expected RGB observation, got shape {self.observation_space.shape}"

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self._height, self._width),
            dtype=np.float32,
        )

        self.last_raw_observation: Optional[np.ndarray] = None
        self.last_resized_observation: Optional[np.ndarray] = None

        self._domain_rand = bool(domain_rand)
        self._aug_brightness = float(aug_brightness)
        self._aug_contrast = float(aug_contrast)
        self._aug_noise_std = float(aug_noise_std)
        self._aug_color_jitter = float(aug_color_jitter)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        try:
            raw = np.asarray(observation)
            if raw.dtype != np.uint8:
                raw = np.clip(raw, 0, 255).astype(np.uint8)
            self.last_raw_observation = raw.copy()
        except Exception:
            self.last_raw_observation = None
            raw = np.asarray(observation)

        resized = cv2.resize(raw, (self._width, self._height), interpolation=cv2.INTER_AREA)
        try:
            self.last_resized_observation = np.asarray(resized).copy()
        except Exception:
            self.last_resized_observation = None

        x = resized.astype(np.float32) / 255.0
        if self._domain_rand:
            # Color jitter: shift in HSV space
            if self._aug_color_jitter > 0:
                # Convert RGB to HSV
                hsv = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                
                # Shift Hue (0-180 in OpenCV)
                h_shift = float(np.random.uniform(-self._aug_color_jitter * 180, self._aug_color_jitter * 180))
                hsv[:, :, 0] = np.clip(hsv[:, :, 0] + h_shift, 0, 180)
                
                # Shift Saturation (0-255)
                s_shift = float(np.random.uniform(-self._aug_color_jitter * 255, self._aug_color_jitter * 255))
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] + s_shift, 0, 255)
                
                # Shift Value (0-255)
                v_shift = float(np.random.uniform(-self._aug_color_jitter * 255, self._aug_color_jitter * 255))
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] + v_shift, 0, 255)
                
                # Convert back to RGB
                x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

            # Contrast: multiply around 1.0
            if self._aug_contrast > 0:
                c = 1.0 + float(np.random.uniform(-self._aug_contrast, self._aug_contrast))
                x = x * c

            # Brightness: additive offset
            if self._aug_brightness > 0:
                b = float(np.random.uniform(-self._aug_brightness, self._aug_brightness))
                x = x + b

            # Noise: additive Gaussian
            if self._aug_noise_std > 0:
                n = np.random.normal(loc=0.0, scale=self._aug_noise_std, size=x.shape).astype(np.float32)
                x = x + n

            x = np.clip(x, 0.0, 1.0)

        chw = x.transpose(2, 0, 1).astype(np.float32)
        return chw
