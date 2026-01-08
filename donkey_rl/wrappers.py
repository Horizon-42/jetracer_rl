from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np


class JetRacerWrapper(gym.ActionWrapper):
    """JetRacer-style action interface: [throttle, steering] -> DonkeyCar [steer, throttle]."""

    def __init__(self, env: gym.Env, steer_scale: float = 1.0, throttle_scale: float = 1.0):
        super().__init__(env)
        self._steer_scale = float(steer_scale)
        self._throttle_scale = float(throttle_scale)

        self.last_raw_action: Optional[np.ndarray] = None
        self.last_mapped_action: Optional[np.ndarray] = None

        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        raw = np.asarray(action, dtype=np.float32)
        throttle = float(raw[0])
        steering = float(raw[1])

        steer = np.clip(steering * self._steer_scale, -1.0, 1.0)
        thr = np.clip(throttle * self._throttle_scale, 0.0, 1.0)

        mapped = np.array([steer, thr], dtype=np.float32)
        self.last_raw_action = raw
        self.last_mapped_action = mapped
        return mapped


class ResizeNormalizeObs(gym.ObservationWrapper):
    """Resize RGB observations and normalize to [0, 1]."""

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)

        self._width = int(width)
        self._height = int(height)

        # Expect original observations to be HWC uint8 images
        _h, _w, c = self.observation_space.shape
        assert c == 3, f"Expected RGB observation, got shape {self.observation_space.shape}"

        # New observation is CHW float32
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self._height, self._width),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        import cv2

        resized = cv2.resize(observation, (self._width, self._height), interpolation=cv2.INTER_AREA)
        chw = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        return chw
