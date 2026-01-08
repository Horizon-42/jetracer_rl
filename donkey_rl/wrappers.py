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
