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

    def __init__(self, env: gym.Env, *, width: int = 84, height: int = 84):
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

        chw = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        return chw
