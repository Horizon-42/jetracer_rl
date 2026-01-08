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


class RandomFrictionWrapper(gym.Wrapper):
    """Domain randomization: simulate different friction by scaling throttle per episode.

    This does not require DonkeySim support for a friction parameter.
    Instead, we sample a multiplier s ~ Uniform([min_scale, max_scale]) on reset
    and apply it to the *Donkey* throttle action (action[1]).
    """

    def __init__(self, env: gym.Env, *, min_scale: float = 0.6, max_scale: float = 1.0):
        super().__init__(env)
        self._min = float(min_scale)
        self._max = float(max_scale)
        if self._min <= 0 or self._max <= 0 or self._min > self._max:
            raise ValueError(f"Invalid friction range: [{self._min}, {self._max}]")
        self.friction_scale: float = 1.0

    def reset(self, **kwargs):  # type: ignore[override]
        # Sample per-episode scale.
        self.friction_scale = float(np.random.uniform(self._min, self._max))
        return self.env.reset(**kwargs)

    def step(self, action):  # type: ignore[override]
        a = np.asarray(action, dtype=np.float32)
        if a.shape[-1] >= 2:
            # Donkey action format: [steer, throttle]
            a = a.copy()
            a[1] = np.clip(a[1] * self.friction_scale, 0.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(a)
        if isinstance(info, dict):
            info.setdefault("friction_scale", float(self.friction_scale))
        return obs, reward, terminated, truncated, info
