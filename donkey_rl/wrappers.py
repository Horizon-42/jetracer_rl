from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np


class StepTimeoutWrapper(gym.Wrapper):
    """Abort hanging env.step()/reset() calls with a wall-clock timeout.

    Why this exists
    ---------------
    When DonkeySim (Unity) disconnects or gets stuck, the underlying client can
    block inside a socket recv. SB3 then appears to "freeze" because it never
    returns from env.step().

    This wrapper uses Unix signals to raise a TimeoutError if step/reset takes
    longer than `timeout_s`.

    Notes
    -----
    - Works on Linux/macOS in the main thread.
    - If signals are unavailable (e.g. Windows) or not in main thread, it falls
      back to no-op (no timeout enforcement).
    """

    def __init__(self, env: gym.Env, *, timeout_s: float = 30.0):
        super().__init__(env)
        self._timeout_s = float(timeout_s)

    def _supports_signals(self) -> bool:
        try:
            import signal
            import threading

            return threading.current_thread() is threading.main_thread() and hasattr(signal, "setitimer")
        except Exception:
            return False

    def _run_with_timeout(self, fn, *args, **kwargs):
        if self._timeout_s <= 0 or not self._supports_signals():
            return fn(*args, **kwargs)

        import signal

        def _handler(_signum, _frame):  # type: ignore[no-untyped-def]
            raise TimeoutError(f"DonkeySim IO timeout after {self._timeout_s:.1f}s")

        prev_handler = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, self._timeout_s)
            return fn(*args, **kwargs)
        finally:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
            except Exception:
                pass
            try:
                signal.signal(signal.SIGALRM, prev_handler)
            except Exception:
                pass

    def reset(self, **kwargs):  # type: ignore[override]
        return self._run_with_timeout(self.env.reset, **kwargs)

    def step(self, action):  # type: ignore[override]
        return self._run_with_timeout(self.env.step, action)


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
