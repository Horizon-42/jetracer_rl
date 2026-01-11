"""Action and environment wrappers for DonkeyCar training.

This module provides wrappers that modify environment behavior:
- StepTimeoutWrapper: Prevents hanging on simulator disconnections
- JetRacerWrapper: Maps JetRacer action format to DonkeyCar format
- RandomFrictionWrapper: Domain randomization via friction scaling
- StallDetectionWrapper: Terminates episode when car is stuck/stalled
"""

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
    - Timeout of 0.0 disables timeout checking.

    Example:
        >>> env = make_donkey_env(...)
        >>> env = StepTimeoutWrapper(env, timeout_s=30.0)
        >>> obs, reward, done, info = env.step(action)  # Raises TimeoutError if >30s
    """

    def __init__(self, env: gym.Env, *, timeout_s: float = 30.0):
        """Initialize the timeout wrapper.

        Args:
            env: The environment to wrap.
            timeout_s: Timeout in seconds. If <= 0, timeout is disabled.
        """
        super().__init__(env)
        self._timeout_s = float(timeout_s)

    def _supports_signals(self) -> bool:
        """Check if signal-based timeout is available.

        Signals only work on Linux/macOS in the main thread.

        Returns:
            True if signals are available, False otherwise.
        """
        try:
            import signal
            import threading

            return (
                threading.current_thread() is threading.main_thread()
                and hasattr(signal, "setitimer")
            )
        except Exception:
            return False

    def _run_with_timeout(self, fn, *args, **kwargs):
        """Execute a function with a timeout using Unix signals.

        If timeout is disabled or signals are unavailable, executes the function
        normally without timeout protection.

        Args:
            fn: The function to execute.
            *args: Positional arguments to pass to fn.
            **kwargs: Keyword arguments to pass to fn.

        Returns:
            The return value of fn(*args, **kwargs).

        Raises:
            TimeoutError: If the function takes longer than timeout_s to execute
                          (only if signals are available).
        """
        if self._timeout_s <= 0 or not self._supports_signals():
            return fn(*args, **kwargs)

        import signal

        def _handler(_signum, _frame):  # type: ignore[no-untyped-def]
            """Signal handler that raises TimeoutError."""
            raise TimeoutError(f"DonkeySim IO timeout after {self._timeout_s:.1f}s")

        prev_handler = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, self._timeout_s)
            return fn(*args, **kwargs)
        finally:
            # Always restore the timer and signal handler
            try:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
            except Exception:
                pass
            try:
                signal.signal(signal.SIGALRM, prev_handler)
            except Exception:
                pass

    def reset(self, **kwargs):  # type: ignore[override]
        """Reset the environment with timeout protection.

        Args:
            **kwargs: Additional arguments passed to env.reset().

        Returns:
            The reset observation and info dict.

        Raises:
            TimeoutError: If reset takes longer than timeout_s (if signals available).
        """
        return self._run_with_timeout(self.env.reset, **kwargs)

    def step(self, action):  # type: ignore[override]
        """Step the environment with timeout protection.

        Args:
            action: Action to take in the environment.

        Returns:
            The step result tuple (obs, reward, terminated, truncated, info).

        Raises:
            TimeoutError: If step takes longer than timeout_s (if signals available).
        """
        return self._run_with_timeout(self.env.step, action)


class JetRacerWrapper(gym.ActionWrapper):
    """JetRacer-style action interface: [throttle, steering] -> DonkeyCar [steer, throttle].

    This wrapper converts actions from JetRacer's expected format to DonkeyCar's
    expected format. JetRacer uses [throttle, steering] where:
    - throttle: [0, 1] (forward throttle)
    - steering: [-1, 1] (left to right)

    DonkeyCar expects [steer, throttle] where:
    - steer: [-1, 1] (left to right)
    - throttle: [-0.5, 1.0] (reverse to forward)

    The wrapper also stores the last raw action for use by reward functions.

    Attributes:
        last_raw_action: Last action received in JetRacer format [throttle, steering].
        last_mapped_action: Last action sent to environment in DonkeyCar format [steer, throttle].
    """

    def __init__(self, env: gym.Env, steer_scale: float = 1.0, throttle_scale: float = 1.0):
        """Initialize the JetRacer action wrapper.

        Args:
            env: The environment to wrap.
            steer_scale: Scaling factor for steering actions (default: 1.0).
            throttle_scale: Scaling factor for throttle actions (default: 1.0).
        """
        super().__init__(env)
        self._steer_scale = float(steer_scale)
        self._throttle_scale = float(throttle_scale)

        self.last_raw_action: Optional[np.ndarray] = None
        self.last_mapped_action: Optional[np.ndarray] = None

        # Update action space to JetRacer format: [throttle, steering]
        self.action_space = gym.spaces.Box(
            low=np.array([0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        """Transform action from JetRacer format to DonkeyCar format.

        Args:
            action: Action in JetRacer format [throttle, steering].

        Returns:
            Action in DonkeyCar format [steer, throttle], with values clipped
            to valid ranges and scaling applied.
        """
        raw = np.asarray(action, dtype=np.float32)
        throttle = float(raw[0])
        steering = float(raw[1])

        # Apply scaling and clip to valid ranges
        steer = np.clip(steering * self._steer_scale, -1.0, 1.0)
        thr = np.clip(throttle * self._throttle_scale, -0.5, 1.0)

        # Store actions for reward functions
        mapped = np.array([steer, thr], dtype=np.float32)
        self.last_raw_action = raw
        self.last_mapped_action = mapped
        return mapped


class RandomFrictionWrapper(gym.Wrapper):
    """Domain randomization: simulate different friction by scaling throttle per episode.

    This wrapper implements domain randomization by randomly scaling the throttle
    action at the start of each episode. This simulates different friction conditions
    (wet road, dry road, different tire conditions, etc.) without requiring simulator
    support for friction parameters.

    How it works:
    - On reset(): Sample a random friction scale factor from [min_scale, max_scale]
    - On step(): Multiply the throttle action (action[1]) by the friction scale

    This helps improve sim-to-real transfer by training the agent to handle
    different effective throttle responses.

    Attributes:
        friction_scale: Current friction scale factor (updated on each reset).

    Example:
        >>> env = make_donkey_env(...)
        >>> env = RandomFrictionWrapper(env, min_scale=0.7, max_scale=1.0)
        >>> obs, info = env.reset()  # Friction scale sampled (e.g., 0.85)
        >>> obs, reward, done, info = env.step([steer, throttle])
        >>> # Throttle was multiplied by 0.85 before being sent to simulator
    """

    def __init__(self, env: gym.Env, *, min_scale: float = 0.6, max_scale: float = 1.0):
        """Initialize the random friction wrapper.

        Args:
            env: The environment to wrap.
            min_scale: Minimum friction scale factor (must be > 0, default: 0.6).
            max_scale: Maximum friction scale factor (must be > 0, default: 1.0).

        Raises:
            ValueError: If min_scale or max_scale are invalid (<= 0, or min_scale > max_scale).
        """
        super().__init__(env)
        self._min = float(min_scale)
        self._max = float(max_scale)
        if self._min <= 0 or self._max <= 0 or self._min > self._max:
            raise ValueError(f"Invalid friction range: [{self._min}, {self._max}]")
        self.friction_scale: float = 1.0

    def reset(self, **kwargs):  # type: ignore[override]
        """Reset the environment and sample a new friction scale factor.

        Args:
            **kwargs: Additional arguments passed to env.reset().

        Returns:
            The reset observation and info dict.
        """
        # Sample a new friction scale factor for this episode
        self.friction_scale = float(np.random.uniform(self._min, self._max))
        return self.env.reset(**kwargs)

    def step(self, action):  # type: ignore[override]
        """Step the environment with friction scaling applied to throttle.

        The throttle component of the action (action[1]) is multiplied by the
        current friction scale factor, then clipped to valid range.

        Args:
            action: Action in DonkeyCar format [steer, throttle].

        Returns:
            The step result tuple with friction_scale added to info dict.
        """
        a = np.asarray(action, dtype=np.float32)
        if a.shape[-1] >= 2:
            # Donkey action format: [steer, throttle]
            a = a.copy()
            # Apply friction scaling to throttle (action[1])
            a[1] = np.clip(a[1] * self.friction_scale, -0.5, 1.0)

        obs, reward, terminated, truncated, info = self.env.step(a)

        # Add friction scale to info for logging/debugging
        if isinstance(info, dict):
            info.setdefault("friction_scale", float(self.friction_scale))

        return obs, reward, terminated, truncated, info


class StallDetectionWrapper(gym.Wrapper):
    """Detect when the car is stuck/stalled and terminate the episode early.

    This wrapper monitors the car's speed and terminates the episode if the
    speed stays below a threshold for too many consecutive steps. This prevents
    the agent from learning to "game" the reward by simply stopping.

    How it works:
    - On each step, check if speed < speed_threshold
    - If yes, increment stall counter
    - If stall counter >= max_stall_steps, terminate episode with penalty
    - If speed >= speed_threshold, reset stall counter

    Attributes:
        stall_count: Current consecutive steps with low speed.

    Example:
        >>> env = make_donkey_env(...)
        >>> env = StallDetectionWrapper(env, speed_threshold=0.1, max_stall_steps=50)
        >>> obs, info = env.reset()
        >>> # Episode will auto-terminate if car stays still for 50+ steps
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        speed_threshold: float = 0.1,
        max_stall_steps: int = 50,
        stall_penalty: float = 20.0,
    ):
        """Initialize the stall detection wrapper.

        Args:
            env: The environment to wrap.
            speed_threshold: Speed below this value is considered "stalled" (default: 0.1).
            max_stall_steps: Terminate after this many consecutive stalled steps (default: 50).
            stall_penalty: Penalty added to reward when episode terminates due to stall (default: 20.0).
        """
        super().__init__(env)
        self._speed_threshold = float(speed_threshold)
        self._max_stall_steps = int(max_stall_steps)
        self._stall_penalty = float(stall_penalty)
        self.stall_count: int = 0

    def reset(self, **kwargs):  # type: ignore[override]
        """Reset the environment and stall counter.

        Args:
            **kwargs: Additional arguments passed to env.reset().

        Returns:
            The reset observation and info dict.
        """
        self.stall_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):  # type: ignore[override]
        """Step the environment with stall detection.

        Monitors speed and terminates episode if car is stalled for too long.

        Args:
            action: Action to take in the environment.

        Returns:
            The step result tuple. If stalled too long, terminated=True and
            reward is reduced by stall_penalty.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Extract speed from info dict
        info_dict = dict(info) if isinstance(info, dict) else {}
        speed = float(info_dict.get("speed", 0.0) or 0.0)

        # Check if stalled
        if speed < self._speed_threshold:
            self.stall_count += 1
        else:
            self.stall_count = 0

        # Terminate if stalled for too long
        stalled_out = False
        if self.stall_count >= self._max_stall_steps:
            stalled_out = True
            terminated = True
            reward -= self._stall_penalty

        # Add stall info for debugging
        if isinstance(info, dict):
            info["stall_count"] = self.stall_count
            info["stalled_out"] = stalled_out
            if stalled_out:
                info["stall_penalty"] = self._stall_penalty

        return obs, reward, terminated, truncated, info
