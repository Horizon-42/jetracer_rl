"""SB3 callbacks for training visualization, model saving, and debugging.

This module provides callback classes for Stable-Baselines3 (SB3) training:
- TrainingVizCallback: Render and print training metrics during training
- BestModelOnEpisodeRewardCallback: Save best model based on training episode rewards
- DebugObsDumpCallback: Save observations as images for debugging

These callbacks wrap SB3's BaseCallback to provide convenient interfaces
for common training monitoring and debugging tasks.
"""

from __future__ import annotations

import os
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback


class TrainingVizCallback:
    """Lightweight training visualization callback.

    This callback provides two visualization features:
    1. Optional rendering of the environment (if render=True)
    2. Periodic printing of training metrics (lane/cte/speed/reward)

    It's designed to be lightweight and not interfere with training performance.

    Example:
        >>> callback = TrainingVizCallback(render=False, print_every_steps=200)
        >>> model.learn(total_timesteps=10000, callback=callback.sb3_callback())
    """

    def __init__(self, render: bool, print_every_steps: int = 200):
        """Initialize the training visualization callback.

        Args:
            render: If True, call env.render() on each step (can be slow).
            print_every_steps: Print metrics every N steps (0 to disable printing).
        """
        self._impl = _TrainingVizCallbackImpl(render=render, print_every_steps=print_every_steps)

    def sb3_callback(self) -> BaseCallback:
        """Return the underlying SB3 callback instance.

        Returns:
            The BaseCallback instance that can be passed to model.learn().
        """
        return self._impl


class _TrainingVizCallbackImpl(BaseCallback):
    """Internal implementation of training visualization callback."""

    def __init__(self, render: bool, print_every_steps: int):
        """Initialize the callback implementation.

        Args:
            render: Whether to render the environment.
            print_every_steps: Print frequency in steps.
        """
        super().__init__()
        self._render = bool(render)
        self._print_every = int(print_every_steps)

    def _on_step(self) -> bool:
        """Called on each training step.

        Performs rendering (if enabled) and periodic metric printing.

        Returns:
            True to continue training.
        """
        # Render environment if requested
        if self._render:
            try:
                if hasattr(self.training_env, "envs") and self.training_env.envs:
                    # VecEnv case: render first environment
                    self.training_env.envs[0].render()
                else:
                    # Single env case
                    self.training_env.render()
            except Exception:
                # Silently ignore render errors (e.g., if display not available)
                pass

        # Print metrics periodically
        if self._print_every > 0 and (self.num_timesteps % self._print_every == 0):
            self._print_training_metrics()

        return True

    def _print_training_metrics(self) -> None:
        """Extract and print training metrics from info dict.

        Looks for reward information in the info dict (from reward wrappers)
        and prints step count, reward, speed, and CTE.
        """
        infos = self.locals.get("infos")
        if not isinstance(infos, (list, tuple)) or not infos:
            return

        # Try to get reward info from the first environment's info dict
        first_info = infos[0]
        if not isinstance(first_info, dict):
            return

        # Look for reward information from various reward wrappers
        reward_info = (
            first_info.get("race_reward")
            or first_info.get("race_reward_track")
            or first_info.get("centerline_v2_reward")
            or first_info.get("centerline_v3_reward")
            or first_info.get("centerline_v4_reward")
            or first_info.get("deepracer_reward")
        )

        if isinstance(reward_info, dict):
            reward = reward_info.get("reward")
            speed = reward_info.get("speed")
            cte = reward_info.get("cte")

            if reward is not None and speed is not None and cte is not None:
                print(
                    f"step={self.num_timesteps} reward={reward:.3f} "
                    f"speed={speed:.2f} cte={cte:.3f}"
                )


class BestModelOnEpisodeRewardCallback:
    """Save best model based on training episode reward.

    This callback monitors training episode rewards and saves the model
    whenever a new best episode reward is achieved. Unlike evaluation-based
    callbacks, this uses the training rewards directly.

    Example:
        >>> callback = BestModelOnEpisodeRewardCallback(save_path="./models/best_train")
        >>> model.learn(total_timesteps=10000, callback=callback.sb3_callback())
    """

    def __init__(self, save_path: str):
        """Initialize the best model callback.

        Args:
            save_path: Path where the best model will be saved.
        """
        self._impl = _BestModelOnEpisodeRewardCallbackImpl(save_path=save_path)

    def sb3_callback(self) -> BaseCallback:
        """Return the underlying SB3 callback instance.

        Returns:
            The BaseCallback instance that can be passed to model.learn().
        """
        return self._impl


class _BestModelOnEpisodeRewardCallbackImpl(BaseCallback):
    """Internal implementation of best model saving callback."""

    def __init__(self, save_path: str):
        """Initialize the callback implementation.

        Args:
            save_path: Path to save best models.
        """
        super().__init__()
        self._save_path = save_path
        self._best_reward = float("-inf")

    def _on_step(self) -> bool:
        """Called on each training step.

        Checks for completed episodes and saves model if reward is best so far.

        Returns:
            True to continue training.
        """
        infos = self.locals.get("infos")
        if not isinstance(infos, (list, tuple)):
            return True

        # Check each environment's info for completed episodes
        for info in infos:
            if not isinstance(info, dict):
                continue

            # SB3 automatically adds episode info when an episode completes
            episode_info = info.get("episode")
            if isinstance(episode_info, dict) and "r" in episode_info:
                episode_reward = float(episode_info["r"])

                # Save model if this is the best episode reward so far
                if episode_reward > self._best_reward:
                    self._best_reward = episode_reward
                    os.makedirs(os.path.dirname(self._save_path) or ".", exist_ok=True)
                    self.model.save(self._save_path)
                    print(
                        f"New best train episode reward={episode_reward:.2f}; "
                        f"saved best to {self._save_path}"
                    )

        return True


class DebugObsDumpCallback:
    """Debug helper: save observations as images during training.

    This callback is designed for debugging and sanity checking during training.
    It saves observations at regular intervals to verify:
    - Observations are being received correctly
    - Preprocessing is working as expected
    - Images look reasonable (not black, wrong channels, etc.)

    The callback can also be used to stop training early for debugging purposes.

    Example:
        >>> callback = DebugObsDumpCallback(
        ...     out_dir="./debug_obs",
        ...     stop_after_steps=1000,
        ...     save_every=10,
        ... )
        >>> model.learn(total_timesteps=10000, callback=callback.sb3_callback())
    """

    def __init__(
        self,
        *,
        out_dir: str,
        stop_after_steps: int,
        save_every: int = 10,
    ):
        """Initialize the debug observation dump callback.

        Args:
            out_dir: Directory to save observation images.
            stop_after_steps: Stop training after this many steps (0 to disable early stopping).
            save_every: Save observations every N steps.
        """
        self._impl = _DebugObsDumpCallbackImpl(
            out_dir=out_dir, stop_after_steps=stop_after_steps, save_every=save_every
        )

    def sb3_callback(self) -> BaseCallback:
        """Return the underlying SB3 callback instance.

        Returns:
            The BaseCallback instance that can be passed to model.learn().
        """
        return self._impl


class _DebugObsDumpCallbackImpl(BaseCallback):
    """Internal implementation of debug observation dump callback."""

    def __init__(self, out_dir: str, stop_after_steps: int, save_every: int):
        """Initialize the callback implementation.

        Args:
            out_dir: Directory for saving images.
            stop_after_steps: Early stopping threshold.
            save_every: Save frequency in steps.
        """
        super().__init__()
        self._out_dir = out_dir
        self._stop_after = int(stop_after_steps)
        self._save_every = max(1, int(save_every))
        self._saved_count = 0

    def _on_training_start(self) -> None:
        """Called once at the start of training.

        Creates the output directory if it doesn't exist.
        """
        os.makedirs(self._out_dir, exist_ok=True)

    def _on_step(self) -> bool:
        """Called on each training step.

        Saves observations periodically and checks for early stopping.

        Returns:
            True to continue training, False to stop early.
        """
        # Check for early stopping
        if self._stop_after > 0 and self.num_timesteps >= self._stop_after:
            return False

        # Save observations at specified intervals
        if self.num_timesteps % self._save_every == 0:
            self._save_observations()

        return True

    def _save_observations(self) -> None:
        """Save observation images for debugging.

        This method tries to save observations from multiple sources:
        1. Cached observations from ObsPreprocess wrapper (last_resized_observation,
           last_transformed_observation)
        2. Processed observations from SB3 (new_obs, in CHW format)

        All images are saved as PNG files in BGR format (OpenCV standard).
        """
        try:
            import numpy as np
            import cv2

            # Helper function to save RGB images as PNG
            def _save_rgb_image(frame_rgb: np.ndarray, name: str) -> None:
                """Save an RGB image to disk.

                Args:
                    frame_rgb: Image array in RGB format (HWC, uint8 or float32).
                    name: Base name for the saved file.
                """
                frame = np.asarray(frame_rgb)
                if frame.ndim != 3 or frame.shape[2] != 3:
                    return

                # Ensure uint8 format
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)

                # Convert RGB to BGR (OpenCV format)
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Save to file
                filename = f"{name}_{self.num_timesteps:07d}.png"
                filepath = os.path.join(self._out_dir, filename)
                cv2.imwrite(filepath, bgr)

            # Try to get cached observations from ObsPreprocess wrapper
            if hasattr(self.training_env, "envs") and self.training_env.envs:
                env0 = self.training_env.envs[0]
                transformed = getattr(env0, "last_transformed_observation", None)
                resized = getattr(env0, "last_resized_observation", None)

                if resized is not None:
                    _save_rgb_image(resized, "resized")
                    self._saved_count += 1

                if transformed is not None:
                    _save_rgb_image(transformed, "transformed")

            # Also save the processed observation that SB3 sees (CHW float32 [0,1])
            obs = self.locals.get("new_obs")
            if obs is not None:
                try:
                    # Get observation from first environment
                    frame = np.asarray(obs[0])
                    if frame.ndim == 3 and frame.shape[0] == 3:
                        # Convert from CHW to HWC and scale to uint8
                        hwc = (np.clip(frame, 0.0, 1.0).transpose(1, 2, 0) * 255.0).astype(
                            np.uint8
                        )
                        _save_rgb_image(hwc, "proc")
                except Exception:
                    # Silently ignore errors when saving processed obs
                    pass

        except Exception:
            # Never crash training from debug dumping
            # Errors are silently ignored to prevent interruption
            pass
