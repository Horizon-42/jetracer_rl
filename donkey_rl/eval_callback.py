"""Ephemeral EvalCallback that creates and closes eval env on demand.

This module provides an evaluation callback that lazily creates evaluation
environments only when needed, rather than keeping them open throughout training.
This is useful for DonkeySim where we want to avoid starting the eval simulator
at the beginning of training.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


class EphemeralEvalCallback:
    """EvalCallback that lazily creates eval_env only when needed.

    Unlike standard EvalCallback, this creates the eval environment only when
    the first evaluation is triggered, and closes it after each evaluation.
    This is useful for DonkeySim where we want to avoid starting the eval sim
    at the beginning of training.

    The callback creates the env before the first eval, uses it, then closes it.
    On subsequent evals, it creates a new env, uses it, and closes it again.

    Attributes:
        best_mean_reward: The best mean reward seen so far during evaluation.

    Example:
        >>> eval_env_fn = lambda: make_donkey_env(...)
        >>> callback = EphemeralEvalCallback(
        ...     eval_env_fn=eval_env_fn,
        ...     best_model_save_path="./models/best_model",
        ...     eval_freq=10000,
        ...     n_eval_episodes=5,
        ... )
        >>> model.learn(total_timesteps=100000, callback=callback.sb3_callback())
    """

    def __init__(
        self,
        *,
        eval_env_fn: Optional[Callable[[], gym.Env]] = None,
        eval_env: Optional[gym.Env] = None,
        best_model_save_path: str,
        log_path: Optional[str] = None,
        eval_freq: int = 10000,
        eval_freq_type: str = "timesteps",  # "timesteps" or "episodes"
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        render: bool = False,
        max_episode_steps: Optional[int] = None,
    ):
        """Initialize the ephemeral evaluation callback.

        Args:
            eval_env_fn: Factory function that creates the eval environment.
                         Either this or eval_env must be provided.
            eval_env: Pre-created eval environment (for backward compatibility).
                      If provided, eval_env_fn is ignored and env won't be closed.
            best_model_save_path: Path where the best model will be saved.
            log_path: Path to save evaluation logs (monitor CSV files).
            eval_freq: Evaluation frequency. If eval_freq_type="timesteps", evaluate
                       every N timesteps. If eval_freq_type="episodes", evaluate
                       every N episodes. Default: 10000.
            eval_freq_type: Type of eval_freq - "timesteps" or "episodes". Default: "timesteps".
            n_eval_episodes: Number of episodes to run during evaluation.
            deterministic: Whether to use deterministic actions during evaluation.
            render: Whether to render the evaluation (usually False during training).
            max_episode_steps: Maximum number of steps per episode during evaluation.
                              If None, no limit is applied. If set, episodes will be
                              truncated (not terminated) when reaching this limit.
                              Default: None (no limit). Recommended: 5000-10000 steps.

        Raises:
            ValueError: If neither eval_env_fn nor eval_env is provided, or if
                        eval_freq_type is not "timesteps" or "episodes".
        """
        if eval_env is not None:
            # Backward compatibility: if eval_env is provided, use it directly
            # and don't close it (let the caller manage its lifecycle)
            self._eval_env_fn = None
            self._should_close = False
            self._eval_env = eval_env
        elif eval_env_fn is not None:
            # Lazy initialization mode: store the factory function
            self._eval_env_fn = eval_env_fn
            self._should_close = True
            self._eval_env = None
        else:
            raise ValueError("Either eval_env_fn or eval_env must be provided")

        self._best_model_save_path = best_model_save_path
        self._log_path = log_path
        self._eval_freq = int(eval_freq)
        self._eval_freq_type = str(eval_freq_type).lower()
        if self._eval_freq_type not in ("timesteps", "episodes"):
            raise ValueError(
                f"eval_freq_type must be 'timesteps' or 'episodes', got '{eval_freq_type}'"
            )
        self._n_eval_episodes = int(n_eval_episodes)
        self._deterministic = bool(deterministic)
        self._render = bool(render)
        self._max_episode_steps = int(max_episode_steps) if max_episode_steps is not None else None

        # Create the actual callback implementation
        self._impl = _EphemeralEvalCallbackImpl(
            outer_self=self,
            eval_env_fn=self._eval_env_fn,
            eval_env=self._eval_env,
            should_close=self._should_close,
            best_model_save_path=self._best_model_save_path,
            log_path=self._log_path,
            eval_freq=self._eval_freq,
            eval_freq_type=self._eval_freq_type,
            n_eval_episodes=self._n_eval_episodes,
            deterministic=self._deterministic,
            render=self._render,
            max_episode_steps=self._max_episode_steps,
        )

    @property
    def best_mean_reward(self) -> float:
        """Access the best mean reward from the underlying callback.

        Returns:
            The best mean reward seen during evaluation, or -inf if no evaluation
            has been performed yet.
        """
        return getattr(self._impl, "best_mean_reward", float("-inf"))

    def sb3_callback(self) -> BaseCallback:
        """Return the underlying SB3 callback instance.

        Returns:
            The BaseCallback instance that can be passed to model.learn().
        """
        return self._impl


class _EphemeralEvalCallbackImpl(BaseCallback):
    """Internal implementation of the ephemeral evaluation callback.

    This class handles the actual callback logic, including:
    - Tracking timesteps/episodes to determine when to evaluate
    - Creating and closing evaluation environments
    - Running evaluations using evaluate_policy
    - Saving best models based on evaluation results
    - Logging evaluation metrics to TensorBoard
    """

    def __init__(
        self,
        outer_self: EphemeralEvalCallback,
        eval_env_fn: Optional[Callable[[], gym.Env]],
        eval_env: Optional[gym.Env],
        should_close: bool,
        best_model_save_path: str,
        log_path: Optional[str],
        eval_freq: int,
        eval_freq_type: str,
        n_eval_episodes: int,
        deterministic: bool,
        render: bool,
        max_episode_steps: Optional[int],
    ):
        """Initialize the callback implementation.

        Args:
            outer_self: Reference to the outer EphemeralEvalCallback instance.
            eval_env_fn: Factory function for creating eval environments.
            eval_env: Pre-created eval environment (if available).
            should_close: Whether to close eval environments after use.
            best_model_save_path: Path to save best models.
            log_path: Path for evaluation logs.
            eval_freq: Evaluation frequency.
            eval_freq_type: Type of frequency ("timesteps" or "episodes").
            n_eval_episodes: Number of episodes per evaluation.
            deterministic: Whether to use deterministic actions.
            render: Whether to render during evaluation.
            max_episode_steps: Maximum steps per episode. If None, no limit.
        """
        super().__init__()
        self._outer = outer_self
        self._eval_env_fn = eval_env_fn
        self._eval_env = eval_env
        self._should_close = should_close
        self._best_model_save_path = best_model_save_path
        self._log_path = log_path
        self._eval_freq = eval_freq
        self._eval_freq_type = eval_freq_type
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self._render = render
        self._max_episode_steps = max_episode_steps

        self.best_mean_reward = float("-inf")

        # Initialize tracking variables based on frequency type
        if self._eval_freq_type == "episodes":
            self._episode_count = 0
            self._episodes_since_last_eval = 0
        else:  # timesteps
            self._last_eval_step = -1  # Use -1 to indicate not initialized yet

    def _init_callback(self) -> None:
        """Initialize callback - called once at the start of training."""
        super()._init_callback()
        freq_type_str = "timesteps" if self._eval_freq_type == "timesteps" else "episodes"
        print(
            f"[EphemeralEvalCallback] Initialized: will evaluate every "
            f"{self._eval_freq} {freq_type_str}."
        )

    def _on_step(self) -> bool:
        """Called on each training step.

        Checks if it's time to evaluate based on the configured frequency type
        (timesteps or episodes) and triggers evaluation if needed.

        Returns:
            True to continue training, False to stop (never used here).
        """
        if self._eval_freq_type == "episodes":
            return self._check_episode_based_eval()
        else:  # timesteps
            return self._check_timestep_based_eval()

    def _check_episode_based_eval(self) -> bool:
        """Check if evaluation should be triggered based on episode count.

        Returns:
            True to continue training.
        """
        # Check if any episodes have completed by looking at infos
        infos = self.locals.get("infos", [])
        if isinstance(infos, (list, tuple)):
            for info in infos:
                if isinstance(info, dict) and "episode" in info:
                    # An episode has completed
                    self._episode_count += 1
                    self._episodes_since_last_eval += 1

                    # Check if we've accumulated enough episodes since last evaluation
                    if self._eval_freq > 0 and self._episodes_since_last_eval >= self._eval_freq:
                        print(
                            f"[EphemeralEvalCallback] Episode {self._episode_count} completed. "
                            f"Triggering evaluation (every {self._eval_freq} episodes)."
                        )
                        self._episodes_since_last_eval = 0  # Reset counter before evaluation
                        self._create_and_evaluate()
        return True

    def _check_timestep_based_eval(self) -> bool:
        """Check if evaluation should be triggered based on timestep count.

        Returns:
            True to continue training.
        """
        if self._eval_freq > 0 and self.num_timesteps > 0:
            # On first call (when _last_eval_step is still -1), initialize it to 0
            if self._last_eval_step == -1:
                self._last_eval_step = 0
                print(
                    f"[EphemeralEvalCallback] Initialized at timestep {self.num_timesteps}. "
                    f"Will evaluate every {self._eval_freq} timesteps."
                )
                return True

            # Check if we've accumulated enough timesteps since last evaluation
            steps_since_last_eval = self.num_timesteps - self._last_eval_step
            if steps_since_last_eval >= self._eval_freq:
                # Update last eval step before evaluation (to avoid re-triggering)
                self._last_eval_step = self.num_timesteps
                print(
                    f"[EphemeralEvalCallback] Triggering evaluation at timestep "
                    f"{self.num_timesteps} (every {self._eval_freq} timesteps)."
                )
                self._create_and_evaluate()
        return True

    def _create_and_evaluate(self) -> None:
        """Create eval env, run evaluation, then close it.

        This method:
        1. Creates or retrieves the evaluation environment
        2. Wraps it with VecMonitor if log_path is provided
        3. Runs evaluation using evaluate_policy
        4. Logs results to TensorBoard
        5. Saves the model if it's the best so far
        6. Closes the environment if we created it
        """
        eval_env = None
        try:
            # Determine which env to use
            if self._eval_env is not None and not self._should_close:
                # Use pre-created env (backward compatibility)
                eval_env = self._eval_env
            elif self._eval_env_fn is not None:
                # Create new env from factory function
                episode_info = (
                    f"after {getattr(self, '_episode_count', 0)} episodes, "
                    if self._eval_freq_type == "episodes"
                    else ""
                )
                print(
                    f"[EphemeralEvalCallback] Creating eval environment "
                    f"({episode_info}timestep {self.num_timesteps})..."
                )
                
                # Wrap eval_env_fn to add TimeLimit if max_episode_steps is set
                def _wrapped_eval_env_fn():
                    env = self._eval_env_fn()
                    if self._max_episode_steps is not None and self._max_episode_steps > 0:
                        # Check if env already has TimeLimit wrapper by unwrapping
                        unwrapped = env
                        has_timelimit = False
                        while hasattr(unwrapped, 'env'):
                            if isinstance(unwrapped, TimeLimit):
                                has_timelimit = True
                                break
                            unwrapped = unwrapped.env
                        
                        if not has_timelimit:
                            env = TimeLimit(env, max_episode_steps=self._max_episode_steps)
                            print(
                                f"[EphemeralEvalCallback] Applied TimeLimit wrapper: "
                                f"max_episode_steps={self._max_episode_steps}"
                            )
                        else:
                            print(
                                f"[EphemeralEvalCallback] Environment already has TimeLimit wrapper, "
                                f"using existing max_episode_steps"
                            )
                    return env
                
                eval_env = DummyVecEnv([_wrapped_eval_env_fn])

                # Add VecMonitor for logging if log_path is provided
                if self._log_path:
                    os.makedirs(self._log_path, exist_ok=True)
                    eval_env = VecMonitor(
                        eval_env, filename=os.path.join(self._log_path, "eval_monitor.csv")
                    )

            if eval_env is None:
                print(
                    "[EphemeralEvalCallback] Warning: No eval environment available, "
                    "skipping evaluation."
                )
                return

            # Run evaluation
            mean_reward, std_reward = evaluate_policy(
                self.model,
                eval_env,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
                render=self._render,
            )

            # Log results to TensorBoard
            if self.logger is not None:
                self.logger.record("eval/mean_reward", float(mean_reward))
                self.logger.record("eval/std_reward", float(std_reward))
                self.logger.dump(self.num_timesteps)

            # Save if this is the best model so far
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self._best_model_save_path is not None:
                    os.makedirs(os.path.dirname(self._best_model_save_path) or ".", exist_ok=True)
                    self.model.save(self._best_model_save_path)
                    episode_info = (
                        f"episode {getattr(self, '_episode_count', 'N/A')}, "
                        if self._eval_freq_type == "episodes"
                        else ""
                    )
                    print(
                        f"[EphemeralEvalCallback] New best mean reward: "
                        f"{mean_reward:.2f} +/- {std_reward:.2f} "
                        f"({episode_info}timestep {self.num_timesteps})"
                    )
            else:
                episode_info = (
                    f"episode {getattr(self, '_episode_count', 'N/A')}, "
                    if self._eval_freq_type == "episodes"
                    else ""
                )
                print(
                    f"[EphemeralEvalCallback] Evaluation ({episode_info}timestep "
                    f"{self.num_timesteps}): mean_reward={mean_reward:.2f} +/- "
                    f"{std_reward:.2f} (best: {self.best_mean_reward:.2f})"
                )

        except Exception as e:
            print(f"[EphemeralEvalCallback] Error during evaluation: {e}")
            import traceback

            traceback.print_exc()

        finally:
            # Close the eval env if we created it
            if eval_env is not None and self._should_close:
                print("[EphemeralEvalCallback] Closing eval environment after evaluation...")
                try:
                    eval_env.close()
                except Exception as e:
                    print(f"[EphemeralEvalCallback] Error closing eval env: {e}")
