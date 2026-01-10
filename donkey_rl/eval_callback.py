"""Ephemeral EvalCallback that creates and closes eval env on demand."""

from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym


class EphemeralEvalCallback:
    """EvalCallback that lazily creates eval_env only when needed.

    Unlike standard EvalCallback, this creates the eval environment only when
    the first evaluation is triggered, and closes it after each evaluation.
    This is useful for DonkeySim where we want to avoid starting the eval sim
    at the beginning of training.

    The callback creates the env before the first eval, uses it, then closes it.
    On subsequent evals, it creates a new env, uses it, and closes it again.
    """

    def __init__(
        self,
        *,
        eval_env_fn: Optional[Callable[[], gym.Env]] = None,
        eval_env: Optional[gym.Env] = None,
        best_model_save_path: str,
        log_path: Optional[str] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        render: bool = False,
    ):
        """
        Args:
            eval_env_fn: Factory function that creates the eval environment.
                         Either this or eval_env must be provided.
            eval_env: Pre-created eval environment (for backward compatibility).
                      If provided, eval_env_fn is ignored and env won't be closed.
            best_model_save_path: Path where the best model will be saved.
            log_path: Path to save evaluation logs.
            eval_freq: Evaluate the model every `eval_freq` timesteps.
            n_eval_episodes: Number of episodes to run during evaluation.
            deterministic: Whether to use deterministic actions during evaluation.
            render: Whether to render the evaluation.
        """
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

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
        self._n_eval_episodes = int(n_eval_episodes)
        self._deterministic = bool(deterministic)
        self._render = bool(render)
        self._DummyVecEnv = DummyVecEnv
        self._VecMonitor = VecMonitor
        self._evaluate_policy = evaluate_policy

        # Create the actual callback
        class _EphemeralEvalCallback(BaseCallback):
            def __init__(self, outer_self):
                super().__init__()
                self._outer = outer_self
                self.best_mean_reward = float("-inf")

            def _on_step(self) -> bool:
                # Check if it's time to evaluate (using n_calls like standard EvalCallback)
                if self._outer._eval_freq > 0 and self.n_calls % self._outer._eval_freq == 0:
                    self._create_and_evaluate()
                return True

            def _create_and_evaluate(self) -> None:
                """Create eval env, run evaluation, then close it."""
                eval_env = None
                try:
                    # Determine which env to use
                    if self._outer._eval_env is not None and not self._outer._should_close:
                        # Use pre-created env (backward compatibility)
                        eval_env = self._outer._eval_env
                    elif self._outer._eval_env_fn is not None:
                        # Create new env from factory function
                        print(f"[EphemeralEvalCallback] Creating eval environment at step {self.num_timesteps}...")
                        eval_env = self._outer._DummyVecEnv([self._outer._eval_env_fn])
                        if self._outer._log_path:
                            import os
                            os.makedirs(self._outer._log_path, exist_ok=True)
                            eval_env = self._outer._VecMonitor(
                                eval_env, filename=os.path.join(self._outer._log_path, "eval_monitor.csv")
                            )

                    if eval_env is None:
                        print("[EphemeralEvalCallback] Warning: No eval environment available, skipping evaluation.")
                        return

                    # Run evaluation
                    mean_reward, std_reward = self._outer._evaluate_policy(
                        self.model,
                        eval_env,
                        n_eval_episodes=self._outer._n_eval_episodes,
                        deterministic=self._outer._deterministic,
                        render=self._outer._render,
                    )

                    # Log results
                    if self.logger is not None:
                        self.logger.record("eval/mean_reward", float(mean_reward))
                        self.logger.record("eval/std_reward", float(std_reward))
                        self.logger.dump(self.num_timesteps)

                    # Save if best
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self._outer._best_model_save_path is not None:
                            import os
                            os.makedirs(os.path.dirname(self._outer._best_model_save_path) or ".", exist_ok=True)
                            self.model.save(self._outer._best_model_save_path)
                        print(
                            f"[EphemeralEvalCallback] New best mean reward: {mean_reward:.2f} +/- {std_reward:.2f} "
                            f"at step {self.num_timesteps}"
                        )
                    else:
                        print(
                            f"[EphemeralEvalCallback] Evaluation at step {self.num_timesteps}: "
                            f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f} "
                            f"(best: {self.best_mean_reward:.2f})"
                        )

                except Exception as e:
                    print(f"[EphemeralEvalCallback] Error during evaluation: {e}")
                    import traceback
                    traceback.print_exc()

                finally:
                    # Close the eval env if we created it
                    if eval_env is not None and self._outer._should_close:
                        print(f"[EphemeralEvalCallback] Closing eval environment after evaluation...")
                        try:
                            eval_env.close()
                        except Exception as e:
                            print(f"[EphemeralEvalCallback] Error closing eval env: {e}")

        self._impl = _EphemeralEvalCallback(self)

    @property
    def best_mean_reward(self) -> float:
        """Access the best mean reward from the underlying callback."""
        return getattr(self._impl, "best_mean_reward", float("-inf"))

    def sb3_callback(self):
        """Return the underlying SB3 callback instance."""
        return self._impl

