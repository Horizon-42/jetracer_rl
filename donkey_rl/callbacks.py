from __future__ import annotations

import os


class TrainingVizCallback:
    """Lightweight training visualization.

    - If rendering is enabled, calls env.render() each step.
    - Periodically prints lane/cte metrics from info.
    """

    def __init__(self, render: bool, print_every_steps: int = 200):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Cb(BaseCallback):
            def __init__(self, render: bool, print_every_steps: int):
                super().__init__()
                self._render = bool(render)
                self._print_every = int(print_every_steps)

            def _on_step(self) -> bool:
                if self._render:
                    try:
                        if hasattr(self.training_env, "envs") and self.training_env.envs:
                            self.training_env.envs[0].render()
                        else:
                            self.training_env.render()
                    except Exception:
                        pass

                if self._print_every > 0 and (self.num_timesteps % self._print_every == 0):
                    infos = self.locals.get("infos")
                    if isinstance(infos, (list, tuple)) and infos and isinstance(infos[0], dict):
                        extra = infos[0].get("race_reward") or infos[0].get("race_reward_track")
                        if isinstance(extra, dict):
                            cte = extra.get("cte")
                            speed = extra.get("speed")
                            reward = extra.get("reward")
                            if reward is not None and speed is not None and cte is not None:
                                print(f"step={self.num_timesteps} reward={reward:.3f} speed={speed:.2f} cte={cte:.3f}")

                return True

        self._impl = _Cb(render=render, print_every_steps=print_every_steps)

    def sb3_callback(self):
        return self._impl


class BestModelOnEpisodeRewardCallback:
    """Save best model based on *training* episode reward."""

    def __init__(self, save_path: str):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Cb(BaseCallback):
            def __init__(self, save_path: str):
                super().__init__()
                self._save_path = save_path
                self._best = float("-inf")

            def _on_step(self) -> bool:
                infos = self.locals.get("infos")
                if not isinstance(infos, (list, tuple)):
                    return True

                for info in infos:
                    if not isinstance(info, dict):
                        continue
                    ep = info.get("episode")
                    if isinstance(ep, dict) and "r" in ep:
                        r = float(ep["r"])
                        if r > self._best:
                            self._best = r
                            os.makedirs(os.path.dirname(self._save_path) or ".", exist_ok=True)
                            self.model.save(self._save_path)
                            print(f"New best train episode reward={r:.2f}; saved best to {self._save_path}")
                return True

        self._impl = _Cb(save_path=save_path)

    def sb3_callback(self):
        return self._impl


class DebugObsDumpCallback:
    """Debug helper: run for a short time and save observations as images.

    How it works
    ------------
    - During SB3 training, `_on_step` has access to `new_obs` (the observation after
      taking an action).
    - Our env wrapper produces observations in CHW float32 in [0, 1].
    - We convert to HWC uint8 and save PNGs.

    This is designed for quick sanity checks:
    - Are we getting images at all?
    - Do they look like the simulator view (not black / wrong channels)?
    """

    def __init__(
        self,
        *,
        out_dir: str,
        stop_after_steps: int,
        save_every: int = 10,
    ):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Cb(BaseCallback):
            def __init__(self, out_dir: str, stop_after_steps: int, save_every: int):
                super().__init__()
                self._out_dir = out_dir
                self._stop_after = int(stop_after_steps)
                self._save_every = max(1, int(save_every))
                self._saved = 0

            def _on_training_start(self) -> None:
                os.makedirs(self._out_dir, exist_ok=True)

            def _on_step(self) -> bool:
                # Early-stop in debug mode.
                if self._stop_after > 0 and self.num_timesteps >= self._stop_after:
                    return False

                if self.num_timesteps % self._save_every != 0:
                    return True

                obs = self.locals.get("new_obs")
                if obs is None:
                    return True

                try:
                    import numpy as np
                    import cv2

                    # VecEnv: obs shape is (n_envs, C, H, W). Take first env.
                    frame = obs[0]
                    if frame is None:
                        return True

                    frame = np.asarray(frame)
                    if frame.ndim != 3 or frame.shape[0] != 3:
                        return True

                    # CHW float [0,1] -> HWC uint8 [0,255]
                    hwc = (np.clip(frame, 0.0, 1.0).transpose(1, 2, 0) * 255.0).astype(np.uint8)
                    # OpenCV expects BGR for correct colors.
                    bgr = cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)

                    path = os.path.join(self._out_dir, f"obs_{self.num_timesteps:07d}.png")
                    cv2.imwrite(path, bgr)
                    self._saved += 1
                except Exception:
                    # Never crash training from debug dumping.
                    return True

                return True

        self._impl = _Cb(out_dir=out_dir, stop_after_steps=stop_after_steps, save_every=save_every)

    def sb3_callback(self):
        return self._impl
