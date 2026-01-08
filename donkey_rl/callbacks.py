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
                            lane = extra.get("lane") or {}
                            dist = lane.get("dist")
                            angle = lane.get("angle_rad")
                            speed = extra.get("speed")
                            reward = extra.get("reward")
                            if dist is not None and angle is not None:
                                print(
                                    f"step={self.num_timesteps} reward={reward:.3f} speed={speed:.2f} "
                                    f"dist={dist:.3f} angle={angle:.3f}"
                                )

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
