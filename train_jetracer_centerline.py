"""Train a DonkeyCar (Unity sim) policy with a JetRacer-style interface.

Goal
----
Train a policy that can drive fast without leaving the drivable area.

Action space (JetRacer-style)
----------------------------
- Action: [throttle, steering]
    - throttle in [0, 1]
    - steering in [-1, 1]

Under the hood, gym-donkeycar expects [steer, throttle].

Notes
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


def _patch_old_gym_render_mode() -> None:
    """Compat: Shimmy expects the underlying Gym env to expose `render_mode`.

    gym-donkeycar is based on old OpenAI Gym environments that often do not define
    `render_mode`, which can crash Shimmy during wrapper init.
    """

    try:
        import gym as old_gym  # type: ignore
    except Exception:
        return

    if getattr(old_gym.make, "__name__", "") == "_make_with_render_mode":
        return

    original_make = old_gym.make

    def _make_with_render_mode(*args, **kwargs):
        env = original_make(*args, **kwargs)
        if not hasattr(env, "render_mode"):
            try:
                setattr(env, "render_mode", None)
            except Exception:
                pass
        return env

    old_gym.make = _make_with_render_mode


@dataclass(frozen=True)
class RaceRewardConfig:
    """Reward shaping weights for a "race" objective.

    Key idea: maximize forward progress/speed, but heavily punish leaving the road.
    """

    # Reward for moving forward along the lane direction (progress proxy)
    w_progress: float = 2.0

    # Additional speed reward (helps policy commit to moving)
    w_speed: float = 0.5

    # Penalty for lateral deviation from lane center (meters)
    w_center: float = 2.0

    # Penalty for heading misalignment (radians)
    w_heading: float = 0.3

    # Penalty for steering magnitude (reduces "zig-zag")
    w_steer: float = 0.10

    # Penalty for steering change between steps
    w_steer_rate: float = 0.05

    # Proximity penalty term (Duckietown's collision-avoidance signal)
    w_proximity: float = 40.0

    # Big penalty when episode ends due to invalid pose / off-road
    offroad_penalty: float = 50.0


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


class ResizeNormalizeObs(gym.ObservationWrapper):
    """Resize RGB observations and normalize to [0, 1]."""

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)

        self._width = int(width)
        self._height = int(height)

        # Expect original observations to be HWC uint8 images
        _h, _w, c = self.observation_space.shape
        assert c == 3, f"Expected RGB observation, got shape {self.observation_space.shape}"

        # New observation is CHW float32
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self._height, self._width),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        import cv2

        resized = cv2.resize(observation, (self._width, self._height), interpolation=cv2.INTER_AREA)
        chw = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        return chw


class JetRacerRaceRewardWrapper(gym.Wrapper):
    """Race-oriented reward shaping.

    Rewards progress/speed, penalizes leaving the road, and discourages zig-zag steering.
    Adds info under `info["race_reward"]`.
    """

    def __init__(self, env: gym.Env, cfg: RaceRewardConfig = RaceRewardConfig()):
        super().__init__(env)
        self.cfg = cfg
        self._prev_steering: float = 0.0

    def step(self, action):
        # gymnasium: obs, reward, terminated, truncated, info
        # old gym:   obs, reward, done, info
        result = self.env.step(action)
        if len(result) == 5:
            obs, _base_reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, _base_reward, done, info = result

        lane = self._extract_lane_position(info)
        speed = self._extract_speed(info)
        proximity = self._extract_proximity_penalty(info)
        done_why = self._extract_done_code(info)
        throttle, steering = self._extract_jetracer_raw_action()

        shaped_reward = 0.0

        if lane is not None:
            dist = float(lane.get("dist", 0.0))
            angle_rad = float(lane.get("angle_rad", 0.0))
            dot_dir = float(lane.get("dot_dir", 0.0))

            progress = max(0.0, dot_dir) * float(speed)
            shaped_reward += self.cfg.w_progress * progress
            shaped_reward += self.cfg.w_speed * float(speed)
            shaped_reward -= self.cfg.w_center * abs(dist)
            shaped_reward -= self.cfg.w_heading * abs(angle_rad)
        else:
            shaped_reward -= 1.0

        shaped_reward += self.cfg.w_proximity * float(proximity)

        shaped_reward -= self.cfg.w_steer * float(steering**2)
        shaped_reward -= self.cfg.w_steer_rate * float((steering - self._prev_steering) ** 2)
        self._prev_steering = float(steering)

        if done and done_why in {"invalid-pose", "offroad-or-collision", "collision"}:
            shaped_reward -= self.cfg.offroad_penalty

        info = dict(info)
        info["race_reward"] = {
            "reward": float(shaped_reward),
            "speed": float(speed),
            "proximity": float(proximity),
            "throttle": float(throttle),
            "steering": float(steering),
            "done_why": done_why,
            "lane": lane,
        }
        if len(result) == 5:
            return obs, float(shaped_reward), bool(terminated), bool(truncated), info
        return obs, float(shaped_reward), bool(done), info

    def _extract_jetracer_raw_action(self) -> Tuple[float, float]:
        if isinstance(self.env, JetRacerWrapper) and self.env.last_raw_action is not None:
            raw = self.env.last_raw_action
            return float(raw[0]), float(raw[1])
        return 0.0, 0.0

    @staticmethod
    def _extract_lane_position(info: Dict) -> Optional[Dict]:
        if not isinstance(info, dict):
            return None

        # Duckietown-style
        sim = info.get("Simulator")
        if isinstance(sim, dict) and isinstance(sim.get("lane_position"), dict):
            return sim["lane_position"]

        # DonkeyCar-style: approximate lane metrics from cte (cross track error)
        # gym-donkeycar commonly returns info like: {"cte": ..., "speed": ..., "hit": ...}
        cte = info.get("cte")
        if cte is None:
            return None
        return {
            "dist": float(cte),
            "angle_rad": 0.0,
            "dot_dir": 1.0,
        }

    @staticmethod
    def _extract_speed(info: Dict) -> float:
        if not isinstance(info, dict):
            return 0.0

        # Duckietown-style
        sim = info.get("Simulator")
        if isinstance(sim, dict):
            speed = sim.get("robot_speed")
            if speed is not None:
                return float(speed)

        # DonkeyCar-style
        speed = info.get("speed")
        if speed is not None:
            return float(speed)
        return 0.0

    @staticmethod
    def _extract_proximity_penalty(info: Dict) -> float:
        if not isinstance(info, dict):
            return 0.0

        # Duckietown-style
        sim = info.get("Simulator")
        if isinstance(sim, dict):
            val = sim.get("proximity_penalty")
            if val is not None:
                return float(val)

        # DonkeyCar envs do not provide an equivalent signal by default.
        return 0.0

    @staticmethod
    def _extract_done_code(info: Dict) -> str:
        if not isinstance(info, dict):
            return "in-progress"

        # Duckietown-style
        sim = info.get("Simulator")
        if isinstance(sim, dict):
            msg = sim.get("msg")
            if isinstance(msg, str):
                low = msg.lower()
                if "invalid pose" in low:
                    return "invalid-pose"
                if "max_steps" in low or "max steps" in low:
                    return "max-steps-reached"

        # DonkeyCar-style
        hit = info.get("hit")
        if isinstance(hit, str) and hit != "none":
            return "collision"
        return "in-progress"


class TrainingVizCallback:
    """Lightweight training visualization.

    - If rendering is enabled, calls env.render() each step.
    - Periodically prints lane metrics from info["race_reward"].

    Implemented as an SB3 callback (BaseCallback) to keep the main loop clean.
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
                        # DummyVecEnv exposes `.envs`; render only the first env.
                        if hasattr(self.training_env, "envs") and self.training_env.envs:
                            self.training_env.envs[0].render()
                        else:
                            self.training_env.render()
                    except Exception:
                        # Rendering can fail on headless/GL setups; don't kill training.
                        pass

                if self._print_every > 0 and (self.num_timesteps % self._print_every == 0):
                    infos = self.locals.get("infos")
                    if isinstance(infos, (list, tuple)) and infos:
                        extra = infos[0].get("race_reward") if isinstance(infos[0], dict) else None
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


def make_duckietown_env(
    env_id: str,
    host: str,
    port: int,
    exe_path: str,
    *,
    fast_mode: bool,
) -> gym.Env:
    """Create a DonkeyCar env through Gymnasium+Shimmy."""

    _patch_old_gym_render_mode()

    # Import so the donkey envs register with old gym.
    import gym_donkeycar  # noqa: F401
    import shimmy  # noqa: F401

    conf: Dict = {
        "exe_path": exe_path,
        "host": host,
        "port": int(port),
        "body_style": "donkey",
        "body_rgb": (255, 165, 0),
        "car_name": "JetRacerAgent",
        "font_size": 100,
    }

    if fast_mode:
        conf["frame_skip"] = 2
        conf["cam_resolution"] = (60, 80, 3)
        conf["cam_config"] = {
            "img_w": 80,
            "img_h": 60,
            "img_d": 3,
            "img_enc": "JPG",
        }

    # GymV21 wrapper avoids passing seed/options to old-gym env.reset
    env = gym.make(
        "GymV21Environment-v0",
        env_id=env_id,
        make_kwargs={"conf": conf},
    )
    return env


def build_env_fn(args: argparse.Namespace) -> Callable[[], gym.Env]:
    def _thunk() -> gym.Env:
        env = make_duckietown_env(
            env_id=args.env_id,
            host=args.host,
            port=args.port,
            exe_path=args.exe_path,
            fast_mode=args.fast,
        )
        env = JetRacerWrapper(env, steer_scale=1.0, throttle_scale=1.0)
        env = JetRacerRaceRewardWrapper(env)
        env = ResizeNormalizeObs(env, width=args.obs_width, height=args.obs_height)
        return env

    return _thunk


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DonkeyCar policy (JetRacer-style actions).")

    parser.add_argument(
        "--env-id",
        type=str,
        default="donkey-waveshare-v0",
        help="DonkeyCar gym env id (e.g. donkey-generated-roads-v0)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument(
        "--exe-path",
        type=str,
        default="remote",
        help="Path to the simulator binary, or 'remote' if you start the sim manually.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable faster training config (lower res + JPG + frame_skip).",
    )

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--obs-width", type=int, default=84)
    parser.add_argument("--obs-height", type=int, default=84)

    parser.add_argument("--total-timesteps", type=int, default=200_000)
    # Default to 1 env to avoid multiprocessing + graphics/GL edge cases.
    # You can increase this once everything is stable.
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments (VecEnv)")

    log_dir = "tensorboard_logs/JetRacer"
    i = 1
    while os.path.exists(log_dir):
        log_dir = f"tensorboard_logs/JetRacer_{i}"
        i += 1

    parser.add_argument("--log-dir", type=str, default=log_dir)
    parser.add_argument("--save-path", type=str, default="models/centerline_ppo.zip")

    # Donkey rendering is handled by the Unity simulator; SB3 won't render by default.
    parser.add_argument("--render", action="store_true", help="Call env.render() each step (usually no-op)")

    args = parser.parse_args()

    if args.render and args.n_envs != 1:
        raise RuntimeError("--render requires --n-envs 1.")

    if args.n_envs != 1:
        raise RuntimeError(
            "DonkeyCar parallel envs require multiple simulator instances on different ports. "
            "Keep --n-envs 1, or extend this script to offset ports per env once you have multi-sim working."
        )

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    except ModuleNotFoundError as e:  # pragma: no cover
        missing = getattr(e, "name", None)
        if missing == "torch":
            raise RuntimeError(
                "Missing dependency: torch. Stable-Baselines3 requires PyTorch. Install torch, then re-run."
            ) from e
        if missing == "stable_baselines3":
            raise RuntimeError("Missing dependency: stable-baselines3. Install stable-baselines3, then re-run.") from e
        raise

    env_fns = [build_env_fn(args) for _ in range(args.n_envs)]
    vec_env = DummyVecEnv(env_fns)

    # Record episode returns/lengths so TensorBoard can plot reward curves (rollout/ep_rew_mean).
    vec_env = VecMonitor(vec_env, filename=os.path.join(args.log_dir, "monitor.csv"))

    # SB3 logger
    # TensorBoard is optional; fall back to stdout-only if not installed.
    format_strings = ["stdout"]
    try:
        from torch.utils.tensorboard import SummaryWriter as _SummaryWriter  # noqa: F401

        format_strings.append("tensorboard")
    except Exception:
        print("NOTE: tensorboard not installed; continuing with stdout logging only.")

    sb3_logger = configure(folder=args.log_dir, format_strings=format_strings)

    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        verbose=1,
        seed=args.seed,
        policy_kwargs={"normalize_images": False},
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )
    model.set_logger(sb3_logger)

    viz_cb = TrainingVizCallback(render=args.render).sb3_callback()
    model.learn(total_timesteps=args.total_timesteps, callback=viz_cb)
    model.save(args.save_path)

    vec_env.close()
    print(f"Saved policy to: {args.save_path}")


if __name__ == "__main__":
    main()
