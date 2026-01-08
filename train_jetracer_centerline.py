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
import shutil
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


@dataclass(frozen=True)
class DonkeyTrackLimitRewardConfig:
    """Extra reward terms for enforcing 'stay on track' in DonkeyCar.

    Uses `cte` (cross-track error) from info.
    """

    # Consider off-track when |cte| > max_cte
    max_cte: float = 8.0

    # Per-step penalty while off-track (even before termination)
    offtrack_step_penalty: float = 5.0


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


class JetRacerRaceRewardWrapperTrackLimit(gym.Wrapper):
    """New reward variant: keep original shaping + explicitly enforce staying on track.

    This wrapper does NOT modify `JetRacerRaceRewardWrapper`.
    It adds a per-step penalty when the car is outside the drivable area, estimated by
    `abs(cte) > max_cte`.

    Info is stored under `info["race_reward_track"]`.
    """

    def __init__(
        self,
        env: gym.Env,
        base_cfg: RaceRewardConfig = RaceRewardConfig(),
        track_cfg: DonkeyTrackLimitRewardConfig = DonkeyTrackLimitRewardConfig(),
    ):
        super().__init__(env)
        self.base_cfg = base_cfg
        self.track_cfg = track_cfg
        self._prev_steering: float = 0.0

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, _base_reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, _base_reward, done, info = result
            terminated, truncated = bool(done), False

        info_dict = dict(info) if isinstance(info, dict) else {}

        cte = info_dict.get("cte")
        speed = info_dict.get("speed", 0.0)
        hit = info_dict.get("hit")

        throttle, steering = self._extract_jetracer_raw_action()

        # Reuse the same structure as the original reward
        shaped_reward = 0.0

        if cte is not None:
            dist = float(cte)
            angle_rad = 0.0
            dot_dir = 1.0

            progress = max(0.0, dot_dir) * float(speed)
            shaped_reward += self.base_cfg.w_progress * progress
            shaped_reward += self.base_cfg.w_speed * float(speed)
            shaped_reward -= self.base_cfg.w_center * abs(dist)
            shaped_reward -= self.base_cfg.w_heading * abs(angle_rad)
        else:
            shaped_reward -= 1.0

        shaped_reward -= self.base_cfg.w_steer * float(steering**2)
        shaped_reward -= self.base_cfg.w_steer_rate * float((steering - self._prev_steering) ** 2)
        self._prev_steering = float(steering)

        # New: explicit track constraint
        offtrack = False
        if cte is not None:
            offtrack = abs(float(cte)) > float(self.track_cfg.max_cte)
            if offtrack:
                shaped_reward -= float(self.track_cfg.offtrack_step_penalty)

        # Terminal penalty (same spirit as original)
        if done:
            if offtrack or (isinstance(hit, str) and hit != "none"):
                shaped_reward -= float(self.base_cfg.offroad_penalty)

        info_dict["race_reward_track"] = {
            "reward": float(shaped_reward),
            "speed": float(speed),
            "throttle": float(throttle),
            "steering": float(steering),
            "cte": None if cte is None else float(cte),
            "max_cte": float(self.track_cfg.max_cte),
            "offtrack": bool(offtrack),
            "hit": hit,
        }

        if len(result) == 5:
            return obs, float(shaped_reward), bool(terminated), bool(truncated), info_dict
        return obs, float(shaped_reward), bool(done), info_dict

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
                        extra = None
                        if isinstance(infos[0], dict):
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
    """Save the best model based on *training* episode reward.

    This is a robust alternative when you cannot run a separate evaluation environment
    (e.g., you only run a single DonkeySim instance).
    """

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


def make_donkey_env(
    env_id: str,
    host: str,
    port: int,
    exe_path: str,
    *,
    fast_mode: bool,
    max_cte: float,
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
        "max_cte": float(max_cte),
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


def build_env_fn(
    args: argparse.Namespace,
    *,
    host: str,
    port: int,
    exe_path: str,
) -> Callable[[], gym.Env]:
    def _thunk() -> gym.Env:
        env = make_donkey_env(
            env_id=args.env_id,
            host=host,
            port=port,
            exe_path=exe_path,
            fast_mode=args.fast,
            max_cte=args.max_cte,
        )
        env = JetRacerWrapper(env, steer_scale=1.0, throttle_scale=1.0)
        if args.reward_type == "base":
            env = JetRacerRaceRewardWrapper(env)
        elif args.reward_type == "track_limit":
            env = JetRacerRaceRewardWrapperTrackLimit(
                env,
                base_cfg=RaceRewardConfig(),
                track_cfg=DonkeyTrackLimitRewardConfig(
                    max_cte=args.max_cte,
                    offtrack_step_penalty=args.offtrack_step_penalty,
                ),
            )
        else:
            raise ValueError(f"Unknown reward_type: {args.reward_type}")
        env = ResizeNormalizeObs(env, width=args.obs_width, height=args.obs_height)
        return env

    return _thunk


def _default_log_dir(base: str = "tensorboard_logs/JetRacer") -> str:
    log_dir = base
    i = 1
    while os.path.exists(log_dir):
        log_dir = f"{base}_{i}"
        i += 1
    return log_dir


def parse_args() -> argparse.Namespace:
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
        default="/home/supercomputing/studys/DonkeySim/DonkeySimLinux/donkey_sim.x86_64",
        help="Path to the simulator binary, or 'remote' if you start the sim manually.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable faster training config (lower res + JPG + frame_skip).",
    )

    # Reward selection (do NOT change the original reward; select via this flag)
    parser.add_argument(
        "--reward-type",
        type=str,
        default="base",
        choices=["base", "track_limit"],
        help="Reward function: base (original) or track_limit (adds explicit off-track penalty).",
    )
    parser.add_argument(
        "--max-cte",
        type=float,
        default=8.0,
        help="DonkeyCar max_cte threshold used to define off-track (|cte| > max_cte).",
    )
    parser.add_argument(
        "--offtrack-step-penalty",
        type=float,
        default=5.0,
        help="Per-step penalty while off-track (only used for reward_type=track_limit).",
    )

    # Best model saving
    parser.add_argument(
        "--best-model-path",
        type=str,
        default="models/best_model.zip",
        help="Where to copy the best model checkpoint.",
    )

    # Optional evaluation (requires a SECOND DonkeySim instance on a different port)
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Enable EvalCallback. Requires a separate eval simulator instance on eval-host/eval-port.",
    )
    parser.add_argument("--eval-host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--eval-port",
        type=int,
        default=0,
        help="Eval simulator port. Default is train port + 1.",
    )
    parser.add_argument(
        "--eval-exe-path",
        type=str,
        default="remote",
        help="Eval simulator binary path or 'remote' if you start eval sim manually.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Evaluate every N environment steps (only when --eval is set).",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=3,
        help="Number of episodes per evaluation (only when --eval is set).",
    )

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--obs-width", type=int, default=84)
    parser.add_argument("--obs-height", type=int, default=84)

    parser.add_argument("--total-timesteps", type=int, default=200_000)
    # Default to 1 env to avoid multiprocessing + graphics/GL edge cases.
    # You can increase this once everything is stable.
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments (VecEnv)")

    parser.add_argument("--log-dir", type=str, default=_default_log_dir())
    parser.add_argument("--save-path", type=str, default="models/centerline_ppo.zip")

    # Donkey rendering is handled by the Unity simulator; SB3 won't render by default.
    parser.add_argument("--render", action="store_true", help="Call env.render() each step (usually no-op)")

    args = parser.parse_args()
    if args.eval_port == 0:
        args.eval_port = int(args.port) + 1
    return args


def _sync_best_model(best_zip_path: str, target_best_path: str) -> None:
    if not os.path.exists(best_zip_path):
        return
    os.makedirs(os.path.dirname(target_best_path) or ".", exist_ok=True)
    shutil.copy2(best_zip_path, target_best_path)


def main() -> None:
    args = parse_args()

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
        from stable_baselines3.common.callbacks import CallbackList, EvalCallback
        from stable_baselines3.common.evaluation import evaluate_policy
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

    train_env = DummyVecEnv([build_env_fn(args, host=args.host, port=args.port, exe_path=args.exe_path)])

    # Record episode returns/lengths so TensorBoard can plot reward curves (rollout/ep_rew_mean).
    train_env = VecMonitor(train_env, filename=os.path.join(args.log_dir, "monitor.csv"))

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
        env=train_env,
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

    callbacks = [TrainingVizCallback(render=args.render).sb3_callback()]

    best_zip_in_log = os.path.join(args.log_dir, "best", "best_model.zip")
    eval_callback = None
    eval_env = None

    if args.eval:
        # IMPORTANT: this requires a separate DonkeySim instance bound to eval-port.
        eval_env = DummyVecEnv([build_env_fn(args, host=args.eval_host, port=args.eval_port, exe_path=args.eval_exe_path)])
        eval_env = VecMonitor(eval_env, filename=os.path.join(args.log_dir, "eval_monitor.csv"))
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(args.log_dir, "best"),
            log_path=os.path.join(args.log_dir, "eval"),
            eval_freq=int(args.eval_freq),
            n_eval_episodes=int(args.n_eval_episodes),
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)
    else:
        # Fallback: save best model based on training episode reward.
        callbacks.append(BestModelOnEpisodeRewardCallback(args.best_model_path).sb3_callback())

    cb = CallbackList(callbacks)

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=cb)
    except KeyboardInterrupt:
        # Ensure we persist something useful on Ctrl+C.
        print("Interrupted. Saving last model...")
        model.save(args.save_path)

        # If using eval, force a final evaluation so best model is updated at least once.
        if args.eval and eval_callback is not None and eval_env is not None:
            try:
                mean_reward, _std = evaluate_policy(
                    model,
                    eval_env,
                    n_eval_episodes=int(max(1, args.n_eval_episodes)),
                    deterministic=True,
                )
                if float(mean_reward) > float(eval_callback.best_mean_reward):
                    os.makedirs(os.path.dirname(best_zip_in_log), exist_ok=True)
                    model.save(best_zip_in_log)
            except Exception:
                pass

        # Copy best model to user-specified path if available.
        _sync_best_model(best_zip_in_log, args.best_model_path)
        raise
    finally:
        # Normal completion or after interrupt.
        if os.path.exists(best_zip_in_log):
            _sync_best_model(best_zip_in_log, args.best_model_path)

        if eval_env is not None:
            eval_env.close()
        train_env.close()

    model.save(args.save_path)
    _sync_best_model(best_zip_in_log, args.best_model_path)
    print(f"Saved last model to: {args.save_path}")
    if os.path.exists(args.best_model_path):
        print(f"Saved best model to: {args.best_model_path}")


if __name__ == "__main__":
    main()
