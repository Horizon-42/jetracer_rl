from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Action:
    throttle: float
    steering: float


def _clip_action(a: Action) -> Action:
    thr = float(np.clip(a.throttle, 0, 1.0))
    steer = float(np.clip(a.steering, -1.0, 1.0))
    return Action(throttle=thr, steering=steer)


def _load_sb3_model(path: str):
    if not os.path.exists(path):
        raise SystemExit(f"Model not found: {path}")
    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as e:
        raise SystemExit("stable-baselines3 not installed in this environment") from e

    return PPO.load(path, device="auto")


def _predict_action(model, obs: np.ndarray, *, deterministic: bool = True) -> Action:
    act, _state = model.predict(obs, deterministic=deterministic)
    act = np.asarray(act, dtype=np.float32).reshape(-1)
    if act.shape[0] != 2:
        raise RuntimeError(f"Expected 2-dim action [throttle, steering], got shape {act.shape}")
    return _clip_action(Action(throttle=float(act[0]), steering=float(act[1])))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run a trained SB3 policy in DonkeySim simulator.\n"
            "The policy action format is [throttle, steering]."
        )
    )

    p.add_argument("--model", type=str, required=True, help="Path to SB3 .zip model (e.g. models/<run>/best_model.zip)")
    p.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions (recommended for eval)")

    p.add_argument("--env-id", type=str, default="donkey-waveshare-v0", help="DonkeySim environment ID")
    p.add_argument("--host", type=str, default="127.0.0.1", help="Simulator host address")
    p.add_argument("--port", type=int, default=9091, help="Simulator port")
    p.add_argument(
        "--exe-path",
        type=str,
        default="/home/supercomputing/studys/DonkeySim/DonkeySimLinux/donkey_sim.x86_64",
        help="Path to simulator binary or 'remote' if you start it manually.",
    )
    p.add_argument("--steps", type=int, default=0, help="If >0, stop after N environment steps")
    p.add_argument("--render", action="store_true", help="Call env.render() each step (may be no-op)")
    p.add_argument("--max-cte", type=float, default=3.0, help="Off-track threshold (passed to env conf)")

    p.add_argument("--obs-width", type=int, default=84, help="Observation width (should match training)")
    p.add_argument("--obs-height", type=int, default=84, help="Observation height (should match training)")
    p.add_argument(
        "--perspective-transform",
        action="store_false",
        default=True,
        help="Enable perspective transformation (bird's-eye view) preprocessing (should match training)",
    )

    return p.parse_args()


def _make_sim_env(
    *,
    env_id: str,
    host: str,
    port: int,
    exe_path: str,
    obs_width: int,
    obs_height: int,
    max_cte: float,
    perspective_transform: bool,
):
    from donkey_rl.compat import patch_gym_donkeycar_stop_join, patch_old_gym_render_mode

    patch_old_gym_render_mode()
    patch_gym_donkeycar_stop_join()

    import gymnasium as gym

    # Ensure old-gym envs are registered.
    import gym_donkeycar  # noqa: F401
    import shimmy  # noqa: F401

    from donkey_rl.obs_preprocess import ObsPreprocess
    from donkey_rl.rewards import JetRacerRaceRewardWrapper, RaceRewardConfig
    from donkey_rl.wrappers import JetRacerWrapper
    from donkey_rl.wrappers import StepTimeoutWrapper

    conf = {
        "exe_path": exe_path,
        "host": host,
        "port": port,
        "max_cte": max_cte,
        "body_style": "donkey",
        "body_rgb": (255, 165, 0),
        "car_name": "JetRacerRunner",
        "font_size": 100,
        "cam_resolution": (240, 320, 3),
        "cam_config": {"img_w": 320, "img_h": 240, "img_d": 3, "img_enc": "JPG"},
    }

    env = gym.make("GymV21Environment-v0", env_id=env_id, make_kwargs={"conf": conf})
    env = StepTimeoutWrapper(env, timeout_s=30.0)
    env = JetRacerWrapper(env)
    # Reward wrapper doesn't matter for inference, but keeps info consistent.
    env = JetRacerRaceRewardWrapper(env, cfg=RaceRewardConfig())
    env = ObsPreprocess(
        env,
        width=obs_width,
        height=obs_height,
        domain_rand=False,
        perspective_transform=perspective_transform,
    )
    return env


def _run_policy(args: argparse.Namespace) -> None:
    env = _make_sim_env(
        env_id=args.env_id,
        host=args.host,
        port=args.port,
        exe_path=args.exe_path,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
        max_cte=args.max_cte,
        perspective_transform=args.perspective_transform,
    )

    model = _load_sb3_model(args.model)

    obs, _info = env.reset()
    steps = 0

    try:
        while True:
            action = _predict_action(model, obs, deterministic=args.deterministic)
            obs, _reward, terminated, truncated, _info = env.step(
                np.array([action.throttle, action.steering], dtype=np.float32)
            )
            steps += 1

            if args.render:
                try:
                    env.render()
                except Exception:
                    pass

            if terminated or truncated:
                obs, _info = env.reset()

            if args.steps > 0 and steps >= args.steps:
                break

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def main() -> None:
    args = _parse_args()
    _run_policy(args)


if __name__ == "__main__":
    main()
