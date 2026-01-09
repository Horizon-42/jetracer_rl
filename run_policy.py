from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Action:
    throttle: float
    steering: float


def _clip_action(a: Action) -> Action:
    thr = float(np.clip(a.throttle, 0.0, 1.0))
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
            "Run a trained SB3 policy in either DonkeySim (sim) or on JetRacer camera input (real).\n"
            "The policy action format is [throttle, steering]."
        )
    )

    p.add_argument("--model", type=str, required=True, help="Path to SB3 .zip model (e.g. models/<run>/best_model.zip)")
    p.add_argument("--mode", type=str, required=True, choices=["sim", "real"], help="Where to run the policy")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic actions (recommended for eval)")

    # Sim options
    p.add_argument("--env-id", type=str, default="donkey-waveshare-v0")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=9091)
    p.add_argument(
        "--exe-path",
        type=str,
        default="/home/supercomputing/studys/DonkeySim/DonkeySimLinux/donkey_sim.x86_64",
        help="Path to simulator binary or 'remote' if you start it manually.",
    )
    p.add_argument("--steps", type=int, default=0, help="If >0, stop after N environment steps")
    p.add_argument("--render", action="store_true", help="Call env.render() each step (may be no-op)")
    p.add_argument("--max-cte", type=float, default=8.0, help="Off-track threshold in sim (passed to env conf)")

    # Real options
    p.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture device index")
    p.add_argument("--fps", type=float, default=15.0, help="Control loop FPS")
    p.add_argument("--show", action="store_true", help="Show camera preview window (requires display)")
    p.add_argument("--dry-run", action="store_true", help="Do not drive motors; just print actions")

    # Preprocess options (should match training)
    p.add_argument("--obs-width", type=int, default=84)
    p.add_argument("--obs-height", type=int, default=84)

    return p.parse_args()


def _make_sim_env(*, env_id: str, host: str, port: int, exe_path: str, obs_width: int, obs_height: int, max_cte: float):
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

    conf = {
        "exe_path": exe_path,
        "host": host,
        "port": int(port),
        "max_cte": float(max_cte),
        "body_style": "donkey",
        "body_rgb": (255, 165, 0),
        "car_name": "JetRacerRunner",
        "font_size": 100,
        "cam_resolution": (240, 320, 3),
        "cam_config": {"img_w": 320, "img_h": 240, "img_d": 3, "img_enc": "JPG"},
    }

    env = gym.make("GymV21Environment-v0", env_id=env_id, make_kwargs={"conf": conf})
    env = JetRacerWrapper(env)
    # Reward wrapper doesn't matter for inference, but keeps info consistent.
    env = JetRacerRaceRewardWrapper(env, cfg=RaceRewardConfig())
    env = ObsPreprocess(env, width=obs_width, height=obs_height, domain_rand=False)
    return env


class _JetRacerActuator:
    """Best-effort JetRacer motor actuator.

    If `jetracer` is installed on the robot, this will drive the car.
    Otherwise, it falls back to a dry-run (printing actions).
    """

    def __init__(self, *, dry_run: bool):
        self._dry_run = bool(dry_run)
        self._car = None
        if not self._dry_run:
            try:
                from jetracer.nvidia_racecar import NvidiaRacecar  # type: ignore

                self._car = NvidiaRacecar()
            except Exception:
                self._car = None
                self._dry_run = True

    def apply(self, action: Action) -> None:
        action = _clip_action(action)
        if self._dry_run or self._car is None:
            print(f"action throttle={action.throttle:.3f} steer={action.steering:.3f}")
            return

        # NvidiaRacecar expects:
        #   steering in [-1,1]
        #   throttle in [-1,1] (usually)
        # We restrict to forward-only here.
        self._car.steering = float(action.steering)
        self._car.throttle = float(action.throttle)

    def stop(self) -> None:
        if self._car is not None:
            try:
                self._car.throttle = 0.0
            except Exception:
                pass


def _run_sim(args: argparse.Namespace) -> None:
    env = _make_sim_env(
        env_id=args.env_id,
        host=args.host,
        port=int(args.port),
        exe_path=str(args.exe_path),
        obs_width=int(args.obs_width),
        obs_height=int(args.obs_height),
        max_cte=float(getattr(args, "max_cte", 8.0)),
    )

    model = _load_sb3_model(str(args.model))

    obs, _info = env.reset()
    steps = 0

    try:
        while True:
            action = _predict_action(model, obs, deterministic=bool(args.deterministic))
            obs, _reward, terminated, truncated, _info = env.step(np.array([action.throttle, action.steering], dtype=np.float32))
            steps += 1

            if args.render:
                try:
                    env.render()
                except Exception:
                    pass

            if terminated or truncated:
                obs, _info = env.reset()

            if int(args.steps) > 0 and steps >= int(args.steps):
                break

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def _run_real(args: argparse.Namespace) -> None:
    import cv2

    from donkey_rl.real_obs_preprocess import preprocess_real_frame_bgr_to_chw_float01

    model = _load_sb3_model(str(args.model))
    actuator = _JetRacerActuator(dry_run=bool(args.dry_run))

    cap = cv2.VideoCapture(int(args.camera))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open camera device: {args.camera}")

    dt = 1.0 / max(1e-6, float(args.fps))

    try:
        while True:
            t0 = time.time()
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue

            obs = preprocess_real_frame_bgr_to_chw_float01(
                frame_bgr,
                out_width=int(args.obs_width),
                out_height=int(args.obs_height),
            )

            action = _predict_action(model, obs, deterministic=bool(args.deterministic))
            actuator.apply(action)

            if args.show:
                cv2.imshow("camera", frame_bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        actuator.stop()
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def main() -> None:
    args = _parse_args()

    if args.mode == "sim":
        _run_sim(args)
    else:
        _run_real(args)


if __name__ == "__main__":
    main()
