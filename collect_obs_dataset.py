from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import numpy as np

from donkey_rl.compat import patch_gym_donkeycar_stop_join, patch_old_gym_render_mode
from donkey_rl.obs_preprocess import ObsPreprocess
from donkey_rl.wrappers import JetRacerWrapper


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect simulator frames for autoencoder training.")

    p.add_argument("--env-id", type=str, default="donkey-waveshare-v0")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=9091)
    p.add_argument(
        "--exe-path",
        type=str,
        default="/home/supercomputing/studys/DonkeySim/DonkeySimLinux/donkey_sim.x86_64",
        help="Path to simulator binary, or 'remote' if you start it manually.",
    )

    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--save-every", type=int, default=1, help="Save one frame every N steps.")
    p.add_argument("--warmup-steps", type=int, default=50)

    p.add_argument("--obs-width", type=int, default=84)
    p.add_argument("--obs-height", type=int, default=84)

    p.add_argument(
        "--out-dir",
        type=str,
        default="datasets/obs_ae",
        help="Output directory for frames.",
    )
    p.add_argument(
        "--format",
        type=str,
        default="npy",
        choices=["npy", "png"],
        help="Storage format. npy is fastest/lossless.",
    )

    p.add_argument(
        "--action-mode",
        type=str,
        default="random",
        choices=["random", "straight"],
        help="Action policy while collecting.",
    )
    p.add_argument("--throttle", type=float, default=0.4, help="Used when action-mode=straight.")
    p.add_argument("--steer", type=float, default=0.0, help="Used when action-mode=straight.")

    return p.parse_args()


def _make_env(*, env_id: str, host: str, port: int, exe_path: str, obs_width: int, obs_height: int):
    patch_old_gym_render_mode()
    patch_gym_donkeycar_stop_join()

    import gymnasium as gym

    # Ensure old-gym envs are registered.
    import gym_donkeycar  # noqa: F401
    import shimmy  # noqa: F401

    conf = {
        "exe_path": exe_path,
        "host": host,
        "port": int(port),
        # Some DonkeySim scenes expect these keys in the car config payload.
        "max_cte": 8.0,
        "body_style": "donkey",
        "body_rgb": (255, 165, 0),
        "car_name": "JetRacerCollector",
        "font_size": 100,
        # Keep Gym's expected observation shape consistent with what we request from the sim.
        "cam_resolution": (240, 320, 3),
        "cam_config": {
            "img_w": 320,
            "img_h": 240,
            "img_d": 3,
            "img_enc": "JPG",
        },
    }

    env = gym.make(
        "GymV21Environment-v0",
        env_id=env_id,
        make_kwargs={"conf": conf},
    )

    env = JetRacerWrapper(env, steer_scale=1.0, throttle_scale=1.0)
    env = ObsPreprocess(env, width=obs_width, height=obs_height)
    return env


def _save_frame(*, out_dir: str, index: int, frame_rgb_u8: np.ndarray, fmt: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    if fmt == "npy":
        path = os.path.join(out_dir, f"frame_{index:08d}.npy")
        np.save(path, frame_rgb_u8)
        return path

    # png
    import cv2

    path = os.path.join(out_dir, f"frame_{index:08d}.png")
    bgr = cv2.cvtColor(frame_rgb_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)
    return path


def main() -> None:
    args = _parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "time": time.time()}, f, ensure_ascii=False, indent=2)

    env = _make_env(
        env_id=args.env_id,
        host=args.host,
        port=args.port,
        exe_path=args.exe_path,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )

    obs, info = env.reset()
    _ = (obs, info)

    saved = 0
    for step in range(int(args.steps)):
        if args.action_mode == "random":
            action = env.action_space.sample()
        else:
            action = np.array([float(args.throttle), float(args.steer)], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        _ = (obs, reward, info)

        if step >= int(args.warmup_steps) and (step % int(args.save_every) == 0):
            frame = getattr(env, "last_resized_observation", None)
            if frame is not None:
                _save_frame(out_dir=args.out_dir, index=saved, frame_rgb_u8=np.asarray(frame), fmt=str(args.format))
                saved += 1

        if terminated or truncated:
            env.reset()

        if (step + 1) % 1000 == 0:
            print(f"step={step+1} saved={saved}")

    env.close()
    print(f"Done. Saved {saved} frames to {args.out_dir}")


if __name__ == "__main__":
    main()
