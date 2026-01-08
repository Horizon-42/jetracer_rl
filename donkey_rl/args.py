from __future__ import annotations

import argparse
import os


def default_log_dir(base: str = "tensorboard_logs/JetRacer") -> str:
    log_dir = base
    i = 1
    while os.path.exists(log_dir):
        log_dir = f"{base}_{i}"
        i += 1
    return log_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DonkeyCar policy (JetRacer-style actions).")

    # Donkey connection
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

    # Speed knobs
    parser.add_argument("--fast", action="store_true", help="Enable faster config (lower res + JPG + frame_skip).")

    # Reward selection
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
        help="Off-track threshold: |cte| > max_cte.",
    )
    parser.add_argument(
        "--offtrack-step-penalty",
        type=float,
        default=5.0,
        help="Per-step penalty while off-track (reward_type=track_limit).",
    )

    # Training
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-width", type=int, default=84)
    parser.add_argument("--obs-height", type=int, default=84)
    parser.add_argument("--total-timesteps", type=int, default=200_000)

    # Saving/logging
    parser.add_argument("--log-dir", type=str, default=default_log_dir())
    parser.add_argument("--save-path", type=str, default="models/centerline_ppo.zip")
    parser.add_argument("--best-model-path", type=str, default="models/best_model.zip")

    # Rendering
    parser.add_argument("--render", action="store_true", help="Call env.render() each step (usually no-op)")

    # Evaluation (requires a second sim)
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Enable EvalCallback. Requires a separate eval sim on eval-host/eval-port.",
    )
    parser.add_argument("--eval-host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--eval-port",
        type=int,
        default=0,
        help="Eval simulator port. Default: train port + 1.",
    )
    parser.add_argument("--eval-exe-path", type=str, default="remote")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=3)

    args = parser.parse_args()
    if args.eval_port == 0:
        args.eval_port = int(args.port) + 1
    return args
