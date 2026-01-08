from __future__ import annotations

import argparse
import os


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name).strip("_")


def _next_jetracer_id(*, models_root: str, log_root: str, prefix: str = "JetRacer") -> str:
    """Return a run id like `JetRacer_1`, `JetRacer_2`, ...

    We pick the smallest i such that BOTH folders don't already exist:
      models/<id>/ and tensorboard_logs/<id>/

    Note: If you start multiple runs at exactly the same time, there is still a small
    chance of a race condition. In practice this is usually fine.
    """

    i = 1
    while True:
        run_id = f"{prefix}_{i}"
        if not os.path.exists(os.path.join(models_root, run_id)) and not os.path.exists(os.path.join(log_root, run_id)):
            return run_id
        i += 1


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

    # Domain randomization (sim -> real): photometric + action dynamics
    parser.add_argument(
        "--domain-rand",
        action="store_true",
        help="Enable photometric augmentation (brightness/contrast/noise) on sim observations.",
    )
    parser.add_argument("--aug-brightness", type=float, default=0.25, help="Brightness jitter amplitude (in [0,1]).")
    parser.add_argument("--aug-contrast", type=float, default=0.25, help="Contrast jitter amplitude (multiplier around 1.0).")
    parser.add_argument("--aug-noise-std", type=float, default=0.02, help="Gaussian noise std (in [0,1]).")

    parser.add_argument(
        "--random-friction",
        action="store_true",
        help="Enable per-episode random friction (implemented as throttle scaling).",
    )
    parser.add_argument("--friction-min", type=float, default=0.6, help="Min throttle scale when random friction enabled.")
    parser.add_argument("--friction-max", type=float, default=1.0, help="Max throttle scale when random friction enabled.")

    # Latent features (autoencoder)
    parser.add_argument(
        "--use-latent",
        action="store_true",
        help="Use a pretrained autoencoder encoder as features extractor (MlpPolicy).",
    )
    parser.add_argument(
        "--ae-checkpoint",
        type=str,
        default="",
        help="Path to AE checkpoint produced by train_autoencoder.py (e.g. ae_runs/ae_*/best_ae.pt).",
    )
    parser.add_argument(
        "--train-encoder",
        action="store_true",
        help="If set, finetune encoder together with PPO (default: freeze encoder).",
    )

    # Debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: run a short training and dump observation images.",
    )
    parser.add_argument(
        "--debug-steps",
        type=int,
        default=500,
        help="Number of timesteps to run when --debug is enabled.",
    )
    parser.add_argument(
        "--debug-obs-every",
        type=int,
        default=10,
        help="Save one observation image every N steps in debug mode.",
    )

    # Saving/logging
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name tag for this run (used in output folder name).",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default="models",
        help="Root directory for saving models. A unique subfolder is created per run.",
    )
    parser.add_argument(
        "--log-root",
        type=str,
        default="tensorboard_logs",
        help="Root directory for TensorBoard logs. A unique subfolder is created per run.",
    )

    # If you provide these explicitly, we will respect them. Otherwise we create
    # per-run defaults under models-root/log-root to avoid collisions.
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--best-model-path", type=str, default=None)

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

    # Per-run output directories (prevents interfering when launching multiple trainings)
    if args.run_name:
        run_id = _sanitize_name(str(args.run_name))
        if not run_id:
            run_id = _next_jetracer_id(models_root=args.models_root, log_root=args.log_root)
    else:
        run_id = _next_jetracer_id(models_root=args.models_root, log_root=args.log_root)

    args.run_id = run_id
    args.run_dir = os.path.join(args.models_root, run_id)

    if args.log_dir is None:
        args.log_dir = os.path.join(args.log_root, run_id)
    if args.save_path is None:
        args.save_path = os.path.join(args.run_dir, "last_model.zip")
    if args.best_model_path is None:
        args.best_model_path = os.path.join(args.run_dir, "best_model.zip")

    # Debug artifacts directory
    args.debug_obs_dir = os.path.join(args.run_dir, "debug_obs")
    return args
