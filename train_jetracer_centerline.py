"""Train a DonkeyCar (Unity sim) policy with a JetRacer-style interface.

This is the entrypoint script. Most logic lives in small modules under
the `donkey_rl/` package for easier learning and editing.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

from donkey_rl.callbacks import BestModelOnEpisodeRewardCallback, DebugObsDumpCallback, TrainingVizCallback
from donkey_rl.env import build_env_fn
from donkey_rl.eval_callback import EphemeralEvalCallback

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

    parser.add_argument(
        "--sim-io-timeout-s",
        type=float,
        default=30.0,
        help="Abort hanging sim step/reset after N seconds (0 disables). Helps when Unity disconnects.",
    )

    # Speed knobs
    parser.add_argument("--fast", action="store_true", help="Enable faster config (lower res + JPG + frame_skip).")

    # Reward selection
    parser.add_argument(
        "--reward-type",
        type=str,
        default="base",
        choices=["base", "track_limit", "deepracer", "centerline_v2", "centerline_v3", "centerline_v4"],
        help="Reward function: base, track_limit, deepracer, centerline_v2, centerline_v3, or centerline_v4 (center+speed+smooth+alive+anti-stall).",
    )
    parser.add_argument(
        "--max-cte",
        type=float,
        default=3.0,
        help="Off-track threshold: |cte| > max_cte.",
    )
    parser.add_argument(
        "--offtrack-step-penalty",
        type=float,
        default=5.0,
        help="Per-step penalty while off-track (reward_type=track_limit).",
    )

    # Reward tuning knobs for centerline_v2 (keep it minimal)
    parser.add_argument("--v2-w-speed", type=float, default=0.8, help="centerline_v2: speed reward weight")
    parser.add_argument("--v2-w-caution", type=float, default=0.6, help="centerline_v2: caution penalty weight")
    parser.add_argument("--v2-min-speed", type=float, default=0.2, help="centerline_v2: anti-stall minimum speed")

    # Reward tuning knobs for centerline_v3 (even simpler, stronger anti-stall)
    parser.add_argument("--v3-w-speed", type=float, default=1.2, help="centerline_v3: speed reward weight")
    parser.add_argument("--v3-min-speed", type=float, default=0.35, help="centerline_v3: minimum desired speed")
    parser.add_argument("--v3-w-stall", type=float, default=2.0, help="centerline_v3: penalty weight for speed below v3-min-speed")
    parser.add_argument("--v3-alive-bonus", type=float, default=0.02, help="centerline_v3: small per-step bonus")

    # Reward tuning knobs for centerline_v4 (simple + smooth + anti-stall)
    parser.add_argument("--v4-w-speed", type=float, default=1.0, help="centerline_v4: speed reward weight")
    parser.add_argument("--v4-w-smooth", type=float, default=0.25, help="centerline_v4: smoothness penalty weight")
    parser.add_argument("--v4-min-speed", type=float, default=0.25, help="centerline_v4: minimum desired speed")
    parser.add_argument("--v4-w-stall", type=float, default=3.0, help="centerline_v4: penalty weight for speed below v4-min-speed")
    parser.add_argument("--v4-alive-bonus", type=float, default=0.03, help="centerline_v4: small per-step bonus")

    # Training
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-width", type=int, default=84)
    parser.add_argument("--obs-height", type=int, default=84)
    parser.add_argument("--total-timesteps", type=int, default=200_000)

    # Domain randomization (sim -> real): photometric + action dynamics
    parser.add_argument(
        "--domain-rand",
        action="store_false",
        default=True,
        help="Disable photometric augmentation (brightness/contrast/noise/color) on sim observations (enabled by default).",
    )
    parser.add_argument("--aug-brightness", type=float, default=0.4, help="Brightness jitter amplitude (in [0,1]).")
    parser.add_argument("--aug-contrast", type=float, default=0.4, help="Contrast jitter amplitude (multiplier around 1.0).")
    parser.add_argument("--aug-noise-std", type=float, default=0.05, help="Gaussian noise std (in [0,1]).")
    parser.add_argument("--aug-color-jitter", type=float, default=0.35, help="Color jitter amplitude for HSV adjustments (hue/saturation/value).")

    parser.add_argument(
        "--random-friction",
        action="store_false",
        default=True,
        help="Disable per-episode random friction (enabled by default, implemented as throttle scaling).",
    )
    parser.add_argument("--friction-min", type=float, default=0.4, help="Min throttle scale when random friction enabled.")
    parser.add_argument("--friction-max", type=float, default=1.2, help="Max throttle scale when random friction enabled.")

    # Stall detection (prevents car from learning to stop)
    parser.add_argument(
        "--stall-detection",
        action="store_false",
        default=True,
        help="Disable stall detection (enabled by default). Terminates episode if car stops moving.",
    )
    parser.add_argument("--stall-speed-threshold", type=float, default=0.1, help="Speed below this is considered stalled.")
    parser.add_argument("--stall-max-steps", type=int, default=50, help="Terminate after this many consecutive stalled steps.")
    parser.add_argument("--stall-penalty", type=float, default=20.0, help="Penalty when episode terminates due to stall.")

    parser.add_argument(
        "--perspective-transform",
        action="store_false",
        default=True,
        help="Enable perspective transform preprocessing on observations (disabled by default). Deprecated: use --obs-mode instead.",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="mix",
        choices=["auto", "raw", "perspective", "mix"],
        help="Observation mode: 'auto' (use --perspective-transform flag), 'raw' (original image only), "
             "'perspective' (bird's-eye view only), 'mix' (stack raw+perspective vertically, compress to 84x84).",
    )

    # Loading existing policy for continued training
    parser.add_argument(
        "--load-model",
        type=str,
        default="",
        help="Path to existing model checkpoint to load and continue training from.",
    )

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
        default=True,
        help="Enable EvalCallback. Requires a separate eval sim on eval-host/eval-port.",
    )
    parser.add_argument("--eval-host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--eval-port",
        type=int,
        default=0,
        help="Eval simulator port. Default: train port + 6.",
    )
    parser.add_argument("--eval-freq", type=int, default=20_000, help="EvalCallback eval frequency. Unit depends on --eval-freq-type.")
    parser.add_argument(
        "--eval-freq-type",
        type=str,
        default="timesteps",
        choices=["timesteps", "episodes"],
        help="Unit for --eval-freq: 'timesteps' (default, recommended) or 'episodes'.",
    )
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument(
        "--max-eval-episode-steps",
        type=int,
        default=5000,
        help="Maximum number of steps per episode during evaluation. If None, no limit is applied. "
             "Episodes will be truncated (not terminated) when reaching this limit. "
             "Recommended: 5000-10000 steps to prevent infinite episodes when model performs well or stalls.",
    )

    args = parser.parse_args()

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


def _sync_best_model(best_zip_path: str, target_best_path: str) -> None:
    if not os.path.exists(best_zip_path):
        return
    os.makedirs(os.path.dirname(target_best_path) or ".", exist_ok=True)
    shutil.copy2(best_zip_path, target_best_path)


def _argv_has_flag(flag: str) -> bool:
    return flag in sys.argv


def _find_nearby_args_json(load_model_path: str) -> str:
    """Best-effort search for args.json near a model zip.

    Supports common layouts:
    - models/<run>/best_model.zip with models/<run>/args.json
    - tensorboard_logs/<run>/best/best_model.zip with tensorboard_logs/<run>/args.json
    """

    candidates = []
    d = os.path.dirname(os.path.abspath(load_model_path))
    for _ in range(4):
        candidates.append(os.path.join(d, "args.json"))
        d = os.path.dirname(d)
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


def _maybe_inherit_reward_args_from_loaded_model(args) -> None:
    """If --load-model is set, inherit reward settings from the source run.

    Behavior:
    - If the user explicitly provided a flag (e.g. --reward-type), we do NOT override it.
    - Otherwise, we read the saved run's args.json and copy reward-related fields.
    """

    load_model_path = str(getattr(args, "load_model", "") or "").strip()
    if not load_model_path:
        return

    args_json = _find_nearby_args_json(load_model_path)
    if not args_json:
        return

    try:
        with open(args_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        saved = payload.get("args", payload)
        if not isinstance(saved, dict):
            return
    except Exception:
        return

    # Reward selection
    if not _argv_has_flag("--reward-type") and "reward_type" in saved:
        try:
            args.reward_type = str(saved["reward_type"])
        except Exception:
            pass

    # Common reward-related knobs
    mapping = {
        "--max-cte": "max_cte",
        "--offtrack-step-penalty": "offtrack_step_penalty",
        "--v2-w-speed": "v2_w_speed",
        "--v2-w-caution": "v2_w_caution",
        "--v2-min-speed": "v2_min_speed",
        "--v3-w-speed": "v3_w_speed",
        "--v3-min-speed": "v3_min_speed",
        "--v3-w-stall": "v3_w_stall",
        "--v3-alive-bonus": "v3_alive_bonus",
        "--v4-w-speed": "v4_w_speed",
        "--v4-min-speed": "v4_min_speed",
        "--v4-w-stall": "v4_w_stall",
        "--v4-w-smooth": "v4_w_smooth",
        "--v4-alive-bonus": "v4_alive_bonus",
    }
    for flag, key in mapping.items():
        if _argv_has_flag(flag):
            continue
        if key not in saved:
            continue
        try:
            setattr(args, key, saved[key])
        except Exception:
            pass


def main() -> None:
    args = parse_args()

    # When continuing training from a checkpoint, inherit reward settings by default
    # to avoid accidental mismatches between the loaded policy and the new env.
    _maybe_inherit_reward_args_from_loaded_model(args)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.best_model_path) or ".", exist_ok=True)
    if getattr(args, "debug", False):
        os.makedirs(getattr(args, "debug_obs_dir", os.path.join(os.path.dirname(args.save_path), "debug_obs")), exist_ok=True)

    # Save the exact run configuration for reproducibility.
    # This prevents confusion when you run multiple trainings in parallel.
    try:
        run_dir = getattr(args, "run_dir", os.path.dirname(args.save_path) or ".")
        os.makedirs(run_dir, exist_ok=True)
        args_path = os.path.join(run_dir, "args.json")
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump({"argv": sys.argv, "args": vars(args)}, f, ensure_ascii=False, indent=2)
    except Exception:
        # Never block training due to logging issues.
        pass

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList, EvalCallback
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    except ModuleNotFoundError as e:  # pragma: no cover
        missing = getattr(e, "name", None)
        if missing == "torch":
            raise RuntimeError("Missing dependency: torch. Install torch, then re-run.") from e
        if missing == "stable_baselines3":
            raise RuntimeError("Missing dependency: stable-baselines3. Install it, then re-run.") from e
        raise

    train_env = DummyVecEnv(
        [
            build_env_fn(
                env_id=args.env_id,
                host=args.host,
                port=args.port,
                exe_path=args.exe_path,
                fast_mode=args.fast,
                reward_type=args.reward_type,
                max_cte=args.max_cte,
                offtrack_step_penalty=args.offtrack_step_penalty,
                v2_w_speed=float(getattr(args, "v2_w_speed", 0.8)),
                v2_w_caution=float(getattr(args, "v2_w_caution", 0.6)),
                v2_min_speed=float(getattr(args, "v2_min_speed", 0.2)),
                v3_w_speed=float(getattr(args, "v3_w_speed", 1.2)),
                v3_min_speed=float(getattr(args, "v3_min_speed", 0.35)),
                v3_w_stall=float(getattr(args, "v3_w_stall", 2.0)),
                v3_alive_bonus=float(getattr(args, "v3_alive_bonus", 0.02)),
                v4_w_speed=float(getattr(args, "v4_w_speed", 1.0)),
                v4_min_speed=float(getattr(args, "v4_min_speed", 0.25)),
                v4_w_stall=float(getattr(args, "v4_w_stall", 3.0)),
                v4_w_smooth=float(getattr(args, "v4_w_smooth", 0.25)),
                v4_alive_bonus=float(getattr(args, "v4_alive_bonus", 0.03)),
                sim_io_timeout_s=float(getattr(args, "sim_io_timeout_s", 30.0)),
                obs_width=args.obs_width,
                obs_height=args.obs_height,
                domain_rand=bool(getattr(args, "domain_rand", False)),
                perspective_transform=bool(getattr(args, "perspective_transform", False)),
                obs_mode=str(getattr(args, "obs_mode", "auto")),
                aug_brightness=float(getattr(args, "aug_brightness", 0.4)),
                aug_contrast=float(getattr(args, "aug_contrast", 0.4)),
                aug_noise_std=float(getattr(args, "aug_noise_std", 0.05)),
                aug_color_jitter=float(getattr(args, "aug_color_jitter", 0.35)),
                random_friction=bool(getattr(args, "random_friction", False)),
                friction_min=float(getattr(args, "friction_min", 0.4)),
                friction_max=float(getattr(args, "friction_max", 1.2)),
                stall_detection=bool(getattr(args, "stall_detection", True)),
                stall_speed_threshold=float(getattr(args, "stall_speed_threshold", 0.1)),
                stall_max_steps=int(getattr(args, "stall_max_steps", 50)),
                stall_penalty=float(getattr(args, "stall_penalty", 20.0)),
                car_name=str(getattr(args, "run_id", "JetRacerAgent")),
            )
        ]
    )
    train_env = VecMonitor(train_env, filename=os.path.join(args.log_dir, "monitor.csv"))

    format_strings = ["stdout"]
    try:
        from torch.utils.tensorboard import SummaryWriter as _SummaryWriter  # noqa: F401

        format_strings.append("tensorboard")
    except Exception:
        print("NOTE: tensorboard not installed; continuing with stdout logging only.")

    sb3_logger = configure(folder=args.log_dir, format_strings=format_strings)

    policy = "CnnPolicy"
    policy_kwargs = {"normalize_images": False}
    if getattr(args, "use_latent", False):
        ae_ckpt = str(getattr(args, "ae_checkpoint", "") or "").strip()
        if not ae_ckpt:
            raise RuntimeError("--use-latent requires --ae-checkpoint")
        from donkey_rl.sb3_latent_extractor import latent_policy_kwargs

        policy = "MlpPolicy"
        policy_kwargs = latent_policy_kwargs(ae_checkpoint=ae_ckpt, freeze=not bool(getattr(args, "train_encoder", False)))

    # Load existing model if specified, otherwise create new one
    load_model_path = str(getattr(args, "load_model", "") or "").strip()
    if load_model_path:
        print(f"Loading existing model from: {load_model_path}")
        model = PPO.load(load_model_path, env=train_env)
        # Update logger for continued training
        model.set_logger(sb3_logger)
        print(f"Loaded model with {model.num_timesteps} timesteps already trained.")
    else:
        model = PPO(
            policy=policy,
            env=train_env,
            verbose=1,
            seed=args.seed,
            policy_kwargs=policy_kwargs,
            n_steps=1024,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.0,
        )
        model.set_logger(sb3_logger)

    callbacks = [TrainingVizCallback(render=args.render).sb3_callback()]

    total_timesteps = int(args.total_timesteps)
    if getattr(args, "debug", False):
        total_timesteps = int(args.debug_steps)
        callbacks.insert(
            0,
            DebugObsDumpCallback(
                out_dir=args.debug_obs_dir,
                stop_after_steps=total_timesteps,
                save_every=int(args.debug_obs_every),
            ).sb3_callback(),
        )

    # Set up model saving paths
    # Training best model: saved based on training episode rewards
    best_train_model_path = os.path.join(args.run_dir, "best_train_model.zip")
    # Eval best model: saved based on evaluation mean reward
    best_eval_model_path = os.path.join(args.run_dir, "best_eval_model.zip")
    
    # Always save training best model (based on training episode rewards)
    callbacks.append(BestModelOnEpisodeRewardCallback(best_train_model_path).sb3_callback())
    
    # If evaluation is enabled, also save eval best model
    if args.eval:
        # Create eval_env_fn that will be called only when evaluation is needed
        # If eval_port is 0, use train port + 6 (original behavior for backward compatibility)
        eval_port = (args.port + 6) if args.eval_port == 0 else args.eval_port
        eval_env_fn = build_env_fn(
            env_id=args.env_id,
            host=args.eval_host,
            port=eval_port,
            exe_path=args.exe_path,
            fast_mode=args.fast,
            reward_type=args.reward_type,
            max_cte=args.max_cte,
            offtrack_step_penalty=args.offtrack_step_penalty,
            v2_w_speed=float(getattr(args, "v2_w_speed", 0.8)),
            v2_w_caution=float(getattr(args, "v2_w_caution", 0.6)),
            v2_min_speed=float(getattr(args, "v2_min_speed", 0.2)),
            v3_w_speed=float(getattr(args, "v3_w_speed", 1.2)),
            v3_min_speed=float(getattr(args, "v3_min_speed", 0.35)),
            v3_w_stall=float(getattr(args, "v3_w_stall", 2.0)),
            v3_alive_bonus=float(getattr(args, "v3_alive_bonus", 0.02)),
            v4_w_speed=float(getattr(args, "v4_w_speed", 1.0)),
            v4_min_speed=float(getattr(args, "v4_min_speed", 0.25)),
            v4_w_stall=float(getattr(args, "v4_w_stall", 3.0)),
            v4_w_smooth=float(getattr(args, "v4_w_smooth", 0.25)),
            v4_alive_bonus=float(getattr(args, "v4_alive_bonus", 0.03)),
            sim_io_timeout_s=float(getattr(args, "sim_io_timeout_s", 30.0)),
            obs_width=args.obs_width,
            obs_height=args.obs_height,
            # Keep evaluation deterministic by default.
            domain_rand=False,
            perspective_transform=bool(getattr(args, "perspective_transform", False)),
            obs_mode=str(getattr(args, "obs_mode", "auto")),
            aug_brightness=float(getattr(args, "aug_brightness", 0.4)),
            aug_contrast=float(getattr(args, "aug_contrast", 0.4)),
            aug_noise_std=float(getattr(args, "aug_noise_std", 0.05)),
            aug_color_jitter=float(getattr(args, "aug_color_jitter", 0.35)),
            random_friction=False,
            friction_min=1.0,
            friction_max=1.0,
            stall_detection=bool(getattr(args, "stall_detection", True)),
            stall_speed_threshold=float(getattr(args, "stall_speed_threshold", 0.1)),
            stall_max_steps=int(getattr(args, "stall_max_steps", 50)),
            stall_penalty=float(getattr(args, "stall_penalty", 20.0)),
            car_name=str(getattr(args, "run_id", "JetRacerAgent")) + "_eval",
        )
        
        # Create callback with lazy env initialization
        # The eval_env will only be created when evaluation is triggered
        # Save eval best model to a separate file
        eval_callback = EphemeralEvalCallback(
            eval_env_fn=eval_env_fn,
            best_model_save_path=best_eval_model_path,
            log_path=os.path.join(args.log_dir, "eval"),
            eval_freq=int(args.eval_freq),
            eval_freq_type=str(getattr(args, "eval_freq_type", "timesteps")),
            n_eval_episodes=int(args.n_eval_episodes),
            deterministic=True,
            render=False,
            max_episode_steps=getattr(args, "max_eval_episode_steps", 5000),
        )
        callbacks.append(eval_callback.sb3_callback())

    cb = CallbackList(callbacks)

    try:
        model.learn(total_timesteps=total_timesteps, callback=cb)
    except KeyboardInterrupt:
        print("Interrupted. Saving last model...")
        model.save(args.save_path)

        # Note: eval_callback manages its own eval_env lifecycle, so we don't need to
        # manually evaluate here. The callback will have already run evaluations during training.
        raise
    except (TimeoutError, ConnectionError, BrokenPipeError, OSError) as e:
        # When Unity disconnects or IO hangs/aborts, fail fast and save progress.
        print(f"Environment connection error: {type(e).__name__}: {e}")
        print("Saving last model and exiting...")
        try:
            model.save(args.save_path)
        except Exception:
            pass

        # Note: eval_callback manages its own eval_env lifecycle, so we don't need to
        # manually evaluate here. The callback will have already run evaluations during training.
        raise SystemExit(2)
    finally:
        # eval_env is managed by eval_callback and will be closed automatically
        # when the callback is garbage collected. No need to close it here.
        train_env.close()

    model.save(args.save_path)
    print(f"Saved last model to: {args.save_path}")
    if os.path.exists(best_train_model_path):
        print(f"Saved best training model to: {best_train_model_path}")
    if args.eval and os.path.exists(best_eval_model_path):
        print(f"Saved best eval model to: {best_eval_model_path}")


if __name__ == "__main__":
    main()

