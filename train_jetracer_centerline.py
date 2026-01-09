"""Train a DonkeyCar (Unity sim) policy with a JetRacer-style interface.

This is the entrypoint script. Most logic lives in small modules under
the `donkey_rl/` package for easier learning and editing.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

from donkey_rl.args import parse_args
from donkey_rl.callbacks import BestModelOnEpisodeRewardCallback, DebugObsDumpCallback, TrainingVizCallback
from donkey_rl.env import build_env_fn


def _sync_best_model(best_zip_path: str, target_best_path: str) -> None:
    if not os.path.exists(best_zip_path):
        return
    os.makedirs(os.path.dirname(target_best_path) or ".", exist_ok=True)
    shutil.copy2(best_zip_path, target_best_path)


def main() -> None:
    args = parse_args()

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
                obs_width=args.obs_width,
                obs_height=args.obs_height,
                domain_rand=bool(getattr(args, "domain_rand", False)),
                aug_brightness=float(getattr(args, "aug_brightness", 0.25)),
                aug_contrast=float(getattr(args, "aug_contrast", 0.25)),
                aug_noise_std=float(getattr(args, "aug_noise_std", 0.02)),
                aug_color_jitter=float(getattr(args, "aug_color_jitter", 0.2)),
                random_friction=bool(getattr(args, "random_friction", False)),
                friction_min=float(getattr(args, "friction_min", 0.6)),
                friction_max=float(getattr(args, "friction_max", 1.0)),
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

    best_zip_in_log = os.path.join(args.log_dir, "best", "best_model.zip")
    eval_callback = None
    eval_env = None

    if args.eval:
        eval_env = DummyVecEnv(
            [
                build_env_fn(
                    env_id=args.env_id,
                    host=args.eval_host,
                    port=args.eval_port,
                    exe_path=args.eval_exe_path,
                    fast_mode=args.fast,
                    reward_type=args.reward_type,
                    max_cte=args.max_cte,
                    offtrack_step_penalty=args.offtrack_step_penalty,
                    v2_w_speed=float(getattr(args, "v2_w_speed", 0.8)),
                    v2_w_caution=float(getattr(args, "v2_w_caution", 0.6)),
                    v2_min_speed=float(getattr(args, "v2_min_speed", 0.2)),
                    obs_width=args.obs_width,
                    obs_height=args.obs_height,
                    # Keep evaluation deterministic by default.
                    domain_rand=False,
                    aug_brightness=float(getattr(args, "aug_brightness", 0.25)),
                    aug_contrast=float(getattr(args, "aug_contrast", 0.25)),
                    aug_noise_std=float(getattr(args, "aug_noise_std", 0.02)),
                    aug_color_jitter=float(getattr(args, "aug_color_jitter", 0.2)),
                    random_friction=False,
                    friction_min=1.0,
                    friction_max=1.0,
                )
            ]
        )
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
        callbacks.append(BestModelOnEpisodeRewardCallback(args.best_model_path).sb3_callback())

    cb = CallbackList(callbacks)

    try:
        model.learn(total_timesteps=total_timesteps, callback=cb)
    except KeyboardInterrupt:
        print("Interrupted. Saving last model...")
        model.save(args.save_path)

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

        _sync_best_model(best_zip_in_log, args.best_model_path)
        raise
    finally:
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

