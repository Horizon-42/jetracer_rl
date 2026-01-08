from __future__ import annotations

from typing import Callable, Dict

import gymnasium as gym

from donkey_rl.compat import patch_gym_donkeycar_stop_join, patch_old_gym_render_mode
from donkey_rl.obs_preprocess import ObsPreprocess
from donkey_rl.rewards import (
    DeepRacerStyleRewardConfig,
    DonkeyTrackLimitRewardConfig,
    JetRacerDeepRacerRewardWrapper,
    JetRacerRaceRewardWrapper,
    JetRacerRaceRewardWrapperTrackLimit,
    RaceRewardConfig,
)
from donkey_rl.wrappers import JetRacerWrapper, RandomFrictionWrapper


def make_donkey_env(
    *,
    env_id: str,
    host: str,
    port: int,
    exe_path: str,
    fast_mode: bool,
    max_cte: float,
) -> gym.Env:
    """Create a DonkeyCar env through Gymnasium+Shimmy."""

    patch_old_gym_render_mode()
    patch_gym_donkeycar_stop_join()

    # Ensure old-gym envs are registered.
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
        # Keep env observation_space consistent with the images requested from the sim.
        "cam_resolution": (240, 320, 3),
        "cam_config":{
            "img_w": 320,
            "img_h": 240,
            "img_d": 3,
            "img_enc": "JPG",
        }
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

    return gym.make(
        "GymV21Environment-v0",
        env_id=env_id,
        make_kwargs={"conf": conf},
    )


def build_env_fn(
    *,
    env_id: str,
    host: str,
    port: int,
    exe_path: str,
    fast_mode: bool,
    reward_type: str,
    max_cte: float,
    offtrack_step_penalty: float,
    obs_width: int,
    obs_height: int,
    domain_rand: bool,
    aug_brightness: float,
    aug_contrast: float,
    aug_noise_std: float,
    random_friction: bool,
    friction_min: float,
    friction_max: float,
) -> Callable[[], gym.Env]:
    def _thunk() -> gym.Env:
        env = make_donkey_env(
            env_id=env_id,
            host=host,
            port=port,
            exe_path=exe_path,
            fast_mode=fast_mode,
            max_cte=max_cte,
        )

        # Apply friction randomization in Donkey action space: [steer, throttle]
        if random_friction:
            env = RandomFrictionWrapper(env, min_scale=friction_min, max_scale=friction_max)

        env = JetRacerWrapper(env, steer_scale=1.0, throttle_scale=1.0)

        if reward_type == "base":
            env = JetRacerRaceRewardWrapper(env, cfg=RaceRewardConfig())
        elif reward_type == "track_limit":
            env = JetRacerRaceRewardWrapperTrackLimit(
                env,
                base_cfg=RaceRewardConfig(),
                track_cfg=DonkeyTrackLimitRewardConfig(
                    max_cte=max_cte,
                    offtrack_step_penalty=offtrack_step_penalty,
                ),
            )
        elif reward_type == "deepracer":
            env = JetRacerDeepRacerRewardWrapper(env, cfg=DeepRacerStyleRewardConfig(max_cte=max_cte))
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")

        env = ObsPreprocess(
            env,
            width=obs_width,
            height=obs_height,
            domain_rand=domain_rand,
            aug_brightness=aug_brightness,
            aug_contrast=aug_contrast,
            aug_noise_std=aug_noise_std,
        )
        return env

    return _thunk
