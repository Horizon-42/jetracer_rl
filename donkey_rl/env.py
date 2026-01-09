from __future__ import annotations

from typing import Callable, Dict

import gymnasium as gym

from donkey_rl.compat import patch_gym_donkeycar_stop_join, patch_old_gym_render_mode
from donkey_rl.obs_preprocess import ObsPreprocess
from donkey_rl.rewards import (
    CenterlineV3RewardConfig,
    CenterlineV2RewardConfig,
    CenterlineV4RewardConfig,
    DeepRacerStyleRewardConfig,
    DonkeyTrackLimitRewardConfig,
    JetRacerCenterlineV3RewardWrapper,
    JetRacerCenterlineV2RewardWrapper,
    JetRacerCenterlineV4RewardWrapper,
    JetRacerDeepRacerRewardWrapper,
    JetRacerRaceRewardWrapper,
    JetRacerRaceRewardWrapperTrackLimit,
    RaceRewardConfig,
)
from donkey_rl.wrappers import JetRacerWrapper, RandomFrictionWrapper
from donkey_rl.wrappers import StepTimeoutWrapper


def make_donkey_env(
    *,
    env_id: str,
    host: str,
    port: int,
    exe_path: str,
    fast_mode: bool,
    max_cte: float,
    car_name: str = "JetRacerAgent",
    io_timeout_s: float = 30.0,
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
        "car_name": str(car_name),
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

    env = gym.make(
        "GymV21Environment-v0",
        env_id=env_id,
        make_kwargs={"conf": conf},
    )

    # Protect training against silent hangs when the sim disconnects.
    # If step/reset blocks longer than io_timeout_s, we raise TimeoutError.
    if float(io_timeout_s) > 0:
        env = StepTimeoutWrapper(env, timeout_s=float(io_timeout_s))

    return env


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
    v2_w_speed: float,
    v2_w_caution: float,
    v2_min_speed: float,
    v3_w_speed: float,
    v3_min_speed: float,
    v3_w_stall: float,
    v3_alive_bonus: float,
    v4_w_speed: float,
    v4_min_speed: float,
    v4_w_stall: float,
    v4_w_smooth: float,
    v4_alive_bonus: float,
    sim_io_timeout_s: float,
    obs_width: int,
    obs_height: int,
    domain_rand: bool,
    aug_brightness: float,
    aug_contrast: float,
    aug_noise_std: float,
    aug_color_jitter: float,
    random_friction: bool,
    friction_min: float,
    friction_max: float,
    car_name: str = "JetRacerAgent",
) -> Callable[[], gym.Env]:
    def _thunk() -> gym.Env:
        env = make_donkey_env(
            env_id=env_id,
            host=host,
            port=port,
            exe_path=exe_path,
            fast_mode=fast_mode,
            max_cte=max_cte,
            car_name=str(car_name),
            io_timeout_s=float(sim_io_timeout_s),
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
        elif reward_type == "centerline_v2":
            env = JetRacerCenterlineV2RewardWrapper(
                env,
                cfg=CenterlineV2RewardConfig(
                    max_cte=max_cte,
                    w_speed=float(v2_w_speed),
                    w_caution=float(v2_w_caution),
                    min_speed=float(v2_min_speed),
                ),
            )
        elif reward_type == "centerline_v3":
            env = JetRacerCenterlineV3RewardWrapper(
                env,
                cfg=CenterlineV3RewardConfig(
                    max_cte=max_cte,
                    w_speed=float(v3_w_speed),
                    min_speed=float(v3_min_speed),
                    w_stall=float(v3_w_stall),
                    alive_bonus=float(v3_alive_bonus),
                ),
            )
        elif reward_type == "centerline_v4":
            env = JetRacerCenterlineV4RewardWrapper(
                env,
                cfg=CenterlineV4RewardConfig(
                    max_cte=max_cte,
                    w_speed=float(v4_w_speed),
                    min_speed=float(v4_min_speed),
                    w_stall=float(v4_w_stall),
                    w_smooth=float(v4_w_smooth),
                    alive_bonus=float(v4_alive_bonus),
                ),
            )
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
            aug_color_jitter=aug_color_jitter,
        )
        return env

    return _thunk
