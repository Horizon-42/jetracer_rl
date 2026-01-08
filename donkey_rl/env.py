from __future__ import annotations

from typing import Callable, Dict

import gymnasium as gym

from donkey_rl.compat import patch_old_gym_render_mode
from donkey_rl.obs_preprocess import ObsPreprocess
from donkey_rl.rewards import (
    DonkeyTrackLimitRewardConfig,
    JetRacerRaceRewardWrapper,
    JetRacerRaceRewardWrapperTrackLimit,
    RaceRewardConfig,
)
from donkey_rl.wrappers import JetRacerWrapper


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
        "cam_config":{
            "img_w": 320,
            "img_h": 320,
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
    preprocess_distort: bool,
    preprocess_distort_k1: float,
    preprocess_distort_k2: float,
    preprocess_color_distort: bool,
    preprocess_red_edge_strength: float,
    preprocess_red_edge_power: float,
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
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")

        env = ObsPreprocess(
            env,
            width=obs_width,
            height=obs_height,
            enable_distortion=preprocess_distort,
            distortion_k1=preprocess_distort_k1,
            distortion_k2=preprocess_distort_k2,
            enable_color_distortion=preprocess_color_distort,
            red_edge_strength=preprocess_red_edge_strength,
            red_edge_power=preprocess_red_edge_power,
        )
        return env

    return _thunk
