"""DonkeyCar environment creation and configuration.

This module provides functions to create and configure DonkeyCar environments
through Gymnasium+Shimmy, with support for various reward types and wrappers.
"""

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
from donkey_rl.wrappers import JetRacerWrapper, RandomFrictionWrapper, StallDetectionWrapper, StepTimeoutWrapper


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
    """Create a DonkeyCar environment through Gymnasium+Shimmy.

    This function creates a basic DonkeyCar environment without reward shaping
    or observation preprocessing. Use `build_env_fn()` for a complete configured
    environment.

    Args:
        env_id: DonkeyCar environment ID (e.g., "donkey-generated-roads-v0").
        host: Host address for the simulator connection.
        port: Port number for the simulator connection.
        exe_path: Path to the simulator executable, or "remote" for remote connection.
        fast_mode: If True, use lower resolution and frame skipping for faster training.
        max_cte: Maximum cross-track error before episode termination.
        car_name: Name identifier for the car in the simulator.
        io_timeout_s: Timeout in seconds for I/O operations (0 to disable).

    Returns:
        A configured Gymnasium environment with StepTimeoutWrapper if timeout > 0.

    Example:
        >>> env = make_donkey_env(
        ...     env_id="donkey-generated-roads-v0",
        ...     host="127.0.0.1",
        ...     port=9091,
        ...     exe_path="remote",
        ...     fast_mode=False,
        ...     max_cte=8.0,
        ... )
    """
    # Apply compatibility patches for gym/gymnasium/shimmy interop
    patch_old_gym_render_mode()
    patch_gym_donkeycar_stop_join()

    # Ensure old-gym envs are registered
    import gym_donkeycar  # noqa: F401
    import shimmy  # noqa: F401

    # Build configuration dictionary for the simulator
    conf: Dict = {
        "exe_path": exe_path,
        "host": host,
        "port": int(port),
        "max_cte": float(max_cte),
        "body_style": "donkey",
        "body_rgb": (255, 165, 0),  # Orange car color
        "car_name": str(car_name),
        "font_size": 50,
        # Standard camera resolution
        "cam_resolution": (240, 320, 3),
        "cam_config": {
            "img_w": 320,
            "img_h": 240,
            "img_d": 3,
            "img_enc": "JPG",
        },
    }

    # Apply fast mode settings if enabled
    if fast_mode:
        conf["frame_skip"] = 2
        conf["cam_resolution"] = (60, 80, 3)
        conf["cam_config"] = {
            "img_w": 80,
            "img_h": 60,
            "img_d": 3,
            "img_enc": "JPG",
        }

    # Create environment through Shimmy (Gymnasium compatibility layer)
    env = gym.make(
        "GymV21Environment-v0",
        env_id=env_id,
        make_kwargs={"conf": conf},
    )

    # Protect training against silent hangs when the sim disconnects
    # If step/reset blocks longer than io_timeout_s, we raise TimeoutError
    if float(io_timeout_s) > 0:
        env = StepTimeoutWrapper(env, timeout_s=float(io_timeout_s))

    return env


def _wrap_with_reward(
    env: gym.Env,
    *,
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
) -> gym.Env:
    """Wrap environment with the specified reward function.

    This function applies the appropriate reward wrapper based on reward_type.
    All reward wrappers compute shaped rewards based on environment state.

    Args:
        env: The base environment to wrap.
        reward_type: Type of reward function to use. Options:
            - "base": Basic race reward (progress, speed, centerline)
            - "track_limit": Race reward with explicit off-track penalties
            - "deepracer": AWS DeepRacer-style reward with centerline bands
            - "centerline_v2": Centerline + speed + smoothness + caution
            - "centerline_v3": Centerline + speed + anti-stall (simplified)
            - "centerline_v4": Centerline + speed + smoothness + anti-stall
        max_cte: Maximum cross-track error threshold.
        offtrack_step_penalty: Per-step penalty when off-track (for track_limit).
        v2_w_speed: Speed weight for centerline_v2.
        v2_w_caution: Caution weight for centerline_v2.
        v2_min_speed: Minimum speed threshold for centerline_v2.
        v3_w_speed: Speed weight for centerline_v3.
        v3_min_speed: Minimum speed threshold for centerline_v3.
        v3_w_stall: Stall penalty weight for centerline_v3.
        v3_alive_bonus: Per-step alive bonus for centerline_v3.
        v4_w_speed: Speed weight for centerline_v4.
        v4_min_speed: Minimum speed threshold for centerline_v4.
        v4_w_stall: Stall penalty weight for centerline_v4.
        v4_w_smooth: Smoothness weight for centerline_v4.
        v4_alive_bonus: Per-step alive bonus for centerline_v4.

    Returns:
        The environment wrapped with the appropriate reward function.

    Raises:
        ValueError: If reward_type is not recognized.
    """
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
        env = JetRacerDeepRacerRewardWrapper(
            env, cfg=DeepRacerStyleRewardConfig(max_cte=max_cte)
        )

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
        raise ValueError(f"Unknown reward_type: {reward_type}. "
                         f"Supported types: base, track_limit, deepracer, "
                         f"centerline_v2, centerline_v3, centerline_v4")

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
    perspective_transform: bool,
    obs_mode: str = "auto",
    aug_brightness: float,
    aug_contrast: float,
    aug_noise_std: float,
    aug_color_jitter: float,
    random_friction: bool,
    friction_min: float,
    friction_max: float,
    stall_detection: bool = True,
    stall_speed_threshold: float = 0.1,
    stall_max_steps: int = 50,
    stall_penalty: float = 20.0,
    car_name: str = "JetRacerAgent",
) -> Callable[[], gym.Env]:
    """Build a factory function that creates fully configured DonkeyCar environments.

    This is the main entry point for creating training environments. It returns
    a function (thunk) that, when called, creates a new environment instance with
    all wrappers applied in the correct order:

    1. Base DonkeyCar environment (with timeout protection)
    2. Random friction wrapper (if enabled)
    3. JetRacer action wrapper (converts [throttle, steering] to [steer, throttle])
    4. Reward wrapper (based on reward_type)
    5. Observation preprocessing wrapper (resize, domain randomization, etc.)

    The returned function is suitable for use with Stable-Baselines3's VecEnv
    implementations, which require environment factory functions.

    Args:
        env_id: DonkeyCar environment ID (e.g., "donkey-generated-roads-v0").
        host: Host address for simulator connection.
        port: Port number for simulator connection.
        exe_path: Path to simulator executable, or "remote" for remote connection.
        fast_mode: If True, use lower resolution and frame skipping.
        reward_type: Type of reward function (see _wrap_with_reward).
        max_cte: Maximum cross-track error threshold.
        offtrack_step_penalty: Per-step penalty when off-track (for track_limit).
        v2_w_speed: Speed weight for centerline_v2.
        v2_w_caution: Caution weight for centerline_v2.
        v2_min_speed: Minimum speed threshold for centerline_v2.
        v3_w_speed: Speed weight for centerline_v3.
        v3_min_speed: Minimum speed threshold for centerline_v3.
        v3_w_stall: Stall penalty weight for centerline_v3.
        v3_alive_bonus: Per-step alive bonus for centerline_v3.
        v4_w_speed: Speed weight for centerline_v4.
        v4_min_speed: Minimum speed threshold for centerline_v4.
        v4_w_stall: Stall penalty weight for centerline_v4.
        v4_w_smooth: Smoothness weight for centerline_v4.
        v4_alive_bonus: Per-step alive bonus for centerline_v4.
        sim_io_timeout_s: I/O timeout in seconds (0 to disable).
        obs_width: Output observation width in pixels.
        obs_height: Output observation height in pixels.
        domain_rand: If True, apply domain randomization (brightness, contrast, noise, color jitter).
        perspective_transform: If True, apply perspective transformation to camera view (deprecated, use obs_mode).
        obs_mode: Observation mode ('auto', 'raw', 'perspective', 'mix'). 'mix' stacks raw+perspective vertically.
        aug_brightness: Brightness augmentation range (applied if domain_rand=True).
        aug_contrast: Contrast augmentation range (applied if domain_rand=True).
        aug_noise_std: Gaussian noise standard deviation (applied if domain_rand=True).
        aug_color_jitter: Color jitter strength in HSV space (applied if domain_rand=True).
        random_friction: If True, randomize friction per episode.
        friction_min: Minimum friction scale factor (applied if random_friction=True).
        friction_max: Maximum friction scale factor (applied if random_friction=True).
        stall_detection: If True, terminate episode when car is stuck (default: True).
        stall_speed_threshold: Speed below this is considered stalled (default: 0.1).
        stall_max_steps: Terminate after this many consecutive stalled steps (default: 50).
        stall_penalty: Penalty when episode terminates due to stall (default: 20.0).
        car_name: Name identifier for the car in the simulator.

    Returns:
        A callable that returns a new environment instance when called.

    Example:
        >>> env_fn = build_env_fn(
        ...     env_id="donkey-generated-roads-v0",
        ...     host="127.0.0.1",
        ...     port=9091,
        ...     exe_path="remote",
        ...     fast_mode=False,
        ...     reward_type="centerline_v3",
        ...     max_cte=8.0,
        ...     # ... other parameters ...
        ... )
        >>> env = env_fn()  # Create a new environment instance
    """
    def _thunk() -> gym.Env:
        # Step 1: Create base DonkeyCar environment with timeout protection
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

        # Step 2: Apply friction randomization (if enabled)
        # This should be applied before action wrapper to modify Donkey action space
        if random_friction:
            env = RandomFrictionWrapper(env, min_scale=friction_min, max_scale=friction_max)

        # Step 3: Apply JetRacer action wrapper
        # Converts [throttle, steering] to DonkeyCar's [steer, throttle]
        env = JetRacerWrapper(env, steer_scale=1.0, throttle_scale=1.0)

        # Step 4: Apply reward wrapper based on reward_type
        env = _wrap_with_reward(
            env,
            reward_type=reward_type,
            max_cte=max_cte,
            offtrack_step_penalty=offtrack_step_penalty,
            v2_w_speed=v2_w_speed,
            v2_w_caution=v2_w_caution,
            v2_min_speed=v2_min_speed,
            v3_w_speed=v3_w_speed,
            v3_min_speed=v3_min_speed,
            v3_w_stall=v3_w_stall,
            v3_alive_bonus=v3_alive_bonus,
            v4_w_speed=v4_w_speed,
            v4_min_speed=v4_min_speed,
            v4_w_stall=v4_w_stall,
            v4_w_smooth=v4_w_smooth,
            v4_alive_bonus=v4_alive_bonus,
        )

        # Step 4.5: Apply stall detection wrapper (if enabled)
        # This terminates episodes early if the car gets stuck
        if stall_detection:
            env = StallDetectionWrapper(
                env,
                speed_threshold=stall_speed_threshold,
                max_stall_steps=stall_max_steps,
                stall_penalty=stall_penalty,
            )
            print(f"Stall detection enabled: threshold={stall_speed_threshold}, "
                  f"max_steps={stall_max_steps}, penalty={stall_penalty}")

        # Step 5: Apply observation preprocessing wrapper
        # This handles resizing, perspective transform, and domain randomization
        print(f"Wrapping env with ObsPreprocess: {obs_width}x{obs_height}, "
              f"domain_rand={domain_rand}, obs_mode={obs_mode}")

        env = ObsPreprocess(
            env,
            width=obs_width,
            height=obs_height,
            domain_rand=domain_rand,
            perspective_transform=perspective_transform,
            obs_mode=obs_mode,
            aug_brightness=aug_brightness,
            aug_contrast=aug_contrast,
            aug_noise_std=aug_noise_std,
            aug_color_jitter=aug_color_jitter,
        )

        return env

    return _thunk
