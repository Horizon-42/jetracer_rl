"""Reward functions (wrappers) for DonkeyCar training.

Big picture
-----------
Stable-Baselines3 (SB3) learns by repeatedly calling:
    obs, reward, terminated, truncated, info = env.step(action)

- `reward` is the number your agent tries to maximize.
- `info` is an *optional* dictionary of extra signals.

About `info` in DonkeyCar
-------------------------
In gym-donkeycar, `info` commonly includes keys like:

- `cte`: **Cross-Track Error** (distance from track centerline)
- `speed`: forward speed
- `hit`: collision signal (string, often 'none' or an object name)

These are typically "simulator ground truth".

Simulator vs real robot (VERY important)
----------------------------------------
On a real car you usually do NOT get `cte/speed/hit` for free.

To deploy in the real world, you need estimators:
- `cte`: estimate from lane detection / localization
- `speed`: estimate from wheel encoders / IMU / vision
- `hit`: detect via bumper switch / IMU shock / motor current / vision heuristics

If your reward depends heavily on simulator-only signals, sim-to-real transfer
becomes harder unless you implement equivalent measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


def _try_get_last_raw_action(env: gym.Env) -> Tuple[float, float]:
    """Best-effort extraction of last JetRacer action.

    Avoids importing wrappers here (keeps modules decoupled).

    Why do we want this?
    --------------------
    Our action wrapper (`JetRacerWrapper` in `donkey_rl/wrappers.py`) converts a
    JetRacer-style action:
        [throttle, steering]
    into DonkeyCar's expected action:
        [steer, throttle]

    The reward sometimes wants to penalize steering magnitude / steering changes.
    The clean way is: read the action we *actually applied*.

    SB3 does not pass the action into reward functions directly (it only passes it
    into `env.step(action)`). But because we control the wrapper, we store the last
    raw action on the env object so the reward wrapper can read it.

    If the attribute is missing, we return (0.0, 0.0) and the steering penalties
    become 0 (safe default).

    Args:
        env: The gym environment, potentially wrapped with JetRacerWrapper.

    Returns:
        A tuple of (throttle, steering) from the last raw action applied,
        or (0.0, 0.0) if not available.
    """
    raw = getattr(env, "last_raw_action", None)
    if raw is None:
        return 0.0, 0.0
    try:
        return float(raw[0]), float(raw[1])
    except Exception:
        return 0.0, 0.0


def _parse_step_result(result: Tuple) -> Tuple:
    """Parse step result to handle both Gymnasium (5-tuple) and old Gym (4-tuple) APIs.

    Gymnasium API returns: (obs, reward, terminated, truncated, info)
    Old Gym API returns: (obs, reward, done, info)

    Args:
        result: The result tuple from env.step(action).

    Returns:
        A tuple of (obs, base_reward, terminated, truncated, info, done).
        For old Gym API, done=True when terminated or truncated, otherwise done=False.
    """
    if len(result) == 5:
        obs, base_reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
        return obs, base_reward, bool(terminated), bool(truncated), info, done
    else:
        obs, base_reward, done, info = result
        return obs, base_reward, bool(done), False, info, bool(done)


def _extract_env_info(info: Dict) -> Tuple[Optional[float], float, Optional[str]]:
    """Extract common environment information from info dict.

    Args:
        info: The info dictionary from env.step().

    Returns:
        A tuple of (cte, speed, hit):
        - cte: Cross-track error (distance from centerline), None if not available.
        - speed: Forward speed, 0.0 if not available.
        - hit: Collision signal (string), None if not available.
    """
    info_dict = dict(info) if isinstance(info, dict) else {}
    cte = info_dict.get("cte")
    speed = float(info_dict.get("speed", 0.0) or 0.0)
    hit = info_dict.get("hit")
    return cte, speed, hit


class BaseRewardWrapper(gym.Wrapper):
    """Base class for reward wrappers that handles common step() logic.

    This base class provides:
    - Compatibility handling for both Gymnasium and old Gym APIs
    - Common info dictionary parsing
    - Action extraction from wrapped environment
    - Standardized return format

    Subclasses should override `_compute_reward()` to implement specific reward logic.
    """

    def __init__(self, env: gym.Env, reward_info_key: str):
        """Initialize the base reward wrapper.

        Args:
            env: The environment to wrap.
            reward_info_key: Key name to store reward details in info dict.
        """
        super().__init__(env)
        self._reward_info_key = reward_info_key

    def step(self, action):
        """Execute one environment step and compute shaped reward.

        This method handles:
        1. Calling the underlying env.step()
        2. Parsing results (Gymnasium vs old Gym compatibility)
        3. Extracting environment signals (cte, speed, hit)
        4. Computing the shaped reward via _compute_reward()
        5. Adding reward details to info dict
        6. Returning standardized format

        Args:
            action: Action to take in the environment.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info) for Gymnasium,
            or (obs, reward, done, info) for old Gym.
        """
        result = self.env.step(action)
        obs, base_reward, terminated, truncated, info, done = _parse_step_result(result)

        # Extract common environment signals
        cte, speed, hit = _extract_env_info(info)
        throttle, steering = _try_get_last_raw_action(self.env)

        # Compute shaped reward (to be implemented by subclasses)
        reward, reward_info = self._compute_reward(
            cte=cte,
            speed=speed,
            hit=hit,
            throttle=throttle,
            steering=steering,
            done=done,
        )

        # Add reward details to info dict
        info_dict = dict(info) if isinstance(info, dict) else {}
        info_dict[self._reward_info_key] = reward_info

        # Return in appropriate format
        if len(result) == 5:
            return obs, float(reward), terminated, truncated, info_dict
        return obs, float(reward), done, info_dict

    def _compute_reward(
        self,
        cte: Optional[float],
        speed: float,
        hit: Optional[str],
        throttle: float,
        steering: float,
        done: bool,
    ) -> Tuple[float, Dict]:
        """Compute the shaped reward based on environment state.

        This method must be implemented by subclasses.

        Args:
            cte: Cross-track error (distance from centerline), None if unavailable.
            speed: Forward speed.
            hit: Collision signal (string), None if no collision.
            throttle: Last throttle action (from raw action).
            steering: Last steering action (from raw action).
            done: Whether the episode is done.

        Returns:
            A tuple of (reward, reward_info_dict):
            - reward: The computed reward value.
            - reward_info_dict: Dictionary with detailed reward breakdown for logging.
        """
        raise NotImplementedError("Subclasses must implement _compute_reward()")


# ============================================================================
# Reward Configuration Dataclasses
# ============================================================================


@dataclass(frozen=True)
class RaceRewardConfig:
    """Reward shaping weights for a "race" objective.

    Each field is a *multiplier* for one part of the shaped reward.
    Higher weight => that behavior matters more.

    Tip for beginners:
    - Change only ONE weight at a time.
    - Keep reward magnitudes roughly within a sane range (e.g. -100..+100 per step)
      so PPO training remains numerically stable.
    """

    # Reward for moving forward along the lane direction.
    # In DonkeyCar we approximate lane direction with dot_dir=1.
    w_progress: float = 2.0

    # Extra reward proportional to speed.
    # WARNING: if this is too big, the agent may learn to go fast even while off-track.
    w_speed: float = 0.5

    # Penalty for being away from the center line.
    # In this DonkeyCar-only code, we use |cte| for this.
    w_center: float = 2.0

    # Penalty for heading error (how misaligned you are with the lane direction).
    # DonkeyCar info usually does not give this, so we use 0.0.
    w_heading: float = 0.3

    # Penalty for large steering magnitude (discourages zig-zag).
    w_steer: float = 0.10

    # Penalty for steering changes (smoothness term).
    w_steer_rate: float = 0.05

    # One-time penalty when the episode terminates due to off-road/collision.
    offroad_penalty: float = 50.0


@dataclass(frozen=True)
class DonkeyTrackLimitRewardConfig:
    """Extra reward terms for enforcing 'stay on track' in DonkeyCar.

    This is used by the `track_limit` reward type.

    Key signal: `cte`
    -----------------
    `cte` stands for **Cross-Track Error**.

    Intuition:
    - Imagine the ideal "centerline" of the track.
    - `cte` is how far the car is from that centerline.

    Typical properties (can vary by environment):
    - Sign (+/-) indicates left vs right of center.
    - Magnitude |cte| indicates how far away.
    - Units are often meters, but in some sims it can be an arbitrary scale.

    In gym-donkeycar, `cte` is provided by the simulator.
    On a real car, you must estimate this from camera or localization.
    """

    # Consider the car "off track" if |cte| > max_cte.
    max_cte: float = 8.0

    # Additional penalty applied EVERY step while off-track.
    # This is different from `offroad_penalty` which is usually applied when the episode ends.
    offtrack_step_penalty: float = 5.0


@dataclass(frozen=True)
class DeepRacerStyleRewardConfig:
    """AWS DeepRacer-style reward ported to DonkeyCar.

    DeepRacer uses:
    - track_width
    - distance_from_center
    - all_wheels_on_track
    - speed

    In DonkeyCar we typically have:
    - cte (cross-track error) ~ distance from centerline
    - speed (simulator-provided)

    We approximate:
    - distance_from_center = abs(cte)
    - all_wheels_on_track = abs(cte) <= max_cte
    - track_width = 2 * max_cte (proxy; adjust if your sim uses a different scale)
    """

    # Off-track threshold in cte units.
    max_cte: float = 8.0

    # Speed encouragement term: reward += speed * speed_scale
    speed_scale: float = 0.5

    # Reward values for centerline bands.
    r_center: float = 1.0
    r_mid: float = 0.5
    r_edge: float = 0.1
    r_offtrack: float = 1e-3

    # Marker fractions of track width (DeepRacer-style)
    m1_frac: float = 0.10
    m2_frac: float = 0.25
    m3_frac: float = 0.50


@dataclass(frozen=True)
class CenterlineV2RewardConfig:
    """Simple, tunable reward for centerline + speed + smooth + caution.

    Design goals
    ------------
    1) Follow the centerline: reward decreases as |cte| increases.
    2) Won't stop: reward includes speed and penalizes too-low speed.
    3) Smooth yet cautious: penalize steering magnitude & steering changes,
       and penalize high-speed while turning / off-center.
    4) Faster is better: positive speed term.

    Required signals (from DonkeySim `info`)
    --------------------------------------
    - cte: cross-track error (distance from center)
    - speed: forward speed

    Steering is taken from our action wrapper via `_try_get_last_raw_action`.
    """

    max_cte: float = 8.0

    # Core terms
    w_center: float = 1.0
    w_speed: float = 0.8

    # Anti-stall
    min_speed: float = 0.2
    w_stall: float = 1.0

    # Smoothness (single knob)
    w_smooth: float = 0.3

    # Caution: penalize high speed when steering or off-center
    w_caution: float = 0.6

    # Safety penalties
    offtrack_penalty: float = 50.0
    collision_penalty: float = 50.0

    # Penalize reversing (negative throttle)
    reverse_penalty: float = 2.0


@dataclass(frozen=True)
class CenterlineV3RewardConfig:
    """Simpler centerline+speed reward with stronger anti-stall bias.

    Compared to CenterlineV2:
    - Drops smoothness/caution terms (less parameters, less risk of learning to stop).
    - Keeps: centerline score, speed reward, anti-stall penalty, safety penalties.
    """

    max_cte: float = 8.0

    # Core terms
    w_center: float = 1.0
    w_speed: float = 1.2

    # Anti-stall
    min_speed: float = 0.35
    w_stall: float = 2.0

    # Small per-step bias to avoid the zero-action local optimum.
    alive_bonus: float = 0.02

    # Safety penalties
    offtrack_penalty: float = 50.0
    collision_penalty: float = 50.0

    # Penalize reversing (negative throttle)
    reverse_penalty: float = 2.0


@dataclass(frozen=True)
class CenterlineV4RewardConfig:
    """Centerline reward with speed, smoothness, alive bonus, and anti-stall.

    Design requirements
    -------------------
    1) follow centerline
    2) smooth turn
    3) faster better
    4) stay longer better
    5) can't just stop in the track

    Minimal knobs:
    - w_speed, min_speed, w_stall, w_smooth, alive_bonus
    """

    max_cte: float = 8.0

    w_center: float = 1.0
    w_speed: float = 1.0

    # Smooth turn (penalize steering magnitude + steering change)
    w_smooth: float = 0.25

    # Anti-stall
    min_speed: float = 0.25
    w_stall: float = 3.0

    # Stay longer better
    alive_bonus: float = 0.03

    # Safety penalties
    offtrack_penalty: float = 50.0
    collision_penalty: float = 50.0

    # Penalize reversing (negative throttle)
    reverse_penalty: float = 2.0


# ============================================================================
# Reward Wrapper Implementations
# ============================================================================


class JetRacerRaceRewardWrapper(BaseRewardWrapper):
    """Original reward wrapper (kept as-is in behavior).

    Rewards progress/speed, penalizes leaving the road, and discourages zig-zag steering.
    Adds info under `info["race_reward"]`.
    """

    def __init__(self, env: gym.Env, cfg: RaceRewardConfig = RaceRewardConfig()):
        """Initialize the race reward wrapper.

        Args:
            env: The environment to wrap.
            cfg: Reward configuration parameters.
        """
        super().__init__(env, reward_info_key="race_reward")
        self.cfg = cfg
        self._prev_steering: float = 0.0

    def _compute_reward(
        self,
        cte: Optional[float],
        speed: float,
        hit: Optional[str],
        throttle: float,
        steering: float,
        done: bool,
    ) -> Tuple[float, Dict]:
        """Compute race-style reward.

        Main idea:
        - Reward moving forward (proxy: speed)
        - Penalize being far from center (proxy: |cte|)
        - Penalize large steering and steering changes (smoothness)

        Args:
            cte: Cross-track error, None if unavailable.
            speed: Forward speed.
            hit: Collision signal.
            throttle: Last throttle action.
            steering: Last steering action.
            done: Whether episode is done.

        Returns:
            Tuple of (reward, info_dict).
        """
        shaped_reward = 0.0
        done_why = self._extract_done_code(hit)

        # Main reward shaping: progress, speed, centerline, heading
        if cte is not None:
            dist = float(cte)
            angle_rad = 0.0  # Heading error not available in DonkeyCar
            dot_dir = 1.0  # Assume forward direction

            # Reward progress (forward movement) and speed
            progress = max(0.0, dot_dir) * float(speed)
            shaped_reward += self.cfg.w_progress * progress
            shaped_reward += self.cfg.w_speed * float(speed)

            # Penalize distance from centerline and heading error
            shaped_reward -= self.cfg.w_center * abs(dist)
            shaped_reward -= self.cfg.w_heading * abs(angle_rad)
        else:
            # If we cannot get cte, provide a small penalty
            shaped_reward -= 1.0

        # Steering penalties help reduce "zig-zag" behavior
        shaped_reward -= self.cfg.w_steer * float(steering**2)
        steer_rate = float(steering) - float(self._prev_steering)
        shaped_reward -= self.cfg.w_steer_rate * float(steer_rate**2)
        self._prev_steering = float(steering)

        # Terminal penalty for off-road or collision
        if done and done_why in {"offroad-or-collision", "collision"}:
            shaped_reward -= self.cfg.offroad_penalty

        reward_info = {
            "reward": float(shaped_reward),
            "speed": float(speed),
            "throttle": float(throttle),
            "steering": float(steering),
            "done_why": done_why,
            "cte": None if cte is None else float(cte),
            "hit": hit,
        }

        return float(shaped_reward), reward_info

    @staticmethod
    def _extract_done_code(hit: Optional[str]) -> str:
        """Map DonkeyCar collision signal into a simple code string.

        gym-donkeycar often reports collisions via `info["hit"]`.
        Common patterns:
        - "none" => no collision
        - some other string => collided with something

        Args:
            hit: The hit/collision signal from info dict.

        Returns:
            A string code indicating the reason for episode termination.
        """
        if isinstance(hit, str) and hit != "none":
            return "collision"
        return "in-progress"


class JetRacerRaceRewardWrapperTrackLimit(BaseRewardWrapper):
    """Reward variant: original shaping + explicit off-track penalty.

    Extends the base race reward with additional off-track step penalties.
    Adds info under `info["race_reward_track"]`.
    """

    def __init__(
        self,
        env: gym.Env,
        base_cfg: RaceRewardConfig = RaceRewardConfig(),
        track_cfg: DonkeyTrackLimitRewardConfig = DonkeyTrackLimitRewardConfig(),
    ):
        """Initialize the track-limit reward wrapper.

        Args:
            env: The environment to wrap.
            base_cfg: Base race reward configuration.
            track_cfg: Track limit specific configuration.
        """
        super().__init__(env, reward_info_key="race_reward_track")
        self.base_cfg = base_cfg
        self.track_cfg = track_cfg
        self._prev_steering: float = 0.0

    def _compute_reward(
        self,
        cte: Optional[float],
        speed: float,
        hit: Optional[str],
        throttle: float,
        steering: float,
        done: bool,
    ) -> Tuple[float, Dict]:
        """Compute race reward with explicit off-track penalties.

        Uses the same base shaping as JetRacerRaceRewardWrapper, plus:
        - Step-by-step penalty when |cte| > max_cte
        - Terminal penalty when episode ends off-track

        Args:
            cte: Cross-track error, None if unavailable.
            speed: Forward speed.
            hit: Collision signal.
            throttle: Last throttle action.
            steering: Last steering action.
            done: Whether episode is done.

        Returns:
            Tuple of (reward, info_dict).
        """
        shaped_reward = 0.0

        # Base reward shaping (same as JetRacerRaceRewardWrapper)
        if cte is not None:
            dist = float(cte)
            angle_rad = 0.0
            dot_dir = 1.0

            progress = max(0.0, dot_dir) * float(speed)
            shaped_reward += self.base_cfg.w_progress * progress
            shaped_reward += self.base_cfg.w_speed * float(speed)
            shaped_reward -= self.base_cfg.w_center * abs(dist)
            shaped_reward -= self.base_cfg.w_heading * abs(angle_rad)
        else:
            shaped_reward -= 1.0

        # Steering penalties
        shaped_reward -= self.base_cfg.w_steer * float(steering**2)
        steer_rate = float(steering) - float(self._prev_steering)
        shaped_reward -= self.base_cfg.w_steer_rate * float(steer_rate**2)
        self._prev_steering = float(steering)

        # Explicit off-track classification and step penalty
        offtrack = False
        if cte is not None:
            offtrack = abs(float(cte)) > float(self.track_cfg.max_cte)
            if offtrack:
                shaped_reward -= float(self.track_cfg.offtrack_step_penalty)

        # Terminal penalty: if episode ended and we were off-track or hit something
        if done:
            collision = isinstance(hit, str) and hit != "none"
            if offtrack or collision:
                shaped_reward -= float(self.base_cfg.offroad_penalty)

        reward_info = {
            "reward": float(shaped_reward),
            "speed": float(speed),
            "throttle": float(throttle),
            "steering": float(steering),
            "cte": None if cte is None else float(cte),
            "max_cte": float(self.track_cfg.max_cte),
            "offtrack": bool(offtrack),
            "hit": hit,
        }

        return float(shaped_reward), reward_info


class JetRacerDeepRacerRewardWrapper(BaseRewardWrapper):
    """DeepRacer-style reward wrapper.

    Implements AWS DeepRacer's reward structure with centerline bands.
    Adds info under `info["deepracer_reward"]`.
    """

    def __init__(self, env: gym.Env, cfg: DeepRacerStyleRewardConfig = DeepRacerStyleRewardConfig()):
        """Initialize the DeepRacer reward wrapper.

        Args:
            env: The environment to wrap.
            cfg: DeepRacer reward configuration.
        """
        super().__init__(env, reward_info_key="deepracer_reward")
        self.cfg = cfg

    def _compute_reward(
        self,
        cte: Optional[float],
        speed: float,
        hit: Optional[str],
        throttle: float,
        steering: float,
        done: bool,
    ) -> Tuple[float, Dict]:
        """Compute DeepRacer-style reward.

        DeepRacer uses discrete reward bands based on distance from centerline:
        - Center band (closest): highest reward
        - Mid band: medium reward
        - Edge band: low reward
        - Off-track: minimal reward

        Args:
            cte: Cross-track error, None if unavailable.
            speed: Forward speed.
            hit: Collision signal (not used in DeepRacer logic).
            throttle: Last throttle action (not used).
            steering: Last steering action (not used).
            done: Whether episode is done (not used).

        Returns:
            Tuple of (reward, info_dict).
        """
        # Map to DeepRacer parameters
        max_cte = float(self.cfg.max_cte)
        track_width = 2.0 * max_cte
        distance_from_center = abs(float(cte)) if cte is not None else float("inf")
        all_wheels_on_track = bool(distance_from_center <= max_cte)

        # DeepRacer logic: reward based on centerline bands
        if not all_wheels_on_track:
            reward = float(self.cfg.r_offtrack)
        else:
            # Compute marker positions as fractions of track width
            marker_1 = self.cfg.m1_frac * track_width
            marker_2 = self.cfg.m2_frac * track_width
            marker_3 = self.cfg.m3_frac * track_width

            # Assign reward based on distance from center
            reward = float(self.cfg.r_offtrack)
            if distance_from_center <= marker_1:
                reward = float(self.cfg.r_center)
            elif distance_from_center <= marker_2:
                reward = float(self.cfg.r_mid)
            elif distance_from_center <= marker_3:
                reward = float(self.cfg.r_edge)

            # Add speed bonus
            reward += speed * float(self.cfg.speed_scale)

        reward_info = {
            "reward": float(reward),
            "speed": float(speed),
            "cte": None if cte is None else float(cte),
            "distance_from_center": float(distance_from_center),
            "all_wheels_on_track": bool(all_wheels_on_track),
            "track_width": float(track_width),
            "max_cte": float(max_cte),
            "done": bool(done),
        }

        return float(reward), reward_info


class JetRacerCenterlineV2RewardWrapper(BaseRewardWrapper):
    """Reward wrapper implementing CenterlineV2RewardConfig.

    Features: centerline following, speed reward, anti-stall, smoothness, caution.
    Adds info under `info["centerline_v2_reward"]`.
    """

    def __init__(self, env: gym.Env, cfg: CenterlineV2RewardConfig = CenterlineV2RewardConfig()):
        """Initialize the CenterlineV2 reward wrapper.

        Args:
            env: The environment to wrap.
            cfg: CenterlineV2 reward configuration.
        """
        super().__init__(env, reward_info_key="centerline_v2_reward")
        self.cfg = cfg
        self._prev_steering: float = 0.0

    def _compute_reward(
        self,
        cte: Optional[float],
        speed: float,
        hit: Optional[str],
        throttle: float,
        steering: float,
        done: bool,
    ) -> Tuple[float, Dict]:
        """Compute CenterlineV2 reward.

        If cte is unavailable, returns a default penalty.

        Args:
            cte: Cross-track error, None if unavailable.
            speed: Forward speed.
            hit: Collision signal.
            throttle: Last throttle action.
            steering: Last steering action.
            done: Whether episode is done.

        Returns:
            Tuple of (reward, info_dict).
        """
        # Handle missing cte
        if cte is None:
            reward_info = {
                "reward": -1.0,
                "cte": None,
                "speed": float(speed),
                "throttle": float(throttle),
                "steering": float(steering),
                "note": "missing_cte",
                "done": bool(done),
            }
            return -1.0, reward_info

        max_cte = float(self.cfg.max_cte)
        abs_cte = abs(float(cte))
        offtrack = bool(abs_cte > max_cte)
        collision = bool(isinstance(hit, str) and hit != "none")

        if offtrack:
            reward = -float(self.cfg.offtrack_penalty)
        else:
            # Centerline score: 1.0 at center, 0.0 at edge
            center_score = float(np.clip(1.0 - (abs_cte / max_cte), 0.0, 1.0))

            reward = 0.0
            reward += float(self.cfg.w_center) * center_score
            reward += float(self.cfg.w_speed) * float(speed)

            # Anti-stall: penalize being below min_speed
            if float(speed) < float(self.cfg.min_speed):
                gap = float(self.cfg.min_speed) - float(speed)
                reward -= float(self.cfg.w_stall) * gap

            # Smoothness: penalize steering magnitude and steering change
            steer_rate = float(steering) - float(self._prev_steering)
            reward -= float(self.cfg.w_smooth) * float((steering**2) + (steer_rate**2))
            self._prev_steering = float(steering)

            # Caution: going fast while turning or off-center is risky
            risk = float(abs(float(steering)) + (abs_cte / max_cte))
            reward -= float(self.cfg.w_caution) * float(speed) * risk

        # Collision penalty
        if collision:
            reward -= float(self.cfg.collision_penalty)

        reward_info = {
            "reward": float(reward),
            "cte": float(cte),
            "abs_cte": float(abs_cte),
            "max_cte": float(max_cte),
            "offtrack": bool(offtrack),
            "speed": float(speed),
            "throttle": float(throttle),
            "steering": float(steering),
            "hit": hit,
            "collision": bool(collision),
            "done": bool(done),
        }

        return float(reward), reward_info


class JetRacerCenterlineV3RewardWrapper(BaseRewardWrapper):
    """Reward wrapper implementing CenterlineV3RewardConfig.

    Features: centerline following, speed reward, strong anti-stall, alive bonus.
    Simplified compared to V2 (no smoothness/caution terms).
    Adds info under `info["centerline_v3_reward"]`.
    """

    def __init__(self, env: gym.Env, cfg: CenterlineV3RewardConfig = CenterlineV3RewardConfig()):
        """Initialize the CenterlineV3 reward wrapper.

        Args:
            env: The environment to wrap.
            cfg: CenterlineV3 reward configuration.
        """
        super().__init__(env, reward_info_key="centerline_v3_reward")
        self.cfg = cfg

    def _compute_reward(
        self,
        cte: Optional[float],
        speed: float,
        hit: Optional[str],
        throttle: float,
        steering: float,
        done: bool,
    ) -> Tuple[float, Dict]:
        """Compute CenterlineV3 reward.

        If cte is unavailable, returns a default penalty.

        Args:
            cte: Cross-track error, None if unavailable.
            speed: Forward speed.
            hit: Collision signal.
            throttle: Last throttle action.
            steering: Last steering action.
            done: Whether episode is done.

        Returns:
            Tuple of (reward, info_dict).
        """
        # Handle missing cte
        if cte is None:
            reward_info = {
                "reward": -1.0,
                "cte": None,
                "speed": float(speed),
                "throttle": float(throttle),
                "steering": float(steering),
                "note": "missing_cte",
                "done": bool(done),
            }
            return -1.0, reward_info

        max_cte = float(self.cfg.max_cte)
        abs_cte = abs(float(cte))
        offtrack = bool(abs_cte > max_cte)
        collision = bool(isinstance(hit, str) and hit != "none")

        if offtrack:
            reward = -float(self.cfg.offtrack_penalty)
        else:
            # Centerline score: 1.0 at center, 0.0 at edge
            center_score = float(np.clip(1.0 - (abs_cte / max_cte), 0.0, 1.0))

            reward = 0.0
            reward += float(self.cfg.alive_bonus)  # Per-step bias to avoid zero-action
            reward += float(self.cfg.w_center) * center_score
            reward += float(self.cfg.w_speed) * float(speed)

            # Strong anti-stall: penalize being below min_speed
            if float(speed) < float(self.cfg.min_speed):
                gap = float(self.cfg.min_speed) - float(speed)
                reward -= float(self.cfg.w_stall) * gap

            # Penalize reversing (negative throttle)
            if float(throttle) < 0.0:
                reward -= float(self.cfg.reverse_penalty) * float(-float(throttle))

        # Collision penalty
        if collision:
            reward -= float(self.cfg.collision_penalty)

        reward_info = {
            "reward": float(reward),
            "cte": float(cte),
            "abs_cte": float(abs_cte),
            "max_cte": float(max_cte),
            "offtrack": bool(offtrack),
            "speed": float(speed),
            "min_speed": float(self.cfg.min_speed),
            "throttle": float(throttle),
            "steering": float(steering),
            "hit": hit,
            "collision": bool(collision),
            "done": bool(done),
        }

        return float(reward), reward_info


class JetRacerCenterlineV4RewardWrapper(BaseRewardWrapper):
    """Reward wrapper implementing CenterlineV4RewardConfig.

    Features: centerline following, speed reward, smoothness, alive bonus, anti-stall.
    Adds info under `info["centerline_v4_reward"]`.
    """

    def __init__(self, env: gym.Env, cfg: CenterlineV4RewardConfig = CenterlineV4RewardConfig()):
        """Initialize the CenterlineV4 reward wrapper.

        Args:
            env: The environment to wrap.
            cfg: CenterlineV4 reward configuration.
        """
        super().__init__(env, reward_info_key="centerline_v4_reward")
        self.cfg = cfg
        self._prev_steering: float = 0.0

    def _compute_reward(
        self,
        cte: Optional[float],
        speed: float,
        hit: Optional[str],
        throttle: float,
        steering: float,
        done: bool,
    ) -> Tuple[float, Dict]:
        """Compute CenterlineV4 reward.

        If cte is unavailable, returns a default penalty.

        Args:
            cte: Cross-track error, None if unavailable.
            speed: Forward speed.
            hit: Collision signal.
            throttle: Last throttle action.
            steering: Last steering action.
            done: Whether episode is done.

        Returns:
            Tuple of (reward, info_dict).
        """
        # Handle missing cte
        if cte is None:
            reward_info = {
                "reward": -1.0,
                "cte": None,
                "speed": float(speed),
                "throttle": float(throttle),
                "steering": float(steering),
                "note": "missing_cte",
                "done": bool(done),
            }
            return -1.0, reward_info

        max_cte = float(self.cfg.max_cte)
        abs_cte = abs(float(cte))
        offtrack = bool(abs_cte > max_cte)
        collision = bool(isinstance(hit, str) and hit != "none")

        steer_rate = 0.0
        smooth_pen = 0.0

        if offtrack:
            reward = -float(self.cfg.offtrack_penalty)
        else:
            # Centerline score: 1.0 at center, 0.0 at edge
            center_score = float(np.clip(1.0 - (abs_cte / max_cte), 0.0, 1.0))

            # Smoothness: penalize steering magnitude and steering change
            steer_rate = float(steering) - float(self._prev_steering)
            smooth_pen = float((steering**2) + (steer_rate**2))
            self._prev_steering = float(steering)

            reward = 0.0
            reward += float(self.cfg.alive_bonus)
            reward += float(self.cfg.w_center) * center_score
            reward += float(self.cfg.w_speed) * float(speed)
            reward -= float(self.cfg.w_smooth) * smooth_pen

            # Anti-stall: penalize being below min_speed
            if float(speed) < float(self.cfg.min_speed):
                gap = float(self.cfg.min_speed) - float(speed)
                reward -= float(self.cfg.w_stall) * gap

            # Penalize reversing (negative throttle)
            if float(throttle) < 0.0:
                reward -= float(self.cfg.reverse_penalty) * float(-float(throttle))

        # Collision penalty
        if collision:
            reward -= float(self.cfg.collision_penalty)

        reward_info = {
            "reward": float(reward),
            "cte": float(cte),
            "abs_cte": float(abs_cte),
            "max_cte": float(max_cte),
            "offtrack": bool(offtrack),
            "speed": float(speed),
            "min_speed": float(self.cfg.min_speed),
            "throttle": float(throttle),
            "steering": float(steering),
            "steer_rate": float(steer_rate),
            "smooth_pen": float(smooth_pen),
            "hit": hit,
            "collision": bool(collision),
            "done": bool(done),
        }

        return float(reward), reward_info
