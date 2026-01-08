from __future__ import annotations

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
    """

    raw = getattr(env, "last_raw_action", None)
    if raw is None:
        return 0.0, 0.0
    try:
        return float(raw[0]), float(raw[1])
    except Exception:
        return 0.0, 0.0


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


class JetRacerDeepRacerRewardWrapper(gym.Wrapper):
    """DeepRacer-style reward wrapper.

    Adds info under `info["deepracer_reward"]`.
    """

    def __init__(self, env: gym.Env, cfg: DeepRacerStyleRewardConfig = DeepRacerStyleRewardConfig()):
        super().__init__(env)
        self.cfg = cfg

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, _base_reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, _base_reward, done, info = result
            terminated, truncated = bool(done), False

        info_dict = dict(info) if isinstance(info, dict) else {}
        cte = info_dict.get("cte")
        speed = float(info_dict.get("speed", 0.0) or 0.0)

        # Map to DeepRacer params.
        max_cte = float(self.cfg.max_cte)
        track_width = 2.0 * max_cte
        distance_from_center = abs(float(cte)) if cte is not None else float("inf")
        all_wheels_on_track = bool(distance_from_center <= max_cte)

        # DeepRacer logic.
        if not all_wheels_on_track:
            reward = float(self.cfg.r_offtrack)
        else:
            marker_1 = self.cfg.m1_frac * track_width
            marker_2 = self.cfg.m2_frac * track_width
            marker_3 = self.cfg.m3_frac * track_width

            reward = float(self.cfg.r_offtrack)
            if distance_from_center <= marker_1:
                reward = float(self.cfg.r_center)
            elif distance_from_center <= marker_2:
                reward = float(self.cfg.r_mid)
            elif distance_from_center <= marker_3:
                reward = float(self.cfg.r_edge)

            reward += speed * float(self.cfg.speed_scale)

        info_dict["deepracer_reward"] = {
            "reward": float(reward),
            "speed": float(speed),
            "cte": None if cte is None else float(cte),
            "distance_from_center": float(distance_from_center),
            "all_wheels_on_track": bool(all_wheels_on_track),
            "track_width": float(track_width),
            "max_cte": float(max_cte),
            "done": bool(done),
        }

        if len(result) == 5:
            return obs, float(reward), bool(terminated), bool(truncated), info_dict
        return obs, float(reward), bool(done), info_dict


class JetRacerRaceRewardWrapper(gym.Wrapper):
    """Original reward wrapper (kept as-is in behavior).

    Rewards progress/speed, penalizes leaving the road, and discourages zig-zag steering.
    Adds info under `info["race_reward"]`.
    """

    def __init__(self, env: gym.Env, cfg: RaceRewardConfig = RaceRewardConfig()):
        super().__init__(env)
        self.cfg = cfg
        self._prev_steering: float = 0.0

    def step(self, action):
        # Gymnasium API returns:
        #   obs, reward, terminated, truncated, info
        # Older Gym API returns:
        #   obs, reward, done, info
        #
        # We support both to keep compatibility.
        result = self.env.step(action)
        if len(result) == 5:
            obs, _base_reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, _base_reward, done, info = result

        # Extract signals from `info`.
        # We keep this wrapper DonkeyCar-only: rely on cte/speed/hit.
        info_dict = dict(info) if isinstance(info, dict) else {}
        cte = info_dict.get("cte")
        speed = float(info_dict.get("speed", 0.0) or 0.0)
        hit = info_dict.get("hit")
        done_why = self._extract_done_code(hit)
        throttle, steering = _try_get_last_raw_action(self.env)

        shaped_reward = 0.0

        # Main idea:
        # - Reward moving forward (proxy: speed)
        # - Penalize being far from center (proxy: |cte|)
        # - Heading error is not available here, so we treat it as 0.
        if cte is not None:
            dist = float(cte)
            angle_rad = 0.0
            dot_dir = 1.0

            # In DonkeyCar we assume dot_dir=1 (we do not have heading alignment info).
            progress = max(0.0, dot_dir) * float(speed)
            shaped_reward += self.cfg.w_progress * progress
            shaped_reward += self.cfg.w_speed * float(speed)
            shaped_reward -= self.cfg.w_center * abs(dist)
            shaped_reward -= self.cfg.w_heading * abs(angle_rad)
        else:
            # If we cannot get cte, we can't measure "staying centered".
            # Provide a small penalty to discourage such states.
            shaped_reward -= 1.0

        # Steering penalties help reduce "zig-zag" which often emerges in vision-based control.
        shaped_reward -= self.cfg.w_steer * float(steering**2)
        shaped_reward -= self.cfg.w_steer_rate * float((steering - self._prev_steering) ** 2)
        self._prev_steering = float(steering)

        if done and done_why in {"offroad-or-collision", "collision"}:
            shaped_reward -= self.cfg.offroad_penalty

        info_dict["race_reward"] = {
            "reward": float(shaped_reward),
            "speed": float(speed),
            "throttle": float(throttle),
            "steering": float(steering),
            "done_why": done_why,
            "cte": None if cte is None else float(cte),
            "hit": hit,
        }

        if len(result) == 5:
            return obs, float(shaped_reward), bool(terminated), bool(truncated), info_dict
        return obs, float(shaped_reward), bool(done), info_dict

    @staticmethod
    def _extract_done_code(hit: object) -> str:
        """Map DonkeyCar collision signal into a simple code string.

        gym-donkeycar often reports collisions via `info["hit"]`.
        Common patterns:
        - "none" => no collision
        - some other string => collided with something
        """
        if isinstance(hit, str) and hit != "none":
            return "collision"
        return "in-progress"


class JetRacerRaceRewardWrapperTrackLimit(gym.Wrapper):
    """New reward variant: original shaping + explicit off-track penalty.

    Adds info under `info["race_reward_track"]`.
    """

    def __init__(
        self,
        env: gym.Env,
        base_cfg: RaceRewardConfig = RaceRewardConfig(),
        track_cfg: DonkeyTrackLimitRewardConfig = DonkeyTrackLimitRewardConfig(),
    ):
        super().__init__(env)
        self.base_cfg = base_cfg
        self.track_cfg = track_cfg
        self._prev_steering: float = 0.0

    def step(self, action):
        # Same compatibility handling as the base reward wrapper.
        result = self.env.step(action)
        if len(result) == 5:
            obs, _base_reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, _base_reward, done, info = result
            terminated, truncated = bool(done), False

        info_dict = dict(info) if isinstance(info, dict) else {}

        # DonkeyCar info fields (typically simulator-provided):
        # - cte: cross-track error (distance from track centerline)
        # - speed: forward speed
        # - hit: collision signal (string)
        #
        # Real car translation:
        # - cte: estimate from lane detection / localization
        # - speed: estimate from encoders/IMU/vision
        # - hit: detect via bumper/IMU/current/vision heuristics
        cte = info_dict.get("cte")
        speed = info_dict.get("speed", 0.0)
        hit = info_dict.get("hit")

        throttle, steering = _try_get_last_raw_action(self.env)

        shaped_reward = 0.0

        # We keep the same "shape" as the base reward:
        #   +progress +speed -|cte| -steering penalties
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

        shaped_reward -= self.base_cfg.w_steer * float(steering**2)
        shaped_reward -= self.base_cfg.w_steer_rate * float((steering - self._prev_steering) ** 2)
        self._prev_steering = float(steering)

        # New part: explicit off-track classification.
        # If the environment reports cte, we treat |cte| > max_cte as "off track".
        #
        # Beginner tip:
        # - If your agent keeps bouncing near the edge, reduce `max_cte` or increase
        #   `offtrack_step_penalty`.
        # - If your agent becomes too slow/cautious, do the opposite.
        offtrack = False
        if cte is not None:
            offtrack = abs(float(cte)) > float(self.track_cfg.max_cte)
            if offtrack:
                shaped_reward -= float(self.track_cfg.offtrack_step_penalty)

        # Terminal penalty: if the episode ended and we were off-track or hit something,
        # apply a larger one-time penalty.
        if done:
            if offtrack or (isinstance(hit, str) and hit != "none"):
                shaped_reward -= float(self.base_cfg.offroad_penalty)

        info_dict["race_reward_track"] = {
            "reward": float(shaped_reward),
            "speed": float(speed),
            "throttle": float(throttle),
            "steering": float(steering),
            "cte": None if cte is None else float(cte),
            "max_cte": float(self.track_cfg.max_cte),
            "offtrack": bool(offtrack),
            "hit": hit,
        }

        if len(result) == 5:
            return obs, float(shaped_reward), bool(terminated), bool(truncated), info_dict
        return obs, float(shaped_reward), bool(done), info_dict
