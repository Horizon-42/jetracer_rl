from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


def _try_get_last_raw_action(env: gym.Env) -> Tuple[float, float]:
    """Best-effort extraction of last JetRacer action.

    Avoids importing wrappers here (keeps modules decoupled).
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
    """Reward shaping weights for a "race" objective."""

    w_progress: float = 2.0
    w_speed: float = 0.5
    w_center: float = 2.0
    w_heading: float = 0.3
    w_steer: float = 0.10
    w_steer_rate: float = 0.05
    w_proximity: float = 40.0
    offroad_penalty: float = 50.0


@dataclass(frozen=True)
class DonkeyTrackLimitRewardConfig:
    """Extra reward terms for enforcing 'stay on track' in DonkeyCar."""

    max_cte: float = 8.0
    offtrack_step_penalty: float = 5.0


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
        result = self.env.step(action)
        if len(result) == 5:
            obs, _base_reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, _base_reward, done, info = result

        lane = self._extract_lane_position(info)
        speed = self._extract_speed(info)
        proximity = self._extract_proximity_penalty(info)
        done_why = self._extract_done_code(info)
        throttle, steering = _try_get_last_raw_action(self.env)

        shaped_reward = 0.0

        if lane is not None:
            dist = float(lane.get("dist", 0.0))
            angle_rad = float(lane.get("angle_rad", 0.0))
            dot_dir = float(lane.get("dot_dir", 0.0))

            progress = max(0.0, dot_dir) * float(speed)
            shaped_reward += self.cfg.w_progress * progress
            shaped_reward += self.cfg.w_speed * float(speed)
            shaped_reward -= self.cfg.w_center * abs(dist)
            shaped_reward -= self.cfg.w_heading * abs(angle_rad)
        else:
            shaped_reward -= 1.0

        shaped_reward += self.cfg.w_proximity * float(proximity)

        shaped_reward -= self.cfg.w_steer * float(steering**2)
        shaped_reward -= self.cfg.w_steer_rate * float((steering - self._prev_steering) ** 2)
        self._prev_steering = float(steering)

        if done and done_why in {"invalid-pose", "offroad-or-collision", "collision"}:
            shaped_reward -= self.cfg.offroad_penalty

        info = dict(info) if isinstance(info, dict) else {}
        info["race_reward"] = {
            "reward": float(shaped_reward),
            "speed": float(speed),
            "proximity": float(proximity),
            "throttle": float(throttle),
            "steering": float(steering),
            "done_why": done_why,
            "lane": lane,
        }

        if len(result) == 5:
            return obs, float(shaped_reward), bool(terminated), bool(truncated), info
        return obs, float(shaped_reward), bool(done), info

    @staticmethod
    def _extract_lane_position(info: Dict) -> Optional[Dict]:
        if not isinstance(info, dict):
            return None

        sim = info.get("Simulator")
        if isinstance(sim, dict) and isinstance(sim.get("lane_position"), dict):
            return sim["lane_position"]

        cte = info.get("cte")
        if cte is None:
            return None
        return {"dist": float(cte), "angle_rad": 0.0, "dot_dir": 1.0}

    @staticmethod
    def _extract_speed(info: Dict) -> float:
        if not isinstance(info, dict):
            return 0.0

        sim = info.get("Simulator")
        if isinstance(sim, dict):
            speed = sim.get("robot_speed")
            if speed is not None:
                return float(speed)

        speed = info.get("speed")
        if speed is not None:
            return float(speed)
        return 0.0

    @staticmethod
    def _extract_proximity_penalty(info: Dict) -> float:
        if not isinstance(info, dict):
            return 0.0

        sim = info.get("Simulator")
        if isinstance(sim, dict):
            val = sim.get("proximity_penalty")
            if val is not None:
                return float(val)

        return 0.0

    @staticmethod
    def _extract_done_code(info: Dict) -> str:
        if not isinstance(info, dict):
            return "in-progress"

        sim = info.get("Simulator")
        if isinstance(sim, dict):
            msg = sim.get("msg")
            if isinstance(msg, str):
                low = msg.lower()
                if "invalid pose" in low:
                    return "invalid-pose"
                if "max_steps" in low or "max steps" in low:
                    return "max-steps-reached"

        hit = info.get("hit")
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
        result = self.env.step(action)
        if len(result) == 5:
            obs, _base_reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, _base_reward, done, info = result
            terminated, truncated = bool(done), False

        info_dict = dict(info) if isinstance(info, dict) else {}

        cte = info_dict.get("cte")
        speed = info_dict.get("speed", 0.0)
        hit = info_dict.get("hit")

        throttle, steering = _try_get_last_raw_action(self.env)

        shaped_reward = 0.0

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

        offtrack = False
        if cte is not None:
            offtrack = abs(float(cte)) > float(self.track_cfg.max_cte)
            if offtrack:
                shaped_reward -= float(self.track_cfg.offtrack_step_penalty)

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
