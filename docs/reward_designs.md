# Reward Function Designs

This document summarizes the reward functions available for JetRacer RL training. All rewards use **CTE (Cross-Track Error)** as the primary signal for centerline following.

---

## Overview

| Reward Type     | Key Features                           | Formula                                                                                                 | Best For                        |
| --------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------- |
| `base`          | Progress + speed + centerline          | `w_progress*progress + w_speed*speed - w_center*\|cte\| - w_steer*steering² - w_steer_rate*steer_rate²` | General racing                  |
| `track_limit`   | Base + explicit off-track penalties    | `base + (if \|cte\|>max_cte: -offtrack_step_penalty)`                                                   | Strict track boundaries         |
| `centerline_v3` | Centerline + speed + strong anti-stall | `alive_bonus + w_center*center_score + w_speed*speed - w_stall*max(0,min_speed-speed)`                  | Fast training, prevent stopping |

---

## 1. Base Reward (`base`)

**Formula:**
```
reward = w_progress * progress + w_speed * speed 
         - w_center * |cte| 
         - w_steer * steering² 
         - w_steer_rate * (steering_change)²
         - offroad_penalty (if done)
```

**Key Terms:**
- `progress`: Forward movement (speed-based)
- `speed`: Direct speed reward
- `|cte|`: Distance from centerline penalty
- `steering²`: Steering magnitude penalty (reduces zig-zag)
- `steer_rate²`: Steering change penalty (smoothness)

**Default Weights:**
- `w_progress = 2.0`, `w_speed = 0.5`, `w_center = 2.0`
- `w_steer = 0.10`, `w_steer_rate = 0.05`
- `offroad_penalty = 50.0`

**Use Case:** General-purpose racing reward with balanced terms.

---

## 2. Track Limit Reward (`track_limit`)

**Formula:** Base reward + **per-step off-track penalty**

```
if |cte| > max_cte:
    reward -= offtrack_step_penalty (every step)
```

**Key Addition:**
- Continuous penalty while off-track (not just at termination)

**Default:** `offtrack_step_penalty = 5.0`, `max_cte = 8.0`

**Use Case:** When you need strict enforcement of track boundaries.

---

## 3. DeepRacer Reward (`deepracer`)

**Formula:** Discrete reward bands based on distance from centerline

```
if |cte| > max_cte:
    reward = r_offtrack (≈ 0)
else:
    reward = r_center/mid/edge (based on distance bands)
    reward += speed * speed_scale
```

**Reward Bands:**
- **Center** (closest): `r_center = 1.0`
- **Mid**: `r_mid = 0.5`
- **Edge**: `r_edge = 0.1`
- **Off-track**: `r_offtrack = 0.001`

**Default:** `speed_scale = 0.5`, `max_cte = 8.0`

**Use Case:** Compatibility with AWS DeepRacer training pipelines.

---

## 4. Centerline V2 Reward (`centerline_v2`)

**Formula:**
```
if |cte| > max_cte:
    reward = -offtrack_penalty
else:
    center_score = 1.0 - |cte|/max_cte
    reward = w_center * center_score
           + w_speed * speed
           - w_stall * max(0, min_speed - speed)
           - w_smooth * (steering² + steer_rate²)
           - w_caution * speed * (|steering| + |cte|/max_cte)
```

**Key Features:**
- **Centerline score**: Linear decay from center (1.0) to edge (0.0)
- **Anti-stall**: Penalizes speed below `min_speed`
- **Smoothness**: Penalizes steering magnitude and changes
- **Caution**: Reduces reward when going fast while turning/off-center

**Tunable Parameters:**
- `--v2-w-speed` (default: 0.8): Speed reward weight
- `--v2-w-caution` (default: 0.6): Caution penalty weight
- `--v2-min-speed` (default: 0.2): Minimum speed threshold

**Use Case:** Balanced performance with smooth, cautious driving.

---

## 5. Centerline V3 Reward (`centerline_v3`)

**Formula:**
```
if |cte| > max_cte:
    reward = -offtrack_penalty
else:
    center_score = 1.0 - |cte|/max_cte
    reward = alive_bonus
           + w_center * center_score
           + w_speed * speed
           - w_stall * max(0, min_speed - speed)
```

**Key Features:**
- **Simplified**: No smoothness/caution terms (fewer parameters)
- **Strong anti-stall**: Higher `w_stall` and `min_speed` defaults
- **Alive bonus**: Small per-step reward to avoid zero-action local optimum

**Tunable Parameters:**
- `--v3-w-speed` (default: 1.2): Speed reward weight
- `--v3-min-speed` (default: 0.35): Minimum speed threshold
- `--v3-w-stall` (default: 2.0): Stall penalty weight
- `--v3-alive-bonus` (default: 0.02): Per-step survival bonus

**Use Case:** Fast training with strong anti-stall bias. Prevents learning to stop.

---

## 6. Centerline V4 Reward (`centerline_v4`)

**Formula:**
```
if |cte| > max_cte:
    reward = -offtrack_penalty
else:
    center_score = 1.0 - |cte|/max_cte
    reward = alive_bonus
           + w_center * center_score
           + w_speed * speed
           - w_smooth * (steering² + steer_rate²)
           - w_stall * max(0, min_speed - speed)
```

**Key Features:**
- **Smooth turns**: Penalizes steering magnitude and changes
- **Anti-stall**: Strong penalty for low speed
- **Alive bonus**: Encourages longer episodes
- **Balanced**: Combines smoothness (V2) with strong anti-stall (V3)

**Tunable Parameters:**
- `--v4-w-speed` (default: 1.0): Speed reward weight
- `--v4-min-speed` (default: 0.25): Minimum speed threshold
- `--v4-w-stall` (default: 3.0): Stall penalty weight
- `--v4-w-smooth` (default: 0.25): Smoothness penalty weight
- `--v4-alive-bonus` (default: 0.03): Per-step survival bonus

**Use Case:** Smooth, fast, stable driving. Recommended for most training.

---

## Common Safety Penalties

All centerline rewards (v2, v3, v4) include:
- **Off-track penalty**: `-50.0` when `|cte| > max_cte`
- **Collision penalty**: `-50.0` when `info["hit"] != "none"`
- **Reverse penalty**: `-2.0 * |throttle|` for negative throttle

---

## Quick Selection Guide

- **New to RL?** → Start with `centerline_v4` (balanced, smooth)
- **Agent stops/crawls?** → Use `centerline_v3` (strong anti-stall)
- **Need smooth turns?** → Use `centerline_v2` or `centerline_v4`
- **Strict boundaries?** → Use `track_limit`
- **DeepRacer compatibility?** → Use `deepracer`

---

## Tuning Tips

1. **Agent stops**: Increase `w_speed` and/or `min_speed`
2. **Agent crashes**: Increase `w_caution` (v2) or reduce `w_speed`
3. **Oscillations**: Increase `w_smooth` (v2/v4)
4. **Too slow**: Increase `w_speed` and reduce `w_caution` (v2)
5. **Off-track too often**: Reduce `max_cte` or increase `offtrack_penalty`

---

## References

- Implementation: `donkey_rl/rewards.py`
- Training script: `train_jetracer_centerline.py`
- Detailed V2 docs: `docs/centerline_v2_reward.md`

