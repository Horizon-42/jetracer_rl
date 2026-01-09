# centerline_v2 reward

This reward is designed to be:

1. **Centerline-following** (lower reward as |cte| increases)
2. **Not stop / keep moving** (positive speed term + anti-stall penalty)
3. **Smooth yet cautious** (penalize steering magnitude + steering rate; penalize going fast while turning or off-center)
4. **Faster is better** (speed is rewarded, but balanced by caution)

Implementation
- Code: `donkey_rl/rewards.py` (`JetRacerCenterlineV2RewardWrapper`, `CenterlineV2RewardConfig`)
- Select with: `--reward-type centerline_v2`

## Reward terms

Let:
- `cte` = cross-track error from simulator info
- `speed` = forward speed from simulator info
- `steering` = last JetRacer steering action (captured by our wrapper)
- `max_cte` = off-track threshold

When `|cte| > max_cte`:
- reward = `-offtrack_penalty`

Otherwise:
- `center_score = clip(1 - |cte|/max_cte, 0, 1)`
- `risk = |steering| + |cte|/max_cte`

Reward is:

- `w_center * center_score`
- `+ w_speed * speed`
- `- w_stall * max(0, min_speed - speed)`
- `- w_smooth * (steering^2 + (steering - prev_steering)^2)`
- `- w_caution * speed * risk`

Additionally, if a collision is detected (`info["hit"] != "none"`):
- reward -= `collision_penalty`

Debugging:
- The wrapper writes a structured breakdown to `info["centerline_v2_reward"]`.

## Minimal tuning knobs (CLI)

These are exposed as args:
- `--v2-w-speed` (default `0.8`): increase to make it go faster.
- `--v2-w-caution` (default `0.6`): increase to slow down in turns / near edges.
- `--v2-min-speed` (default `0.2`): increase to penalize stopping more strongly.

(Other weights exist in `CenterlineV2RewardConfig`, but the CLI keeps knobs minimal on purpose.)

### Practical tuning guidance

- If it **stops or crawls**: increase `--v2-w-speed` and/or increase `--v2-min-speed`.
- If it **goes too fast and crashes**: increase `--v2-w-caution`.
- If it **oscillates / zig-zags**: consider increasing `w_smooth` inside `CenterlineV2RewardConfig`.

## Usage example

Train with this reward:

```bash
python train_jetracer_centerline.py \
  --reward-type centerline_v2 \
  --v2-w-speed 0.9 \
  --v2-w-caution 0.7 \
  --v2-min-speed 0.25
```
