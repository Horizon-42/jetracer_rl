# Random Friction (Domain Randomization)

## Goal
In sim-to-real, one common gap is **vehicle dynamics**: different floors/tires/battery levels change how much the car accelerates for the same throttle command.

DonkeySim (via `gym-donkeycar`) does not consistently expose a simple “friction coefficient” parameter across scenes.
So we implement **random friction** as a practical proxy: **scale the throttle**.

This gives a similar effect:
- lower friction ⇒ for the same throttle, the car effectively accelerates less / moves slower
- higher friction ⇒ for the same throttle, the car moves faster

## Implementation Summary
We implement random friction by sampling a **per-episode throttle multiplier**:

- Sample once on `reset()`:

$$
 s \sim \mathrm{Uniform}(s_{min}, s_{max})
$$

- Apply on every `step()` to the **Donkey action** (the environment expects `[steer, throttle]`):

$$
 throttle' = \mathrm{clip}(throttle \cdot s, 0, 1)
$$

This is done in a wrapper that sits **before** the JetRacer action mapping wrapper:

- `RandomFrictionWrapper` operates in Donkey action space (`[steer, throttle]`).
- `JetRacerWrapper` maps JetRacer action space (`[throttle, steering]`) → Donkey action space.

Order in `donkey_rl/env.py`:
1) base env (`gym.make(...)`)
2) `RandomFrictionWrapper` (if enabled)
3) `JetRacerWrapper`
4) reward wrapper
5) observation preprocessing

## Where in Code
- Wrapper: `donkey_rl/wrappers.py` → `RandomFrictionWrapper`
- Wiring: `donkey_rl/env.py` applies it when `random_friction=True`
- CLI flags: `donkey_rl/args.py`
  - `--random-friction`
  - `--friction-min`
  - `--friction-max`

The wrapper also writes the sampled value into `info`:
- `info["friction_scale"] = s`

So you can log/inspect it during training.

## How to Use
Enable it during training:

- Basic:
  - `python3 train_jetracer_centerline.py --random-friction`

- With custom range:
  - `python3 train_jetracer_centerline.py --random-friction --friction-min 0.6 --friction-max 1.0`

## Tuning Tips
- Start mild: `0.8 .. 1.0` to avoid making control too hard early.
- If real car is often slower than sim, shift range down (e.g. `0.6 .. 0.9`).
- If training becomes unstable, narrow the range.

## Notes / Limitations
- This is a **proxy** for friction; it does not model lateral slip, tire saturation, or true contact physics.
- It is still useful because many policies are sensitive to speed dynamics.
- If you later find a reliable DonkeySim parameter for friction, we can swap the implementation to set that directly.
