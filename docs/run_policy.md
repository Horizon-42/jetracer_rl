# Run a Trained Model (Sim + JetRacer)

This repo trains SB3 PPO policies that output JetRacer-style actions:

- action = `[throttle, steering]`
  - `throttle` in `[0, 1]`
  - `steering` in `[-1, 1]`

To run a trained model in **DonkeySim** or on a **real JetRacer camera loop**, use:

- `run_policy.py`

## 1) Run in DonkeySim

Start DonkeySim (or let gym start it).

Example:
- `python3 run_policy.py --mode sim --model models/<run_id>/best_model.zip --port 9091 --exe-path /path/to/donkey_sim.x86_64 --deterministic`

Notes:
- `--exe-path remote` means you start the sim manually.
- `--steps N` can stop after N steps.

## 2) Run on JetRacer (Real)

This mode reads frames from a camera and applies the repo’s A-route real preprocessing:
- undistort + color correction + resize → CHW float `[0,1]`

Example (safe, does NOT drive motors):
- `python3 run_policy.py --mode real --model models/<run_id>/best_model.zip --dry-run --camera 0 --deterministic`

To actually drive the robot:
- install the JetRacer control library (`jetracer`), then run without `--dry-run`.

Example:
- `python3 run_policy.py --mode real --model models/<run_id>/best_model.zip --camera 0 --fps 15 --deterministic`

Safety:
- Keep one hand on the power / emergency stop.
- The script sets throttle to 0 on exit.

## Common Gotchas
- If your model was trained with `--use-latent`, the `.zip` still runs the same way. The AE encoder is loaded internally by the policy when the model was created.
- If you used a different observation size during training, pass matching `--obs-width/--obs-height`.
