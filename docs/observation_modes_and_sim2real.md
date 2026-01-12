# Observation Modes and Sim-to-Real Transfer

This document summarizes observation preprocessing modes and domain randomization techniques used to bridge the sim-to-real gap for JetRacer RL training.

---

## Observation Modes Overview

| Mode | Description | Output Shape | Preprocessing | Best For |
|------|-------------|--------------|---------------|----------|
| `raw` | Original camera image, resized only | `(3, H, W)` | Resize | Natural camera view, minimal processing |
| `perspective` | Bird's-eye view transformation | `(3, H, W)` | Perspective transform + resize | Lane detection, geometric consistency |
| `mix` | Raw + perspective stacked vertically | `(3, H, W)` | Stack + resize | Maximum information, dual perspective |
| `mask` | Binary mask from HSV thresholds | `(3, H, W)` | HSV extraction + resize | Sim-to-real, geometry-focused |

**Key Differences:**
- **Raw**: Fastest, preserves natural perspective
- **Perspective**: Geometric normalization, better for structured tasks
- **Mix**: Most information-rich, requires larger effective input
- **Mask**: Most robust to visual variations, requires HSV tuning

---

## Observation Modes

The `ObsPreprocess` wrapper (`donkey_rl/obs_preprocess.py`) provides four observation modes to transform raw simulator frames into policy inputs.

### 1. Raw Mode (`obs_mode="raw"`)

**What it does:**
- Uses the original camera image directly from the simulator
- Applies only resizing to target dimensions (default: 84×84)
- No perspective transformation

**Output:** `(3, height, width)` RGB image in CHW format, float32 [0, 1]

**Pros:**
- Preserves original camera perspective
- Minimal preprocessing overhead
- Natural view similar to real camera

**Cons:**
- May include irrelevant background/distractions
- No geometric normalization

**Use Case:** When you want the policy to learn from the natural camera view.

---

### 2. Perspective Mode (`obs_mode="perspective"`)

**What it does:**
- Applies perspective transformation to create a bird's-eye view
- Transforms camera view → top-down view
- Resizes to target dimensions

**Transformation:**
- Source points: `[(75, 154), (242, 154), (319, 238), (0, 238)]` (camera ROI)
- Destination points: `[(10, 10), (310, 10), (310, 230), (10, 230)]` (bird's-eye view)
- Output size: 320×240 before resizing

**Output:** `(3, height, width)` transformed RGB image in CHW format, float32 [0, 1]

**Pros:**
- Geometric normalization (consistent view regardless of camera angle)
- Better for lane detection and path following
- Reduces perspective distortion effects

**Cons:**
- Loses depth cues from original perspective
- May introduce artifacts at transformation boundaries

**Use Case:** When geometric consistency is more important than preserving natural perspective.

---

### 3. Mix Mode (`obs_mode="mix"`)

**What it does:**
- Stacks raw + perspective images **vertically**
- Both images resized to same width (320×240) before stacking
- Final stack resized to target dimensions

**Output:** `(3, height, width)` stacked image in CHW format, float32 [0, 1]
- Height is effectively doubled (e.g., 84×84 becomes 84×168 effective)

**Pros:**
- Combines benefits of both views
- Network can learn from both perspectives simultaneously
- More information-rich representation

**Cons:**
- Larger effective input size (may require architecture adjustment)
- More computation during preprocessing

**Use Case:** When you want maximum information and can handle larger inputs.

---

### 4. Mask Mode (`obs_mode="mask"`)

**What it does:**
- Extracts binary mask using HSV color thresholds
- Converts RGB → HSV → binary mask (track segmentation)
- Stacks single channel to 3 channels for CNN compatibility
- Resizes to target dimensions

**HSV Thresholds:**
- `mask_hsv_lower`: Lower bound (default: `(0, 0, 0)`)
- `mask_hsv_upper`: Upper bound (default: `(180, 50, 80)`)
- Adjust based on track color in your environment

**Output:** `(3, height, width)` binary mask (repeated 3×) in CHW format, float32 [0, 1]

**Pros:**
- Focuses on track structure, ignores textures/colors
- Robust to lighting variations (if HSV thresholds are tuned)
- Reduces sim-to-real visual gap (geometric vs. photometric)

**Cons:**
- Requires careful HSV threshold tuning
- May lose useful visual cues (shadows, texture)
- Sensitive to color calibration differences

**Use Case:** When track geometry is more important than visual appearance, or for sim-to-real transfer with different lighting/colors.

**Tuning HSV Thresholds:**
```python
# Example: Yellow track centerline
mask_hsv_lower = (10, 100, 100)  # H: 10-25, S: 100-255, V: 100-255
mask_hsv_upper = (25, 255, 255)

# Example: Dark track surface
mask_hsv_lower = (0, 0, 0)
mask_hsv_upper = (180, 50, 80)
```

---

## Domain Randomization (Photometric)

Domain randomization applies random augmentations to simulate real-world variations in lighting, camera settings, and sensor noise.

### Components

**1. Color Jitter (HSV space)**
- Randomly shifts Hue, Saturation, and Value channels
- Simulates different lighting conditions and camera color calibration
- Strength: `aug_color_jitter` (default: 0.2)

**2. Brightness Adjustment**
- Additive offset: `brightness ± random(-range, +range)`
- Simulates exposure variations
- Range: `aug_brightness` (default: 0.25)

**3. Contrast Adjustment**
- Multiplicative factor: `contrast × (1 ± random(-range, +range))`
- Simulates camera gain settings
- Range: `aug_contrast` (default: 0.25)

**4. Gaussian Noise**
- Additive noise: `image + N(0, std²)`
- Simulates sensor noise and quantization artifacts
- Std: `aug_noise_std` (default: 0.02)

**Application Order:**
1. Color jitter (HSV transformations)
2. Contrast adjustment
3. Brightness adjustment
4. Gaussian noise
5. Clip to [0, 1]

**Enable:** `--domain-rand` (enabled by default)

**Tuning Tips:**
- Start with defaults, increase gradually if needed
- Too strong → training instability
- Too weak → limited sim-to-real benefit

---

## Random Friction (Dynamics Randomization)

Random friction simulates variations in vehicle dynamics by scaling throttle per episode.

### How It Works

**Per-Episode Sampling:**
- On `reset()`: Sample friction scale `s ~ Uniform(min, max)`
- Default range: `[0.4, 1.2]`

**Per-Step Application:**
- Multiply throttle by friction scale: `throttle' = clip(throttle × s, -0.5, 1.0)`
- Applied to DonkeyCar action format `[steer, throttle]`

**Implementation:**
- `RandomFrictionWrapper` in `donkey_rl/wrappers.py`
- Applied **before** `JetRacerWrapper` (operates in Donkey action space)
- Friction scale stored in `info["friction_scale"]` for logging

### Why It Helps

**Real-World Variations:**
- Different floor surfaces (carpet, tile, concrete)
- Tire wear and pressure
- Battery voltage (affects motor torque)
- Temperature effects

**Effect:**
- Lower friction → same throttle produces less acceleration
- Higher friction → same throttle produces more acceleration
- Forces policy to adapt to varying dynamics

**Enable:** `--random-friction` (enabled by default)

**Tuning:**
- Conservative: `--friction-min 0.8 --friction-max 1.0`
- Moderate: `--friction-min 0.6 --friction-max 1.2` (default)
- Aggressive: `--friction-min 0.4 --friction-max 1.5`

---

## Sim-to-Real Alignment Strategy

### The Gap

**Simulation:**
- Perfect sensors (no noise)
- Consistent lighting
- Known dynamics
- Perfect ground truth (CTE, speed)

**Real World:**
- Camera noise, compression artifacts
- Varying lighting (indoor/outdoor, shadows)
- Unknown/uncertain dynamics
- Estimated signals (CTE from vision, speed from encoders)

### Alignment Techniques

**1. Observation Alignment**

| Technique | Purpose | Real-World Equivalent |
|-----------|---------|----------------------|
| Domain randomization | Robust to lighting/camera variations | Natural lighting changes |
| Mask mode | Focus on geometry, ignore textures | Track segmentation in real camera |
| Perspective transform | Geometric normalization | Consistent view regardless of camera angle |

**2. Dynamics Alignment**

| Technique | Purpose | Real-World Equivalent |
|-----------|---------|----------------------|
| Random friction | Robust to varying motor/floor conditions | Different surfaces, battery levels |
| Action space matching | Same control interface | JetRacer [throttle, steering] format |

**3. Signal Alignment**

| Signal | Simulation | Real World | Alignment Strategy |
|--------|-----------|------------|-------------------|
| CTE | Ground truth from sim | Estimated from vision | Use same CTE estimation in both |
| Speed | Ground truth from sim | Estimated from throttle/encoders | Train with speed estimation |
| Observations | Perfect RGB | Noisy camera frames | Domain randomization |

---

## Recommended Configurations

### For Sim-to-Real Transfer

**Observation:**
```bash
--obs-mode mask  # Focus on geometry
--mask-hsv-lower "10,100,100"  # Tune for your track
--mask-hsv-upper "25,255,255"
```

**Domain Randomization:**
```bash
--domain-rand  # Enabled by default
--aug-brightness 0.4
--aug-contrast 0.4
--aug-noise-std 0.05
--aug-color-jitter 0.35
```

**Dynamics:**
```bash
--random-friction  # Enabled by default
--friction-min 0.6
--friction-max 1.2
```

### For Pure Simulation Training

**Observation:**
```bash
--obs-mode mix  # Maximum information
# or
--obs-mode perspective  # Geometric normalization
```

**Domain Randomization:**
```bash
--domain-rand  # Still recommended for robustness
```

**Dynamics:**
```bash
--random-friction  # Still recommended
```

---

## Implementation Details

### Wrapper Order

The environment wrappers are applied in this order:

1. **Base environment** (`make_donkey_env`)
   - Creates DonkeyCar environment
   - Adds timeout protection (`StepTimeoutWrapper`)

2. **Random Friction** (`RandomFrictionWrapper`)
   - Scales throttle per episode
   - Operates in Donkey action space `[steer, throttle]`

3. **Action Mapping** (`JetRacerWrapper`)
   - Converts `[throttle, steering]` → `[steer, throttle]`
   - Stores raw action for reward functions

4. **Reward Shaping** (various reward wrappers)
   - Computes shaped rewards based on CTE, speed, etc.

5. **Stall Detection** (`StallDetectionWrapper`)
   - Terminates episodes if car is stuck

6. **Observation Preprocessing** (`ObsPreprocess`)
   - Applies observation mode (raw/perspective/mix/mask)
   - Applies domain randomization (if enabled)
   - Converts to CHW float32 format

### Code Locations

- **Observation preprocessing**: `donkey_rl/obs_preprocess.py`
- **Random friction**: `donkey_rl/wrappers.py` → `RandomFrictionWrapper`
- **Environment building**: `donkey_rl/env.py` → `build_env_fn()`
- **Training script**: `train_jetracer_centerline.py`

---

## Debugging Tips

**Check observation shape:**
```python
print(env.observation_space.shape)  # Should be (3, height, width)
```

**Visualize processed observations:**
- Use `DebugObsDumpCallback` during training
- Check `last_raw_observation`, `last_transformed_observation`, `last_resized_observation`

**Monitor friction scale:**
```python
# In training loop
info = env.step(action)[-1]
print(f"Friction scale: {info.get('friction_scale', 1.0)}")
```

**Tune HSV thresholds:**
- Use OpenCV to visualize HSV ranges
- Test with real camera images if available
- Adjust thresholds based on track color in your environment

---

## References

- Implementation: `donkey_rl/obs_preprocess.py`, `donkey_rl/wrappers.py`
- Training script: `train_jetracer_centerline.py`
- Real car environment: `real_car_env.py` (uses similar preprocessing)

