# SB3 to ONNX Model Conversion Pipeline

A concise guide for converting Stable-Baselines3 (SB3) PPO models to ONNX format for deployment on edge devices like Jetson Nano.

---

## Overview

The conversion process extracts the Actor network (policy) from SB3 models, exports it to ONNX format, and validates the output consistency.

**Pipeline:**
1. **Wrap Policy**: Extract Actor network, remove Value Net and Log Prob
2. **Export to ONNX**: Use PyTorch's ONNX export
3. **Verify**: Compare PyTorch vs ONNX Runtime outputs
4. **Deploy**: Run inference with ONNX Runtime

---

## Conversion Steps

### 1. Policy Wrapper

Wrap SB3 policy to output deterministic actions (mean of action distribution):

```python
class OnnxablePolicy(th.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        # deterministic=True returns mean, not sampled action
        return self.policy(observation, deterministic=True)[0]
```

### 2. Export

```python
# Load model on CPU
model = PPO.load(model_path, device="cpu")
onnx_policy = OnnxablePolicy(model.policy)

# Create dummy input (match observation shape)
dummy_input = th.randn(1, channels, height, width)

# Export
th.onnx.export(
    onnx_policy,
    dummy_input,
    output_path,
    opset_version=11,          # Use 11 for Jetson Nano compatibility
    input_names=["input"],
    output_names=["action"],
    dynamic_axes={"input": {0: "batch_size"}, "action": {0: "batch_size"}}
)
```

### 3. Verify

```python
# PyTorch inference
torch_out = torch_model(dummy_input).numpy()

# ONNX Runtime inference
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
onnx_out = sess.run(None, {input_name: dummy_input.numpy()})[0]

# Compare (allow small numerical errors)
np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-03, atol=1e-05)
```

---

## Usage

```bash
python export_sb3_ppo_to_onnx.py \
    --model models/your_model/last_model.zip \
    --output onnx_models \
    --height 84 \
    --width 84 \
    --channels 3
```

---

## Key Considerations

### 1. Device Compatibility
- Load model on **CPU** (`device="cpu"`) to avoid CUDA device mismatch
- Use **opset_version=11** for Jetson Nano (Python 3.6) compatibility
- Training: Python 3.9+, Inference: Python 3.6 (Jetson)

### 2. Input Format Consistency

**Training preprocessing:**
- Format: RGB, HWC → CHW (channels-first)
- Normalization: `[0, 255]` → `[0.0, 1.0]` (float32)
- Shape: `(batch, channels, height, width)`

**Inference preprocessing must match exactly:**
```python
# 1. Image preprocessing (perspective transform, resize, etc.)
# 2. BGR → RGB
# 3. HWC → CHW
# 4. Normalize to [0, 1]
# 5. Add batch dimension
obs = preprocess_image(frame, model_width, model_height, ...)
```

### 3. Observation Mode Matching

Ensure inference `obs_mode` matches training:
- `raw`: Original image
- `perspective`: Bird's-eye view
- `mix`: Raw + perspective stacked vertically
- `mask`: HSV color threshold mask

**Check:**
- Perspective transform parameters match
- Image dimensions match (e.g., 84x84)
- Color space conversion (BGR → RGB)

### 4. Output Processing

ONNX model outputs raw action values:

```python
action_raw = sess.run(None, {input_name: obs})[0].flatten()

# Post-process
throttle = np.clip(action_raw[0], 0.0, 1.0)
steering = np.clip(action_raw[1], -1.0, 1.0)

# Optional: Apply gains/offsets
throttle = throttle_gain * (throttle_boost + throttle_scale * throttle)
steering = steering_gain * steering + steering_offset
```

### 5. ONNX Runtime Providers

On Jetson Nano, prefer GPU acceleration:

```python
sess = ort.InferenceSession(
    onnx_path,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

---

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Output mismatch | Preprocessing inconsistency | Verify preprocessing matches training exactly |
| ONNX Runtime load fails | Incompatible opset or missing deps | Use `opset_version=11`, check ONNX Runtime install |
| Slow inference | Using CPU instead of GPU | Ensure `CUDAExecutionProvider` is available |
| Out of memory | Batch size too large | Use batch size=1, consider model quantization |

---

## Deployment Checklist

Before deploying to Jetson Nano:

- [ ] ONNX model verified (outputs match PyTorch)
- [ ] Observation dimensions match (channels, height, width)
- [ ] Preprocessing matches training
- [ ] Observation mode (`obs_mode`) matches
- [ ] ONNX Runtime GPU provider available
- [ ] Input dtype is `float32`, range `[0.0, 1.0]`
- [ ] Output action ranges correct (throttle: [0, 1], steering: [-1, 1])

---

## Related Files

- **Export script**: `export_sb3_ppo_to_onnx.py`
- **Inference script**: `run_policy_onnx.py`
- **Install guide**: `install_onnx_nano.md`
- **Verification script**: `experiments/check_onnx.py`

---

## References

- [ONNX Documentation](https://onnx.ai/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)

