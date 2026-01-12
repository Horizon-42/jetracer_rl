# Setup
Use python 3.9
Follow setup.sh to build env.
We use stable-baselines3, gymnasium


# Jetson Nano Setup (Python 3.6, inference-only)

Jetson Nano (JetPack 4.x) often needs Python 3.6 to use NVIDIA prebuilt PyTorch wheels.
This repo’s training stack (SB3 2.x + Gymnasium) is NOT compatible with Python 3.6.

For Nano we recommend a separate inference-only environment use onnx, read detail in install_onnx_nano.md.

# Run on Jetson Nano (CSI camera)

```bash
python3 run_policy_onnx.py --model model.onnx 
```

# PPO Training Configuration

The PPO model configuration used in `train_jetracer_centerline.py`:

| Parameter | Value |
|-----------|-------|
| `policy` | `policy` |
| `env` | `train_env` |
| `verbose` | `1` |
| `seed` | `args.seed` |
| `policy_kwargs` | `policy_kwargs` |
| `n_steps` | `1024` |
| `batch_size` | `256` |
| `learning_rate` | `3e-4` |
| `gamma` | `0.99` |
| `gae_lambda` | `0.95` |
| `clip_range` | `0.1` |
| `ent_coef` | `0.0` |