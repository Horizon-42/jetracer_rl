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