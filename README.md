# Setup
Use python 3.9
Follow setup.sh to build env.
We use stable-baselines3, gymnasium


# Jetson Nano Setup (Python 3.6, inference-only)

Jetson Nano (JetPack 4.x) often needs Python 3.6 to use NVIDIA prebuilt PyTorch wheels.
This repo’s training stack (SB3 2.x + Gymnasium) is NOT compatible with Python 3.6.

For Nano we recommend a separate inference-only environment:

```bash
# ===== Complete Jetson Nano Setup =====

# Step 1: Install system dependencies (numpy/opencv from apt to avoid Illegal Instruction)
sudo apt-get update
sudo apt-get install -y python3-pip python3-numpy python3-opencv

# Step 2: (IMPORTANT) Delete old venv if exists (ensures clean install)
rm -rf .venv_nano_py36

# Step 3: Run setup script (installs virtualenv, torch, sb3, etc.)
bash setup_nano.sh

# Step 4: Activate the environment
source .venv_nano_py36/bin/activate

# Step 5: Verify everything works
python -c "import numpy; import cv2; import torch; import stable_baselines3; print('All OK!')"
```

**Note**: If you need a different PyTorch wheel for your JetPack version, set before running:
```bash
export TORCH_WHEEL=/path/to/your/torch-xxx.whl
bash setup_nano.sh
```


# Run on Jetson Nano (CSI camera)

```bash
python3 run_policy_real.py --mode real --model models/<run>/best_model.zip --camera-gst "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1 sync=false" --fps 15 --deterministic
```

```bash
python3 run_policy_real.py --mode mock --model models/<run>/best_model.zip --camera-gst "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1 sync=false" --fps 15 --deterministic
```