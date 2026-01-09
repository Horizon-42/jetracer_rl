# Setup
Use python 3.9
Follow setup.sh to build env.
We use stable-baselines3, gymnasium


# Jetson Nano Setup (Python 3.6, inference-only)

Jetson Nano (JetPack 4.x) often needs Python 3.6 to use NVIDIA prebuilt PyTorch wheels.
This repo’s training stack (SB3 2.x + Gymnasium) is NOT compatible with Python 3.6.

For Nano we recommend a separate inference-only environment:

```bash
# 1) (on Jetson) make sure pip + OpenCV are installed
sudo apt-get update
sudo apt-get install -y python3-pip python3-opencv

# 2) (optional) override torch wheel (path or URL)
# If you do NOT set this, setup_nano.sh will use the default NVIDIA Box wheel URL.
# export TORCH_WHEEL=/path/to/torch-*-cp36-cp36m-linux_aarch64.whl

# 3) run the nano setup
bash setup_nano.sh

# 4) activate
source .venv_nano_py36/bin/activate
```


# Run on Jetson Nano (CSI camera)

```bash
python3 run_policy_real.py --mode real --model models/<run>/best_model.zip --camera-gst "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1 sync=false" --fps 15 --deterministic
```

```bash
python3 run_policy_real.py --mode mock --model models/<run>/best_model.zip --camera-gst "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1 sync=false" --fps 15 --deterministic
```