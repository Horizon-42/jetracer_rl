# Setup
Use python 3.9
Follow setup.sh to build env.
We use stable-baselines3, gymnasium


# Run on Jetson Nano (CSI camera)

```bash
python3 run_policy_real.py --mode real --model models/<run>/best_model.zip --camera-gst "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1 sync=false" --fps 15 --deterministic
```

```bash
python3 run_policy_real.py --mode mock --model models/<run>/best_model.zip --camera-gst "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1 sync=false" --fps 15 --deterministic
```