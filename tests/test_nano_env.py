#!/usr/bin/env python3
"""
Test script to verify RL environment on Jetson Nano (Python 3.6).
Checks for:
- Python version
- PyTorch (and CUDA)
- Gym (version 0.21.0 expected)
- Stable Baselines 3
- Basic training loop functionality
"""

import sys
import platform

def print_header(msg):
    print("\n" + "="*40)
    print(msg)
    print("="*40)

def main():
    print_header("1. System Info")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Check Python 3.6
    if sys.version_info[:2] != (3, 6):
        print("WARNING: Not running on Python 3.6. Jetson Nano usually requires 3.6 for compatibility.")
    else:
        print("Python version OK (3.6).")

    print_header("2. Library Checks")

    # PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA Available: Yes ({torch.cuda.get_device_name(0)})")
        else:
            print("CUDA Available: NO (Running on CPU only, might be slow)")
    except ImportError:
        print("ERROR: Failed to import torch.")
        return

    # NumPy
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except ImportError:
        print("ERROR: Failed to import numpy.")
        return

    # Gym
    try:
        import gym
        print(f"Gym: {gym.__version__}")
        # Check expected version for Nano legacy
        if gym.__version__ != "0.21.0":
            print(f"NOTE: Gym version is {gym.__version__}, legacy Nano setups usually use 0.21.0.")
    except ImportError:
        print("ERROR: Failed to import gym.")
        return

    # Stable Baselines 3
    try:
        import stable_baselines3
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        print(f"Stable Baselines 3: {stable_baselines3.__version__}")
    except ImportError:
        print("ERROR: Failed to import stable_baselines3.")
        return

    print_header("3. Functional Test (Minimal Training Loop)")
    try:
        # Create a simple environment
        env_id = "CartPole-v1" 
        print(f"Creating environment: {env_id}")
        
        # We use a dummy env for basic checking
        env = make_vec_env(env_id, n_envs=1)
        
        print("Initialize PPO model (MlpPolicy)...")
        # Use MlpPolicy for CartPole (vector inputs)
        model = PPO("MlpPolicy", env, verbose=1, device="auto")
        
        print("Running learn(total_timesteps=100)...")
        model.learn(total_timesteps=100)
        
        print("Predicting usage...")
        obs = env.reset()
        action, _ = model.predict(obs)
        print(f"Action predicted: {action}")
        
        print("[SUCCESS] RL stack is functional.")
    except Exception as e:
        print(f"[FAILURE] Functional test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
