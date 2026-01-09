#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Batch convert models for Jetson Nano compatibility.")
    parser.add_argument("runs", nargs="+", help="List of run names (subdirectories in models/) to convert.")
    
    args = parser.parse_args()
    
    base_models_dir = "models"
    base_output_dir = "models_converted"
    
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        print(f"Created output directory: {base_output_dir}")
    
    for run_name in args.runs:
        input_path = os.path.join(base_models_dir, run_name, "best_model.zip")
        if not os.path.exists(input_path):
            print(f"Skipping '{run_name}': File not found: {input_path}")
            continue
            
        output_run_dir = os.path.join(base_output_dir, run_name)
        if not os.path.exists(output_run_dir):
            os.makedirs(output_run_dir)
            
        output_path = os.path.join(output_run_dir, "best_model.zip")
        
        print(f"Converting '{run_name}'...")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        
        # Invoke run_policy_real.py with the conversion flag
        # We pass minimal required arguments for the script to start up.
        cmd = [
            sys.executable, "run_policy_real.py",
            "--model", input_path,
            "--mode", "real",       # Dummy mode to satisfy argument parser
            "--convert-model", output_path
        ]
        
        try:
            # Run the conversion process
            subprocess.check_call(cmd)
            print(f"  [OK] Converted {run_name}")
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Failed to convert {run_name}. Exit code: {e.returncode}")
        except Exception as e:
            print(f"  [ERROR] Unexpected error converting {run_name}: {e}")
            
    print("\nBatch conversion finished.")

if __name__ == "__main__":
    main()
