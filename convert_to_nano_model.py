#!/usr/bin/env python3
"""
Convert a newer SB3 PPO model (potentially trained with share_features_extractor=False)
to a format compatible with SB3 1.2.0 (standard shared CnnPolicy) on Jetson Nano.

Usage:
  python convert_to_nano_model.py --input models/my_run/best_model.zip --output models/my_run/best_model_nano.zip
"""

import argparse
import sys
import os
import zipfile
import json
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Input model path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output model path")
    args = parser.parse_args()

    # Load the state dict directly from the zip to avoid SB3 loading logic issues
    # (SB3 load requires compatible policy code, which we are trying to fix)
    try:
        with zipfile.ZipFile(args.input, "r") as z:
            # Load PyTorch parameters
            with z.open("policy.pth") as f:
                state_dict = torch.load(f, map_location="cpu")
            
            # Load PyTorch optimizer parameters (optional, but good to preserve if we wanted to continue training)
            # But for inference on Nano, we usually only need policy.
            # We copy other files as-is.
            
            # Check for data/args to see if there are other metadata we need to patch?
            # 'data' file contains the serialized class and arguments.
            # We rely on SB3 1.2.0 default loading which uses default values if fields are missing in constructor.
            # But 'policy_kwargs' in 'data' might cause issues if it contains 'share_features_extractor'.
            
            # For this simple conversion, we only patch the state_dict (policy.pth)
            # and filtering 'data' JSON if needed.
            
            pass
    except Exception as e:
        print(f"Error opening input file: {e}")
        sys.exit(1)

    print(f"Loaded state_dict with {len(state_dict)} keys.")
    
    # 1. Transform state_dict
    new_state_dict = {}
    renamed_count = 0
    skipped_count = 0
    
    for k, v in state_dict.items():
        if k.startswith("pi_features_extractor."):
            # Map Policy encoder -> Shared encoder
            new_k = k.replace("pi_features_extractor.", "features_extractor.")
            new_state_dict[new_k] = v
            renamed_count += 1
        elif k.startswith("vf_features_extractor."):
            # Discard Value encoder
            skipped_count += 1
            pass
        else:
            new_state_dict[k] = v
            
    print(f"Converted keys: {renamed_count} renamed, {skipped_count} skipped (value fn).")
    
    # 2. Modify 'data' JSON if necessary to remove conflicting kwargs
    # We need to read 'data' from zip
    modified_data_json = None
    with zipfile.ZipFile(args.input, "r") as z_in:
        # We need to copy all files to new zip, but replace policy.pth and data
        with zipfile.ZipFile(args.output, "w") as z_out:
            for item in z_in.infolist():
                if item.filename == "policy.pth":
                    continue
                
                content = z_in.read(item.filename)
                
                if item.filename == "data":
                    # Try to patch JSON to remove bad kwargs
                    try:
                        # SB3 stores data as serialized JSON string
                        # But it might be pickled inside?
                        # "data" file in SB3 zip IS a serialized JSON usually.
                        # Wait, save_util.py says: json_to_data...
                        # It is JSON.
                        data_dict = json.loads(content.decode('utf-8'))
                        
                        # Fix policy_kwargs
                        if "policy_kwargs" in data_dict:
                            pk = data_dict["policy_kwargs"]
                            if "share_features_extractor" in pk:
                                print(f"Removing share_features_extractor={pk['share_features_extractor']} from policy_kwargs")
                                del pk["share_features_extractor"]
                        
                        # Also fix observation_space/action_space pickle if we can?
                        # No, they are base64 encoded pickle strings.
                        # We leave them alone, relying on run_policy_real.py to override them manually if needed.
                        # But actually, if we convert the model, we ideally fix them too.
                        # But that requires unpickling/repickling (which requires lib compatibility).
                        # Let's just fix the JSON text structure.
                        
                        modified_data_json = json.dumps(data_dict).encode('utf-8')
                        z_out.writestr("data", modified_data_json)
                        continue
                    except Exception as e:
                        print(f"Warning: Failed to parse/patch 'data' JSON: {e}")
                        # Fallback to writing original
                        z_out.writestr(item, content)
                        continue
                
                # Write original file
                z_out.writestr(item, content)
            
            # Write new policy.pth
            print("Writing adapted policy.pth...")
            with z_out.open("policy.pth", "w") as f_out:
                torch.save(new_state_dict, f_out)
                
    print(f"Saved converted model to: {args.output}")

if __name__ == "__main__":
    main()
