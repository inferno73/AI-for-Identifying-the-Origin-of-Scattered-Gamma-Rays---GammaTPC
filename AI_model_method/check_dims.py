
import numpy as np
import sys
import os

def check(path):
    print(f"Checking: {path}")
    if not os.path.exists(path):
        print("File not found!")
        return
    
    try:
        data = np.load(path, allow_pickle=True)
        if 'features' not in data:
            print("Key 'features' missing!")
            return
            
        feats = data['features']
        print(f"Total Samples: {len(feats)}")
        if len(feats) > 0:
            shape = feats[0].shape
            print(f"Feature Shape of Sample 0: {shape}")
            if len(shape) > 1 and shape[1] == 4:
                print("PASSED: 4 Channels detected.")
            else:
                print(f"FAILED: Expected 4 channels, got {shape}. This is the OLD dataset.")
        else:
            print("Dataset is empty.")
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_dims.py <path_to_npz>")
    else:
        check(sys.argv[1])
