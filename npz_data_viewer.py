import numpy as np
import os

def inspect_npz(file_path):
    """
    Loads an .npz file and prints information about its contents.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Inspecting: {file_path} ")
    try:
        data = np.load(file_path)
        for key in data.files:
            arr = data[key]
            print(f"\nKey: {key}")
            print(f"  Shape: {arr.shape}")
            print(f"  Dtype: {arr.dtype}")
            
            # Print sample stats for numerical arrays
            if np.issubdtype(arr.dtype, np.number):
                print(f"  Min: {np.min(arr)}")
                print(f"  Max: {np.max(arr)}")
                print(f"  Mean: {np.mean(arr)}")
                # Print a small sample of data if it's 1D or 2D
                if arr.size < 20: 
                     print(f"  Data: {arr}")
                else:
                    print(f"  Data (first 5 flat): {arr.flatten()[:5]}")

    except Exception as e:
        print(f"Error loading .npz file: {e}")

if __name__ == "__main__":
    # Example usage with one of the files found in the directory
    # Adjust this path if needed based on where you run the script from
    
    # Compton
    sample_file = os.path.join("data-samples", "compton", "TrackE0010000_D20250906_T1635995585_drift_02.npz")
    inspect_npz(sample_file)
    
    sample_truth_file = os.path.join("data-samples", "compton", "TrackE0010000_D20250906_T1635995585_drift_02_truth.npz")
    inspect_npz(sample_truth_file)

    # Pair
    sample_file = os.path.join("data-samples", "pair", "TrackE0003000_D20250906_T1617483253_drift_10.npz")
    inspect_npz(sample_file)
    
    sample_truth_file = os.path.join("data-samples", "pair", "TrackE0003000_D20250906_T1617483253_drift_10_truth.npz")
    inspect_npz(sample_truth_file)