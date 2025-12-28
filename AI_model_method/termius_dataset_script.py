import os
import glob
import numpy as np
import argparse
from tqdm import tqdm
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

# --- Termius Configuration ---
DEFAULT_INPUT_DIR = "/sdf/home/b/bahrudin/gammaTPC/MLstudy723/processed_tracks906/"
DEFAULT_OUTPUT_DIR = "." # Current directory on Termius

def find_files(input_dir):
    """
    Recursively finds all .npz files in the input directory, excluding _truth files.
    """
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        return []

    print(f"Searching for files in {input_dir}...")
    search_pattern = os.path.join(input_dir, '**', '*.npz')
    all_files = glob.glob(search_pattern, recursive=True)
    track_files = [f for f in all_files if not f.endswith('_truth.npz')]
    return track_files

def load_track(track_path):
    """
    Loads track and truth data. Returns None if truth is missing.
    """
    truth_path = track_path.replace('.npz', '_truth.npz')
    if not os.path.exists(truth_path):
        return None, None
    
    try:
        with np.load(track_path) as data:
            if 'x' in data and 'y' in data and 'z' in data:
                 hits = np.stack([data['x'], data['y'], data['z']], axis=1)
            elif 'hits' in data:
                 hits = data['hits']
            else:
                 hits = data['hits'] # Fallback
                 
            charge = data['charge']
            
        with np.load(truth_path, allow_pickle=True) as truth_data:
            origin = truth_data['origin']
            direction = truth_data['direction']
            
        return hits, charge, origin, direction
        
    except Exception as e:
        # print(f"Error loading {track_path}: {e}") # Reduce noise for large datasets
        return None, None

def augment_and_center(hits, origin, sigma=2.0):
    """
    Perturbs origin and centers the cloud.
    """
    # 1. Perturb Origin
    noise = np.random.normal(0, sigma, 3)
    noisy_origin = origin + noise
    
    # 2. Center cloud
    centered_hits = hits - noisy_origin
    
    return centered_hits, noisy_origin

def process_dataset(input_dir, output_file, limit=None):
    files = find_files(input_dir)
    
    if len(files) == 0:
        print("No files found!")
        return

    if limit:
        files = files[:limit]
        
    print(f"Found {len(files)} tracks. Processing...")
    
    all_coords = []
    all_features = []
    all_labels_origin = [] 
    all_labels_direction = []
    
    # Batch saving to avoid memory overflow for massive datasets?
    # For now, let's keep it in memory as requested, but add a warning.
    
    for f in tqdm(files):
        hits, charge, true_origin, true_direction = load_track(f)
        if hits is None: continue
        if len(hits) < 50: continue

        # Feature Engineering (Fast version)
        features = np.stack([charge], axis=1) 
        
        # Augmentation
        centered_hits, noisy_origin = augment_and_center(hits, true_origin)
        
        # Prepare Label
        label_origin_offset = true_origin - noisy_origin
        
        all_coords.append(centered_hits)
        all_features.append(features)
        all_labels_origin.append(label_origin_offset)
        all_labels_direction.append(true_direction)

    print(f"Splitting {len(all_coords)} samples into Train (80%) and Test (20%)...")
    
    # Create indices
    indices = np.arange(len(all_coords))
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    
    def save_split(idx, name):
        filename = f"{name}.npz"
        path = os.path.join(args.output_dir, filename)
        print(f"Saving {len(idx)} samples to {filename}...")
        np.savez(path, 
                 coords=np.array(all_coords, dtype=object)[idx],
                 features=np.array(all_features, dtype=object)[idx],
                 labels_origin=np.array(all_labels_origin)[idx],
                 labels_direction=np.array(all_labels_direction)[idx])

    save_split(train_idx, "dataset_train")
    save_split(test_idx, "dataset_test")
    
    print(f"Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default to the path user provided
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--limit', type=int, default=None, help="Debug: limit number of tracks")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "dataset_descriptor.npz")
    
    process_dataset(args.input_dir, output_path, args.limit)
