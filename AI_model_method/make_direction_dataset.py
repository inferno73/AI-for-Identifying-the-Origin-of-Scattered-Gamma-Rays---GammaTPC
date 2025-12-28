import os
import glob
import numpy as np
import argparse
from tqdm import tqdm
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

def find_files(input_dir):
    """
    Recursively finds all .npz files in the input directory, excluding _truth files.
    """
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
            # Assuming keys based on previous analysis
            # hits = np.stack([data['x'], data['y'], data['z']], axis=1) # Original structure might differ
            # Let's adjust to the viewer output: data['hits'] might not exist, 
            # let's try to reconstruct from typical x,y,z if simple, or use 'hits' if available.
            
            # Based on typical gammaTPC data and previous viewer output:
            if 'x' in data and 'y' in data and 'z' in data:
                 hits = np.stack([data['x'], data['y'], data['z']], axis=1)
            elif 'hits' in data:
                 hits = data['hits']
            else:
                 # Fallback for unknown structure, or check viewer output again?
                 # Viewer output said: hits (298686, 3). So 'hits' key exists in the sample I checked.
                 hits = data['hits']
                 
            charge = data['charge']
            
        with np.load(truth_path, allow_pickle=True) as truth_data:
            origin = truth_data['origin']
            direction = truth_data['direction']
            
        return hits, charge, origin, direction
        
    except Exception as e:
        print(f"Error loading {track_path}: {e}")
        return None, None

def get_local_features(hits, charge, k=25):
    """
    Computes local features for each point using VECTORIZED operations.
    100x faster than looping sklearn.PCA.
    
    Features:
    0: Charge (Raw)
    1: dQ (Contrast)
    2: Linearity (Eigenvalue ratio)
    3: Density (Neighbor dist sum or count)
    """
    N = len(hits)
    features = np.zeros((N, 4))
    
    # 0. Charge
    features[:, 0] = charge
    
    # Build Tree
    tree = cKDTree(hits)
    
    # Query Fixed K for vectorization (Padding handled by collecting k)
    # k=25 provides robust local shape
    dists, indices = tree.query(hits, k=k)
    
    # Gather neighbors: (N, k, 3)
    # Note: indices can include self, which is fine
    neighbors = hits[indices]
    neighbor_charges = charge[indices]
    
    # 1. dQ (Contrast)
    # Average charge of neighbors (N,)
    avg_n_charge = np.mean(neighbor_charges, axis=1)
    features[:, 1] = charge - avg_n_charge

    # 2. Linearity (Vectorized PCA)
    # Center neighbors around their local mean
    # local_means: (N, 1, 3)
    local_means = np.mean(neighbors, axis=1, keepdims=True)
    centered = neighbors - local_means # (N, k, 3)
    
    # Compute Covariance Matrix: (N, 3, 3)
    # Cov = (X^T @ X) / (k-1)
    # matrix multiplication of (N, 3, k) @ (N, k, 3) -> (N, 3, 3)
    cov = np.matmul(centered.transpose(0, 2, 1), centered) / (k - 1)
    
    # Eigen decomposition of symmetric matrices
    # eigvalsh is faster and suitable for covariance matrices
    evals = np.linalg.eigvalsh(cov) # Returns in ascending order
    
    # Linearity = (lambda_3 - lambda_2) / lambda_3  ? Or just lambda_0 / sum?
    # Standard metric: linearity = (e2 - e1) / e2 (sorted)
    # Here evals are e0 <= e1 <= e2 (ascending)
    # Principal axis is e2.
    # Linearity strength = 1.0 if e2 >> e1, e0.
    # Ratio approach: e2 / (sum) = explained variance of PC1.
    
    sum_evals = np.sum(evals, axis=1)
    # Avoid division by zero
    sum_evals[sum_evals < 1e-9] = 1.0
    
    features[:, 2] = evals[:, 2] / sum_evals # Explained variance of 1st component
    
    # 3. Density
    # Sum of distances to neighbors (inverse density proxy) or count within radius?
    # Since we use fixed k, dists is (N, k).
    # Mean distance to 20th neighbor?
    features[:, 3] = np.mean(dists, axis=1) # Smaller = Dense. 
    # Or number of neighbors within 5mm?
    # Vectorized radius count is harder. Let's stick to "Mean Dist to K neighbors" as Density feature.
    # But NN wants "Density" (High = Dense).
    # So: 1.0 / (MeanDist + 1e-3)
    mean_dist = np.mean(dists, axis=1)
    features[:, 3] = 1.0 / (mean_dist + 1e-4)

    return features

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
    if limit:
        files = files[:limit]
        
    print(f"Found {len(files)} tracks. Processing...")
    
    all_coords = []
    all_features = []
    all_labels_origin = [] # True origin RELATIVE to centered system (which is -noise)
    all_labels_direction = []
    
from concurrent.futures import ProcessPoolExecutor
import functools

def process_single_file(f):
    """
    Worker function for parallel processing.
    Returns (hits, features, label_org, label_dir) or None
    """
    try:
        data = load_track(f)
        hits, charge, true_origin, true_direction = data
        if hits is None: return None
        if len(hits) < 50: return None
        
        # Feature Engineering
        features = get_local_features(hits, charge) 
        
        # Augmentation
        centered_hits, noisy_origin = augment_and_center(hits, true_origin)
        
        # Label
        label_origin_offset = true_origin - noisy_origin
        
        return (centered_hits, features, label_origin_offset, true_direction, os.path.basename(f))
    except Exception as e:
        return None

def process_dataset(input_dir, output_file, limit=None):
    files = find_files(input_dir)
    if limit:
        files = files[:limit]
        
    print(f"Found {len(files)} tracks. Processing with Parallel Execution...")
    
    all_coords = []
    all_features = []
    all_labels_origin = [] 
    all_labels_direction = []
    all_filenames = []
    
    # Use 90% of available CPUs
    max_workers = max(1, os.cpu_count() - 2)
    print(f"Starting ProcessPoolExecutor with {max_workers} workers.")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # submit all tasks
        results = list(tqdm(executor.map(process_single_file, files), total=len(files)))
        
    # Filter Nones
    print("Aggregating results...")
    for res in results:
        if res is None: continue
        c, f, lo, ld, fname = res
        all_coords.append(c)
        all_features.append(f)
        all_labels_origin.append(lo)
        all_labels_direction.append(ld)
        all_filenames.append(fname)

    # Splitting into Train (80%) and Test (20%)
    print(f"Splitting {len(all_coords)} samples into Train (80%) and Test (20%)...")
    
    indices = np.arange(len(all_coords))
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    
    def save_split(idx, name):
        filename = f"{name}.npz"
        path = os.path.join(output_dir, filename)
        
        print(f"Saving {len(idx)} samples to {filename}...")
        np.savez(path, 
                 coords=np.array(all_coords, dtype=object)[idx],
                 features=np.array(all_features, dtype=object)[idx],
                 labels_origin=np.array(all_labels_origin)[idx],
                 labels_direction=np.array(all_labels_direction)[idx],
                 filenames=np.array(all_filenames)[idx])

    # The argument 'output_file' passed to this function was 'dataset_descriptor.npz'
    # We will derive the dir from it.
    output_dir = os.path.dirname(output_file)
    if output_dir == "": output_dir = "."
    
    save_split(train_idx, "dataset_train")
    save_split(test_idx, "dataset_test")
    
    print(f"Done. Saved dataset_train.npz and dataset_test.npz to {output_dir}")


if __name__ == "__main__":
    DEFAULT_INPUT_DIR = "/sdf/home/b/bahrudin/gammaTPC/MLstudy723/processed_tracks906/"
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output_dir', type=str, default='/sdf/home/b/bahrudin/gammaTPC/workfolder-zerinaa-new/AI_playground/run2/') # Keep for backward compatibility
    parser.add_argument('--output_file', type=str, default=None, help="Descriptor for output (dir/name)")
    parser.add_argument('--limit', type=int, default=None, help="Debug: limit number of tracks")
    args = parser.parse_args()
    
    if args.output_file:
         output_path = args.output_file
    else:
         os.makedirs(args.output_dir, exist_ok=True)
         output_path = os.path.join(args.output_dir, "dataset_descriptor.npz")
    
    # Ensure dir exists for output_file too
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    process_dataset(args.input_dir, output_path, args.limit)
