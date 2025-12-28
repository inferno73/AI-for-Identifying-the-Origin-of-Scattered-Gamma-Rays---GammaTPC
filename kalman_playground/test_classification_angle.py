import numpy as np
import glob
import os
import sys
import plotly.graph_objects as go
from tqdm import tqdm

# Fix path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(proj_root)
sys.path.append(os.path.join(proj_root, 'AI_model_method'))

try:
    from AI_model_method.kalman_tracking import TrackKalmanFilter
except ImportError:
    print("Failed to import KF")
    sys.exit(1)

def load_track_raw(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        if 'hits' in data: coords = data['hits']
        elif 'coords' in data: coords = data['coords']
        elif 'x' in data: coords = np.stack([data['x'], data['y'], data['z']], axis=1)
        else: return None
        if coords.shape[1] > 3: coords = coords[:, :3]
    return coords

def sort_points_nn(coords, origin):
    if len(coords) == 0: return coords
    dists_origin = np.linalg.norm(coords - origin, axis=1)
    start_idx = np.argmin(dists_origin)
    sorted_coords = [coords[start_idx]]
    mask = np.ones(len(coords), dtype=bool)
    mask[start_idx] = False
    current_idx = start_idx
    count = 0 
    while np.any(mask):
        last_pt = coords[current_idx]
        unvisited_indices = np.where(mask)[0]
        dists = np.linalg.norm(coords[unvisited_indices] - last_pt, axis=1)
        local_min_idx = np.argmin(dists)
        real_idx = unvisited_indices[local_min_idx]
        mask[real_idx] = False
        current_idx = real_idx
        sorted_coords.append(coords[real_idx])
        count+=1
        if count > len(coords) + 1: break
    return np.array(sorted_coords)

def get_kf_density_fraction(coords, origin, threshold_deg=3.0):
    # 1. KF Direction
    kf = TrackKalmanFilter(dt=1.0)
    kf.Q = np.eye(6) * 0.001
    kf.R = np.eye(3) * 0.1
    kf_dir, _ = kf.fit(coords)
    
    # Check first 20 pts
    n = min(len(coords), 20)
    local_pts = coords[:n]
    if len(local_pts) < 1: return 0
    
    vecs = local_pts - origin
    norms = np.linalg.norm(vecs, axis=1)
    valid = norms > 1e-9
    vecs = vecs[valid]
    norms = norms[valid]
    vecs = vecs / norms[:, None]
    
    # Dot product
    dots = np.dot(vecs, kf_dir)
    # Clamp
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    
    # We care about alignment in either direction?
    # Usually KF aligns with the track.
    # Check absolute angle or normal?
    # Points should clearly be close to 0 deg relative to KF dir.
    
    in_cone = np.sum(angles < threshold_deg)
    
    return in_cone / len(local_pts)

def analyze():
    ROOT_DIR = r"C:\Users\Korisnik\PycharmProjects\gammaAIModel\data-samples\local_data"
    COMPTON_DIR = os.path.join(ROOT_DIR, "compton")
    PAIR_DIR = os.path.join(ROOT_DIR, "pair")
    
    vals_c = []
    vals_p = []
    
    # 1. Compton
    print("Scanning Compton (KF Density)...")
    c_files = glob.glob(os.path.join(COMPTON_DIR, "**", "*.npz"), recursive=True)
    c_files = [f for f in c_files if not f.endswith('_truth.npz')]
    np.random.shuffle(c_files)
    c_files = c_files[:300]
    
    for f in tqdm(c_files):
        coords = load_track_raw(f)
        try:
             # Sort
             sorted_coords = sort_points_nn(coords, np.array([0,0,0]))
             metric = get_kf_density_fraction(sorted_coords, np.array([0,0,0]))
             vals_c.append(metric)
        except Exception as e: pass
        
    # 2. Pair
    print("Scanning Pair (KF Density)...")
    p_files = glob.glob(os.path.join(PAIR_DIR, "**", "*.npz"), recursive=True)
    p_files = [f for f in p_files if not f.endswith('_truth.npz')]
    
    for f in tqdm(p_files):
        coords = load_track_raw(f)
        try:
             sorted_coords = sort_points_nn(coords, np.array([0,0,0]))
             metric = get_kf_density_fraction(sorted_coords, np.array([0,0,0]))
             vals_p.append(metric)
        except: pass

    # Stats
    vc = np.array(vals_c)
    vp = np.array(vals_p)
    print(f"Compton Mean Density: {np.mean(vc):.4f} +/- {np.std(vc):.4f}")
    print(f"Pair Mean Density:    {np.mean(vp):.4f} +/- {np.std(vp):.4f}")
    
    # Histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=vals_c, name='Compton', opacity=0.75, marker_color='blue', nbinsx=20))
    fig.add_trace(go.Histogram(x=vals_p, name='Pair', opacity=0.75, marker_color='red', nbinsx=20))
    
    fig.update_layout(title="KF Cone Density Fraction (3 deg)", xaxis_title="Fraction in Cone", barmode='overlay')
    fig.write_html("track_classification_density.html")
    print("Saved plot.")

if __name__ == "__main__":
    analyze()
