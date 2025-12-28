import numpy as np
import glob
import os
import sys
from tqdm import tqdm
import pandas as pd

# Fix path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(proj_root)
sys.path.append(os.path.join(proj_root, 'AI_model_method'))

try:
    from AI_model_method.kalman_tracking import TrackKalmanFilter
except ImportError:
    pass

def load_track_raw(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        if 'hits' in data: coords = data['hits']
        elif 'coords' in data: coords = data['coords']
        elif 'x' in data: coords = np.stack([data['x'], data['y'], data['z']], axis=1)
        else: return None, None, None
        if coords.shape[1] > 3: coords = coords[:, :3]
        
    truth_dir = np.array([0, 0, 1])
    truth_org = np.mean(coords, axis=0)
    
    truth_path = file_path.replace('.npz', '_truth.npz')
    if os.path.exists(truth_path):
        with np.load(truth_path, allow_pickle=True) as tdata:
            if 'direction' in tdata: truth_dir = tdata['direction']
            if 'origin' in tdata: truth_org = tdata['origin']
            
    return coords, truth_dir, truth_org

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

def select_best_direction_biased(kf_dir, sorted_coords, origin, kf_bias=0):
    # Candidates
    candidates = {}
    candidates['KF'] = kf_dir
    candidates['KF_Neg'] = -kf_dir
    
    # Secant
    k = min(5, len(sorted_coords)-1)
    vec_sec = sorted_coords[k] - origin
    if np.linalg.norm(vec_sec) > 1e-9:
        candidates['Secant'] = vec_sec / np.linalg.norm(vec_sec)
        
    # Local PCA (N=20)
    n_pca = min(20, len(sorted_coords))
    local_pts = sorted_coords[:n_pca]
    if len(local_pts) > 2:
        mean_l = np.mean(local_pts, axis=0)
        centered_l = local_pts - mean_l
        u, s, vh = np.linalg.svd(centered_l, full_matrices=False)
        pca_vec = vh[0]
        candidates['LocalPCA'] = pca_vec
        candidates['LocalPCA_Neg'] = -pca_vec
        
    # Scoring
    cos_thresh = np.cos(np.radians(30))
    n_check = min(20, len(sorted_coords))
    check_points = sorted_coords[:n_check]
    vecs_to_pts = check_points - origin
    dists = np.linalg.norm(vecs_to_pts, axis=1)
    valid = dists > 1e-9
    vecs_to_pts = vecs_to_pts[valid]
    dists = dists[valid]
    vecs_to_pts = vecs_to_pts / dists[:, None]
    
    best_name = 'KF'
    best_score = -999
    best_vec = kf_dir
    
    for name, vec in candidates.items():
        dots = np.dot(vecs_to_pts, vec)
        raw_score = np.sum(dots > cos_thresh)
        
        # APPLY BIAS
        final_score = raw_score
        if 'KF' in name: # KF or KF_Neg
            final_score += kf_bias
            
        if final_score > best_score:
            best_score = final_score
            best_name = name
            best_vec = vec
            
    if best_score <= 0 and 'Secant' in candidates:
         return candidates['Secant']
         
    return best_vec

def run_experiment(bias_val):
    ROOT_DIR = r"C:\Users\Korisnik\PycharmProjects\gammaAIModel\data-samples\local_data"
    
    # 1. Compton (Sample 50 for speed?) No, variance is high. Use 100.
    c_files = glob.glob(os.path.join(ROOT_DIR, "compton", "**", "*.npz"), recursive=True)
    c_files = [f for f in c_files if not f.endswith('_truth.npz')][:100]
    
    # 2. Pair
    p_files = glob.glob(os.path.join(ROOT_DIR, "pair", "**", "*.npz"), recursive=True)
    p_files = [f for f in p_files if not f.endswith('_truth.npz')][:100] # Use 100
    
    # Run
    def eval_files(files):
        errs = []
        kf = TrackKalmanFilter(dt=1.0)
        kf.Q = np.eye(6) * 0.001
        kf.R = np.eye(3) * 0.1
        
        for f in files:
            try:
                coords, true_dir, true_org = load_track_raw(f)
                sorted_coords = sort_points_nn(coords, true_org)
                kf_dir, _ = kf.fit(sorted_coords)
                
                # BIASED SELECTOR
                pred_dir = select_best_direction_biased(kf_dir, sorted_coords, true_org, kf_bias=bias_val)
                
                cos_sim = np.dot(true_dir, pred_dir)
                ang_err = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
                errs.append(ang_err)
            except: pass
        return np.median(errs)

    med_c = eval_files(c_files)
    med_p = eval_files(p_files)
    
    print(f"Bias={bias_val} | Compton Med: {med_c:.2f} | Pair Med: {med_p:.2f}")

if __name__ == "__main__":
    print("Tuning Bias...")
    for b in [0, 1, 2, 3, 5]:
        run_experiment(b)
