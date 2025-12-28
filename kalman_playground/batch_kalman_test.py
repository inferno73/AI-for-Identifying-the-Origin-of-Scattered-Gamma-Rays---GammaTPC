
import os
import sys
import numpy as np
import glob
import pandas as pd
import argparse
import plotly.graph_objects as go
from tqdm import tqdm

# Adjust path to find modules: Parent directory of 'kalman_playground' is project root
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(proj_root)
sys.path.append(os.path.join(proj_root, 'AI_model_method')) # Fix for sibling imports
print(f"DEBUG: Appended to sys.path: {proj_root} and AI_model_method")
print(f"DEBUG: Appended to sys.path: {proj_root}")
print(f"DEBUG: sys.path: {sys.path}")

try:
    from AI_model_method.kalman_tracking import TrackKalmanFilter
    from AI_model_method.evaluate_kf import plot_track_3d_plotly # Reuse plotter if possible, or copy it
except ImportError as e1:
    # If running from root without AI_model_method in path
    try:
        from kalman_tracking import TrackKalmanFilter
        # We might need to copy the plotter if import fails or structure differs
    except ImportError as e2:
        print(f"Could not import modules. Ensure you are in project root.")
        print(f"Error 1 (AI_model_method...): {e1}")
        print(f"Error 2 (Direct import): {e2}")
        sys.exit(1)

# --- REIMPLEMENTING PLOTTER TO BE SAFE & INDEPENDENT ---
def plot_track_3d_plotly_local(result, save_dir, prefix):
    # Simplified version of the one in evaluate_kf.py, adapted for Raw Tracks (no Pred Origin)
    positions = result['coords'] 
    
    true_origin = result['true_origin_abs']
    
    # Directions
    true_dir = result['lbl_dir']
    pred_dir = result['pred_dir']
    
    # 1. Track Points
    track_trace = go.Scatter3d(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], mode='markers',
                               marker=dict(size=5, color=positions[:, 2], colorscale='Viridis', opacity=0.9), 
                               name='Track Points')
                                           
    # 2. True Origin
    true_org_trace = go.Scatter3d(x=[true_origin[0]], y=[true_origin[1]], z=[true_origin[2]],
                                     mode='markers', marker=dict(color='red', size=5, symbol='circle'),
                                     name='True Origin')
                                     
    # Direction vectors (scaled)
    # Dynamic scale based on track extent
    bbox_diag = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))
    scale = bbox_diag * 0.75 if bbox_diag > 1e-6 else 1.0 # Smaller vectors as requested
    
    # 3. True Direction
    td_end = true_origin + true_dir * scale
    true_dir_trace = go.Scatter3d(x=[true_origin[0], td_end[0]], y=[true_origin[1], td_end[1]],
                             z=[true_origin[2], td_end[2]],
                             mode='lines', line=dict(color='red', width=5), name='True Direction')

    # 4. Pred Direction
    # Requested to start from True Origin for comparison
    pd_end = true_origin + pred_dir * scale
    
    pred_dir_trace = go.Scatter3d(x=[true_origin[0], pd_end[0]], y=[true_origin[1], pd_end[1]],
                             z=[true_origin[2], pd_end[2]],
                             mode='lines', line=dict(color='magenta', width=5), name='KF Direction')
                             
    # 5. KF Smoothed
    kf_path = result.get('kf_smooth', None)
    if kf_path is not None:
         kf_trace = go.Scatter3d(x=kf_path[:,0], y=kf_path[:,1], z=kf_path[:,2],
                                mode='lines', line=dict(color='cyan', width=3), name='KF Path')
         data = [track_trace, true_org_trace, true_dir_trace, pred_dir_trace, kf_trace]
    else:
         data = [track_trace, true_org_trace, true_dir_trace, pred_dir_trace]
    
    fname = result.get('filename', 'Unknown')
    title = (f"Track: {fname}<br>"
             f"{prefix} | AngErr: {result['ang_err']:.2f} deg")

    layout = go.Layout(title=title, 
                       legend=dict(x=0.01, y=0.99),
                       scene=dict(
                           xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                           aspectmode='data'
                       ))
                       
    fig = go.Figure(data=data, layout=layout)
    
    # Sortable Error Prefix: KF_test_012.34_TrackName.html
    err_str = f"{result['ang_err']:06.2f}"
    out_name = f"KF_test_{err_str}_{fname.replace('.npz','')}.html"
    fig.write_html(os.path.join(save_dir, out_name))


# --- LOADER ---
def load_track_raw(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        # Coordinates
        if 'hits' in data: coords = data['hits']
        elif 'coords' in data: coords = data['coords']
        elif 'x' in data: coords = np.stack([data['x'], data['y'], data['z']], axis=1)
        else: return None, None, None
        
        # Cleanup column count
        if coords.shape[1] > 3: coords = coords[:, :3]
        
    # Truth
    truth_dir = np.array([0, 0, 1])
    truth_org = np.mean(coords, axis=0) # Fallback
    
    truth_path = file_path.replace('.npz', '_truth.npz')
    if os.path.exists(truth_path):
        with np.load(truth_path, allow_pickle=True) as tdata:
            if 'direction' in tdata: truth_dir = tdata['direction']
            if 'origin' in tdata: truth_org = tdata['origin']
            
    return coords, truth_dir, truth_org

# --- MAIN ---
def sort_points_nn(coords, origin):
    """
    Sorts points using Nearest Neighbor greedy strategy starting from
    the point closest to 'origin'.
    """
    if len(coords) == 0: return coords
    
    # 1. Find start point (closest to origin)
    dists_origin = np.linalg.norm(coords - origin, axis=1)
    start_idx = np.argmin(dists_origin)
    
    sorted_coords = [coords[start_idx]]
    
    # Mask for unvisited
    mask = np.ones(len(coords), dtype=bool)
    mask[start_idx] = False
    
    current_idx = start_idx
    
    # 2. Greedy NN
    # N is small (~50-100), so O(N^2) is fine.
    count = 0 
    while np.any(mask):
        last_pt = coords[current_idx]
        
        # Get indices of unvisited points
        unvisited_indices = np.where(mask)[0]
        
        # Distances from last_pt to all unvisited
        dists = np.linalg.norm(coords[unvisited_indices] - last_pt, axis=1)
        
        # Find closest
        local_min_idx = np.argmin(dists)
        real_idx = unvisited_indices[local_min_idx]
        
        # Update
        mask[real_idx] = False
        current_idx = real_idx
        sorted_coords.append(coords[real_idx])
        
        count+=1
        if count > len(coords) + 1: break # Safety
        
    return np.array(sorted_coords)

def select_best_direction_density(kf_dir, sorted_coords, origin, angle_deg=30, kf_bias=1.0):
    """
    Selects the best direction vector among candidates by counting 
    how many points fall within a cone of 'angle_deg' around the vector.
    
    kf_bias: Extra 'votes' added to KF scores.
    """
    # 1. Generate Candidates
    candidates = {}
    
    # KF
    candidates['KF'] = kf_dir
    candidates['KF_Neg'] = -kf_dir
    
    # Secant (Safety)
    k = min(5, len(sorted_coords)-1)
    vec_sec = sorted_coords[k] - origin
    norm_sec = np.linalg.norm(vec_sec)
    if norm_sec > 1e-9:
        candidates['Secant'] = vec_sec / norm_sec
    else:
        candidates['Secant'] = kf_dir # Fallback
        
    # Local PCA (Run 8 Feature)
    n_pca = min(20, len(sorted_coords))
    local_pts = sorted_coords[:n_pca]
    if len(local_pts) > 2:
        mean_l = np.mean(local_pts, axis=0)
        centered_l = local_pts - mean_l
        u, s, vh = np.linalg.svd(centered_l, full_matrices=False)
        pca_vec = vh[0]
        candidates['LocalPCA'] = pca_vec
        candidates['LocalPCA_Neg'] = -pca_vec
        
    # 2. Score Candidates (Cone Density)
    # Cosine threshold for cone
    cos_thresh = np.cos(np.radians(angle_deg))
    
    # Vectors from origin to all points
    n_check = min(20, len(sorted_coords))
    check_points = sorted_coords[:n_check]
    
    vecs_to_pts = check_points - origin
    dists = np.linalg.norm(vecs_to_pts, axis=1)
    valid = dists > 1e-9
    vecs_to_pts = vecs_to_pts[valid]
    dists = dists[valid]
    # Normalize
    vecs_to_pts = vecs_to_pts / dists[:, None]
    
    best_name = 'KF'
    best_score = -999.0
    best_vec = kf_dir
    
    # Score logic
    for name, vec in candidates.items():
        # Dot product
        dots = np.dot(vecs_to_pts, vec)
        # Count in cone
        score = float(np.sum(dots > cos_thresh))
        
        # APPLY BIAS (Hybrid Strategy)
        if 'KF' in name:
            score += kf_bias
        
        if score > best_score:
            best_score = score
            best_name = name
            best_vec = vec
            
    # If tie or 0 score, fallback to Secant
    if best_score <= 0 and 'Secant' in candidates:
        return candidates['Secant']
        
    return best_vec

def run_batch():
    # PATHS
    ROOT_DATA = r"C:\Users\Korisnik\PycharmProjects\gammaAIModel\data-samples\local_data"
    DATASETS = [
        ('compton', os.path.join(ROOT_DATA, 'compton')),
        ('pair', os.path.join(ROOT_DATA, 'pair'))
    ]
    
    # PARAMS (Run 3 Best)
    DT = 1.0
    Q_SCALE = 0.001
    R_SCALE = 0.1
    KF_BIAS = 1.0 # Run 9 Hybrid Strategy
    
    for ds_name, ds_path in DATASETS:
        print(f"=== PROCESSING DATASET: {ds_name} ===")
        OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "run9_hybrid", ds_name)
        
        os.makedirs(OUTPUT_BASE, exist_ok=True)
        plots_dir = os.path.join(OUTPUT_BASE, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Files (Recursive)
        print(f"Searching for tracks in {ds_path}...")
        all_files = glob.glob(os.path.join(ds_path, "**", "*.npz"), recursive=True)
        track_files = [f for f in all_files if not f.endswith('_truth.npz')]
        print(f"Found {len(track_files)} tracks total.")
        
        results = []
        
        # PROCESS ALL
        for f in tqdm(track_files):
            coords, true_dir, true_org = load_track_raw(f)
            if coords is None: continue
            if len(coords) < 10: continue
            
            # 1. Sort (Nearest Neighbor)
            sorted_coords = sort_points_nn(coords, true_org)
                 
            # 2. KF
            kf = TrackKalmanFilter(dt=DT)
            kf.Q = np.eye(6) * Q_SCALE
            kf.R = np.eye(3) * R_SCALE
            kf_dir, kf_smooth = kf.fit(sorted_coords)
            
            # 3. FORCE DIRECTION (Hybrid Biased Density)
            try:
                pred_dir = select_best_direction_density(kf_dir, sorted_coords, true_org, kf_bias=KF_BIAS)
            except Exception as e:
                print(f"Error in density selector for {f}: {e}")
                pred_dir = kf_dir
                
            # 4. Angle Error
            cos_sim = np.dot(true_dir, pred_dir) / (np.linalg.norm(true_dir)*np.linalg.norm(pred_dir) + 1e-9)
            ang_err = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
            
            fname = os.path.basename(f)
            
            results.append({
                'filename': fname,
                'filepath': f, 
                'ang_err': ang_err,
                'coords': coords, # Ensure we keep these for plotting if needed
                'kf_smooth': kf_smooth,
                'pred_dir': pred_dir,
                'lbl_dir': true_dir,
                'lbl_org': np.array([0,0,0]),
                'true_origin_abs': true_org
            })
    
        # SELECT BEST/WORST & PLOT
        df = pd.DataFrame(results)
        if len(df) > 0:
            df_sorted = df.sort_values('ang_err')
            best_30 = df_sorted.head(30)
            worst_30 = df_sorted.tail(30)
            
            plot_rows = pd.concat([best_30, worst_30]).drop_duplicates(subset=['filename'])
            
            print(f"Generating plots for {len(plot_rows)} selected tracks...")
            for idx, row in tqdm(plot_rows.iterrows(), total=len(plot_rows)):
                res = row.to_dict()
                plot_track_3d_plotly_local(res, plots_dir, "KF_Hybrid")
    
        # REPORT
        report_path = os.path.join(OUTPUT_BASE, "report.txt")
        with open(report_path, "w") as f:
            f.write(f"=== KALMAN FILTER HYBRID REPORT (RUN 9 - {ds_name.upper()}) ===\n")
            f.write(f"Note: Run 9 (Cone Density + Local PCA + Bias={KF_BIAS})\n")
            f.write(f"Data Source: {ds_path}\n")
            f.write(f"Files Processed: {len(df)}\n\n")
            
            f.write("--- PARAMETERS ---\n")
            f.write(f"dt: {DT}\n")
            f.write(f"Q_scale: {Q_SCALE}\n")
            f.write(f"R_scale: {R_SCALE}\n")
            f.write(f"KF_Bias: {KF_BIAS}\n\n")
            
            f.write("--- RESULTS (Angle Error) ---\n")
            if len(df) > 0:
                f.write(f"Mean:   {df['ang_err'].mean():.4f} deg\n")
                f.write(f"Median: {df['ang_err'].median():.4f} deg\n")
                f.write(f"StdDev: {df['ang_err'].std():.4f} deg\n")
                f.write(f"Min:    {df['ang_err'].min():.4f} deg\n")
                f.write(f"Max:    {df['ang_err'].max():.4f} deg\n")
            else:
                f.write("No valid tracks found.\n")
                
        print(f"Done {ds_name}. Report saved to {report_path}")
        if len(df) > 0:
            print(f"{ds_name} Mean Error: {df['ang_err'].mean():.2f} deg | Median: {df['ang_err'].median():.2f} deg")

if __name__ == "__main__":
    run_batch()
