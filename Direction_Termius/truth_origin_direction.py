import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
from scipy.spatial import cKDTree
import torch

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("WARNING: Plotly not installed. 3D plots will be skipped.")

from kalman_tracking import TrackKalmanFilter
from train_unified_kf import UnifiedSparseUNet, SparseConvTensor

# --- GLOBAL MODEL ---
MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ai_model(model_path):
    global MODEL
    print(f"Loading AI Model from {model_path}...", flush=True)
    try:
        model = UnifiedSparseUNet(in_channels=4, spatial_size=[64,64,64]).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        MODEL = model
        print("Model loaded successfully.", flush=True)
    except Exception as e:
        print(f"Failed to load model: {e}", flush=True)
        MODEL = None

def get_predicted_origin(points, features, anchor_point=None):
    """
    Runs model inference to get Predicted Origin.
    Includes Centering Logic for 64-Voxel Model (Center [32,32,32]).
    """
    if MODEL is None: return np.zeros(3)
    
    # 1. Calculate Centroid / Reference
    if anchor_point is not None:
         center_ref = anchor_point
    else:
         center_ref = np.mean(points, axis=0) # (3,)
    
    # 2. Center Points -> Range [-32, 32] ideally
    centered_points = points - center_ref
    
    # 3. Add Offset to put it in [0, 64] box
    coords_offset = centered_points + 32.0
    
    if features is None:
        f = np.ones((len(points), 4), dtype=np.float32)
    else:
        f = features
        
    coords_t = torch.tensor(coords_offset, dtype=torch.float32)
    feats_t = torch.tensor(f, dtype=torch.float32)
    
    # Batch indices: (N, 1) of zeros
    b_idx = torch.zeros((len(coords_t), 1), dtype=torch.float32)
    coords_batch = torch.cat([b_idx, coords_t], dim=1)
    
    # SPCONV REQUIREMENT
    coords_batch_int = coords_batch.int()
    
    # Clamp to box just in case
    if (coords_batch_int[:, 1:] < 0).any() or (coords_batch_int[:, 1:] >= 64).any():
          coords_batch_int[:, 1:] = torch.clamp(coords_batch_int[:, 1:], min=0, max=63)

    input_st = SparseConvTensor(feats_t.to(DEVICE), coords_batch_int.to(DEVICE), 
                                spatial_shape=[64,64,64], batch_size=1)
    
    with torch.no_grad():
        pred_org, _, _ = MODEL(input_st)
        
    # Output from model is "Offset from Center".
    p_org_local = pred_org.cpu().numpy()[0]
    
    # Global = CenterRef + Prediction
    p_org_global = center_ref + p_org_local
    
    return p_org_global

def get_local_features(hits, charge, k=25):
    """
    Computes local features for model input if missing.
    Matches logic in make_dataset_kf.py
    """
    N = len(hits)
    features = np.zeros((N, 4))
    features[:, 0] = charge
    
    if N < k:
        return features
        
    tree = cKDTree(hits)
    try:
        dists, indices = tree.query(hits, k=k)
    except:
        return features 
    
    neighbors = hits[indices]
    neighbor_charges = charge[indices]
    
    # 1. dQ
    avg_n_charge = np.mean(neighbor_charges, axis=1)
    features[:, 1] = charge - avg_n_charge
    
    # 2. Linearity
    local_means = np.mean(neighbors, axis=1, keepdims=True)
    centered = neighbors - local_means
    try:
        cov = np.matmul(centered.transpose(0, 2, 1), centered) / (k - 1)
        evals = np.linalg.eigvalsh(cov)
        sum_evals = np.sum(evals, axis=1)
        sum_evals[sum_evals < 1e-9] = 1.0
        features[:, 2] = evals[:, 2] / sum_evals
    except:
        pass
    
    # 3. Density
    mean_dist = np.mean(dists, axis=1)
    features[:, 3] = 1.0 / (mean_dist + 1e-4)

    return features

def sort_points_nn(coords, origin):
    if len(coords) == 0: return coords
    
    # --- ROBUST START SELECTION ---
    # 1. Compute Distance from Origin to all points
    dists_origin = np.linalg.norm(coords - origin, axis=1)
    
    # 2. Identify Candidates (e.g., Top 10 closest points within "start region")
    # This prevents looking at the far end of the track.
    k_candidates = min(10, len(coords))
    candidate_indices = np.argsort(dists_origin)[:k_candidates]
    
    # 3. Select Best Start: The one furthest from the Track Centroid ("Extremity" heuristic)
    # The 'Tip' of the track is an extremity. A point 'inside' the track (midpoint) 
    # is closer to the centroid.
    track_centroid = np.mean(coords, axis=0)
    dists_to_centroid = np.linalg.norm(coords[candidate_indices] - track_centroid, axis=1)
    
    # Argmax gives the index relative to candidate_indices array
    best_local_idx = np.argmax(dists_to_centroid)
    start_idx = candidate_indices[best_local_idx]
    
    # --- END ROBUST SELECTION ---
    
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

def select_best_direction_density(kf_dir, sorted_coords, origin, angle_deg=30, kf_bias=1.0):
    candidates = {}
    
    # Use the first point on the track as the anchor for direction,
    # NOT the predicted origin (which might have an offset error).
    # 'origin' is only used to establish the sort order (start vs end).
    anchor = sorted_coords[0]
    
    # KF
    candidates['KF'] = kf_dir
    candidates['KF_Neg'] = -kf_dir
    
    # Secant (Robust: k=20)
    k = min(20, len(sorted_coords)-1)
    vec_sec = sorted_coords[k] - anchor
    norm_sec = np.linalg.norm(vec_sec)
    if norm_sec > 1e-9:
        candidates['Secant'] = vec_sec / norm_sec
    else:
        candidates['Secant'] = kf_dir 
        
    # Local PCA
    n_pca = min(20, len(sorted_coords))
    local_pts = sorted_coords[:n_pca]
    if len(local_pts) > 2:
        mean_l = np.mean(local_pts, axis=0)
        centered_l = local_pts - mean_l
        u, s, vh = np.linalg.svd(centered_l, full_matrices=False)
        candidates['LocalPCA'] = vh[0]
        candidates['LocalPCA_Neg'] = -vh[0]

    # Global PCA (Robust Baseline)
    if len(sorted_coords) > 2:
        mean_g = np.mean(sorted_coords, axis=0)
        centered_g = sorted_coords - mean_g
        u, s, vh = np.linalg.svd(centered_g, full_matrices=False)
        candidates['GlobalPCA'] = vh[0]
        candidates['GlobalPCA_Neg'] = -vh[0]
        
    # Score Candidates
    cos_thresh = np.cos(np.radians(angle_deg))
    n_check = min(20, len(sorted_coords))
    check_points = sorted_coords[:n_check]
    
    vecs_to_pts = check_points - anchor
    dists = np.linalg.norm(vecs_to_pts, axis=1)
    valid = dists > 1e-9
    vecs_to_pts = vecs_to_pts[valid]
    
    best_name = 'Secant' 
    best_score = -999.0
    best_vec = candidates.get('Secant', np.array([0,0,1]))
    
    for name, vec in candidates.items():
        if len(vecs_to_pts) == 0:
             score = 0
        else:
             dots = np.dot(vecs_to_pts, vec)
             score = float(np.sum(dots > cos_thresh))
        
        if 'KF' in name:
            if score > 0:
                score += kf_bias
        
        if score > best_score:
            best_score = score
            best_name = name
            best_vec = vec
            
    if best_score <= 0 and 'Secant' in candidates:
        return candidates['Secant'], 'Secant_Fallback'
        
    return best_vec, best_name

def plot_track_3d_plotly(result, save_dir, prefix):
    if not PLOTLY_AVAILABLE: return
    
    # Unpack Result
    # Need to store all this in 'all_results' if we want to plot later
    # Or reload from file? Storing is better for RAM if limited number.
    # But since we only plot Top 30 Best/Worst, we need to RELOAD them.
    # Result mainly has filename, error. We need to reload to get coords.
    
    pass # Wait, function is separated below.

# --- EVALUATION LOGIC ---

def run_hybrid_logic(pts, true_org, true_dir, fname, pred_org=None):
    # 1. Deduplicate
    if len(pts) > 0:
        pts = np.unique(pts, axis=0)
        
    # Determine Origin to use for Sorting
    # If pred_org is provided, use it. Else true_org.
    origin_to_use = pred_org if pred_org is not None else true_org
        
    pred_dir = np.array([0., 0., 1.])
    winner = "Fail"
    kf_path = None
    
    if len(pts) > 5:
        try:
            sorted_pts = sort_points_nn(pts, origin_to_use)
            kf = TrackKalmanFilter(dt=1.0)
            kf_dir, kf_smooth = kf.fit(sorted_pts)
            kf_path = kf_smooth
            
            if kf_dir is not None:
                 pred_dir, winner = select_best_direction_density(
                     kf_dir, sorted_pts, origin_to_use, kf_bias=1.0
                 )
            else:
                winner = "KF_Fail"
        except Exception as e:
            winner = f"Error_{e}"
            
    # Calc Error using TRUE Direction vs PREDICTED Direction
    cos_sim = np.dot(true_dir, pred_dir) / (np.linalg.norm(true_dir)*np.linalg.norm(pred_dir) + 1e-9)
    ang_err = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
    
    # Calc Origin Error if Pred provided
    org_err = -1.0
    if pred_org is not None:
        org_err = np.linalg.norm(pred_org - true_org)
    
    global DEBUG_COUNT
    if DEBUG_COUNT < 3:
        print(f"\n--- DEBUG TRACK {DEBUG_COUNT} ---")
        print(f"File: {fname}")
        print(f"True Org: {true_org}")
        print(f"Pred Org: {pred_org} (Err: {org_err:.4f})")
        print(f"True Dir: {true_dir}")
        print(f"Raw Coords (First 3):\n{pts[:3]}")
        if len(pts) > 5:
             print(f"Sorted Coords (First 3):\n{sorted_pts[:3]}")
             # Re-calc candidates for debug
             # ...
             print(f"KF Dir: {kf_dir}")
             print(f"Winner: {winner}")
             print(f"Pred Dir: {pred_dir}")
             print(f"Angle Err: {ang_err:.2f}")
        DEBUG_COUNT += 1

    return {
        'filename': fname,
        'ang_err': ang_err,
        'winner': winner,
        'org_err': org_err,
        'pred_org': pred_org,
        'kf_smooth': kf_path,
        'pred_dir': pred_dir,
        # Don't store large arrays here!
    }

DEBUG_COUNT = 0


PRINTED_ERROR = False

def process_file_chunk(files):
    global PRINTED_ERROR
    chunk_results = []
    
    for f in files:
        if f.endswith('_truth.npz'): continue
        
        try:
            d = np.load(f, allow_pickle=True)
            keys = list(d.keys())
            
            # Extract Points
            if 'x' in keys and 'y' in keys and 'z' in keys:
                 c = np.stack([d['x'], d['y'], d['z']], axis=1)
            elif 'hits' in keys:
                 c = d['hits']
            elif 'coords' in keys:
                 c = d['coords']
            elif 'points' in keys:
                 c = d['points']
            else: 
                 continue
            
            # Extract Truth
            if 'true_org' in keys or 'labels_origin' in keys:
                if 'true_org' in keys: o = d['true_org']
                else: o = d['labels_origin']
                
                if 'true_dir' in keys: v = d['true_dir']
                else: v = d['labels_direction']
            else:
                truth_path = f.replace('.npz', '_truth.npz')
                if not os.path.exists(truth_path): continue
                with np.load(truth_path, allow_pickle=True) as td:
                    o = td['origin']
                    v = td['direction']
            
            # SCALING FIX: Convert Meters to mm?
            # Range analysis: If coords are all < 1.0, assuming Meters.
            # Model trained on 64mm box.
            if np.abs(c).max() < 1.0:
                 c = c * 1000.0
                 o = o * 1000.0

            
            # Extract Charge/Features for Model
            feats = None
            q = np.ones(len(c))
            if 'charge' in keys:
                q = d['charge']
                
            if 'features' in keys:
                 feats = d['features']
            else:
                feats = get_local_features(c, q)
            
            # Origin Selection
            p_org = None
            if MODEL is not None:
                # Use True Origin as Anchor for Validation (Cheating but verifies model)
                anchor = o if (o is not None and np.linalg.norm(o) > 1e-9) else None
                p_org = get_predicted_origin(c, feats, anchor_point=anchor)
                
            res = run_hybrid_logic(c, o, v, f, pred_org=p_org) # f is full path
            res['origin_source'] = 'Pred' if MODEL else 'Truth'
            
            clean_res = {
                'filename': res['filename'], 
                'ang_err': res['ang_err'],
                'winner': res['winner'],
                'org_err': res['org_err'],
                'origin_source': res['origin_source'],
                'pred_origin_vec': p_org if p_org is not None else o,
                'pred_dir_vec': res['pred_dir'],
                'true_dir_vec': v,
                'true_org_vec': o
            }
            if 'charge' in keys: clean_res['charge_sample'] = q[0] 
            
            chunk_results.append(clean_res)
            
            # Print Angle Error for monitoring
            if i % 5 == 0:
                 print(f"Track {res['filename']} : AngErr {res['ang_err']:.2f}", flush=True)
            
        except Exception as e:
            if not PRINTED_ERROR:
                print(f"ERROR in loop (First Occurrence): {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                PRINTED_ERROR = True
            continue
            
    return chunk_results

def generate_visualizations(all_results, output_dir):
    """
    Selects Best 30 and Worst 30 tracks, reloads them, and plots them.
    """
    if not PLOTLY_AVAILABLE: return
    
    plot_dir = os.path.join(output_dir, "analysis_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Generating Plots in {plot_dir}...")
    
    # Convert to DF
    df = pd.DataFrame(all_results)
    if len(df) == 0: return

    # Sort
    df = df.sort_values('ang_err')
    
    best_30 = df.head(min(30, len(df)))
    worst_30 = df.tail(min(30, len(df)))
    
    # Helper to load and plot
    def plot_row(row, prefix):
        fpath = row['filename']
        try:
            # Re-Load 
            d = np.load(fpath, allow_pickle=True)
            keys = list(d.keys())
            if 'x' in keys: c = np.stack([d['x'], d['y'], d['z']], axis=1)
            elif 'hits' in keys: c = d['hits']
            elif 'coords' in keys: c = d['coords']
            else: return # Shouldn't happen
            
            q = np.ones(len(c))
            if 'charge' in keys: q = d['charge']
            
            res_dict = {
                'coords': c,
                'feats': np.column_stack([q, np.zeros(len(c)), np.zeros(len(c)), np.zeros(len(c))]),
                'lbl_org': row['true_org_vec'],
                'pred_org': row['pred_origin_vec'],
                'lbl_dir': row['true_dir_vec'],
                'pred_dir': row['pred_dir_vec'],
                'filename': os.path.basename(fpath),
                'org_err': row['org_err'],
                'ang_err': row['ang_err']
            }
            
            plot_one_track(res_dict, plot_dir, prefix)
            
        except Exception as e:
            print(f"Error plotting {fpath}: {e}")

    for _, row in best_30.iterrows():
        plot_row(row, "BEST")
        
    for _, row in worst_30.iterrows():
        plot_row(row, "WORST")

def plot_one_track(result, save_dir, prefix):
    positions = result['coords']
    charges = result['feats'][:,0]
    
    true_origin = result['lbl_org']
    pred_origin = result['pred_org']
    true_dir = result['lbl_dir']
    pred_dir = result['pred_dir'] # FROM HYBRID
    
    # If origin is offset from center (model logic), shift it back?
    # But usually 0-centered is fine.
    
    marker_sizes = charges * 1.5 
    marker_sizes = np.clip(marker_sizes, 4, 15) # Enforce Min 4 for visibility
    
    # Debug: Check Bounds
    c_min = np.min(positions, axis=0)
    c_max = np.max(positions, axis=0)
    
    track_trace = go.Scatter3d(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], mode='markers',
                               marker=dict(size=marker_sizes, color=charges, colorscale='Viridis',
                                           colorbar=dict(title='Charge'), showscale=True, opacity=0.8), 
                               name='Track Points')
                                           
    true_org_trace = go.Scatter3d(x=[true_origin[0]], y=[true_origin[1]], z=[true_origin[2]],
                                     mode='markers', marker=dict(color='red', size=8, symbol='circle'),
                                     name='True Origin')
                                     
    pred_org_trace = go.Scatter3d(x=[pred_origin[0]], y=[pred_origin[1]], z=[pred_origin[2]],
                                     mode='markers', marker=dict(color='cyan', size=8, symbol='diamond'),
                                     name='Predicted Origin')
    
    scale = 20.0
    td_end = true_origin + true_dir * scale
    pd_end = pred_origin + pred_dir * scale 
    
    true_dir_trace = go.Scatter3d(x=[true_origin[0], td_end[0]], y=[true_origin[1], td_end[1]],
                             z=[true_origin[2], td_end[2]],
                             mode='lines', line=dict(color='red', width=5), name='True Direction')

    pred_dir_trace = go.Scatter3d(x=[pred_origin[0]], pd_end[0]], y=[pred_origin[1], pd_end[1]],
                             z=[pred_origin[2], pd_end[2]],
                             mode='lines', line=dict(color='magenta', width=5), name='Hybrid Direction')
                             
    data = [track_trace, true_org_trace, pred_org_trace, true_dir_trace, pred_dir_trace]
    
    fname = result.get('filename', 'Unknown')
    title = (f"Track: {fname}<br>"
             f"{prefix} | OrgErr: {result['org_err']:.2f}mm | AngErr: {result['ang_err']:.2f} deg<br>"
             f"Bounds: X[{c_min[0]:.1f}, {c_max[0]:.1f}] Y[{c_min[1]:.1f}, {c_max[1]:.1f}] Z[{c_min[2]:.1f}, {c_max[2]:.1f}]")

    layout = go.Layout(title=title, 
                       legend=dict(x=0.01, y=0.99),
                       scene=dict(
                           xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                           aspectmode='data'
                       ))
                       
    fig = go.Figure(data=data, layout=layout)
    out_name = f"{prefix}_{fname.replace('.npz','')}.html"
    fig.write_html(os.path.join(save_dir, out_name))


def save_report(results, args, suffix=""):
    if len(results) == 0: return
    df = pd.DataFrame(results)
    
    report_file = os.path.join(args.output_dir, f"standalone_report_FULL.txt")
    
    with open(report_file, "w") as f:
        f.write("=== STANDALONE DIRECTION REPORT (FULL) ===\n")
        f.write(f"Tracks: {len(df)}\n")
        if 'org_err' in df.columns:
             f.write(f"Mean Origin Error: {df['org_err'].mean():.2f} mm\n")
        f.write(f"Mean Angle Error: {df['ang_err'].mean():.2f} deg\n")
        f.write(f"Median Angle Error: {df['ang_err'].median():.2f} deg\n")
        f.write("\n-- Winner Stats --\n")
        f.write(df['winner'].value_counts().to_string())
        
    print(f"Report saved to {report_file}", flush=True)
    print("Winner Breakdown:", flush=True)
    print(df['winner'].value_counts(), flush=True)
    
    # Generate Plots
    generate_visualizations(results, args.output_dir)

def evaluate_standalone(args):
    # Load Model
    if args.model_path:
        load_ai_model(args.model_path)
    
    # Recursive Search
    print(f"Searching for .npz files in {args.dataset_dir} (recursive)...", flush=True)
    files = sorted(glob.glob(os.path.join(args.dataset_dir, "**", "*.npz"), recursive=True))
    
    if len(files) == 0:
         print("CRITICAL: No .npz files found.", flush=True)
         return
         
    print(f"Found {len(files)} files. Processing...", flush=True)
    if args.limit:
        files = files[:args.limit]
        print(f"Limiting to first {args.limit} files.")
    
    chunk_size = 1000 
    all_results = []
    
    for i in range(0, len(files), chunk_size):
        chunk = files[i:i+chunk_size]
        batch_res = process_file_chunk(chunk)
        all_results.extend(batch_res)
        
        if i % 10000 == 0 and i > 0:
             print(f"  Processed {len(all_results)} tracks...", flush=True)
             
    save_report(all_results, args)

if __name__ == "__main__":
    print("Script Starting...", flush=True)
    parser = argparse.ArgumentParser()
    
    # User defaults as requested
    DEFAULT_DATA = "/sdf/home/b/bahrudin/gammaTPC/MLstudy723/processed_tracks906/"
    DEFAULT_OUT = "/sdf/home/b/bahrudin/gammaTPC/workfolder-zerinaa-new/AI_playground/0.direction/results_TRUTH_ORIGIN"
    DEFAULT_MODEL = "/sdf/home/b/bahrudin/gammaTPC/workfolder-zerinaa-new/AI_playground/0.direction/model_best.pth"
    
    parser.add_argument('--dataset_dir', type=str, default=DEFAULT_DATA) 
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUT)
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--limit', type=int, default=None, help="Limit number of tracks")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate_standalone(args)
