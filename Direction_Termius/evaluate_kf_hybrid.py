import sys
import traceback
print("DEBUG: Script Start")
sys.stdout.flush()

import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

print("DEBUG: Core imports done")
sys.stdout.flush()

import matplotlib
matplotlib.use('Agg') # Force headless
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

print("DEBUG: Plotting imports done")
sys.stdout.flush()

# Try Plotly, else fallback
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("WARNING: Plotly not installed. Will fallback to Matplotlib (static images) for 3D plots.")

from train_unified_kf import UnifiedSparseUNet, GammaDataset, sparse_collate_fn, SparseConvTensor
from kalman_tracking import TrackKalmanFilter

# Thresholds from analytical method
GOOD_ORIGIN_THRESH_MM = 1.6
EXCEP_BAD_ANGLE_THRESH_DEG = 100.0

# --- HYBRID DIRECTION LOGIC (RUN 9) ---

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
        if count > len(coords) + 1: break # Safety
        
    return np.array(sorted_coords)

def select_best_direction_density(kf_dir, sorted_coords, origin, angle_deg=30, kf_bias=1.0):
    """
    Hybrid Biased Voting Strategy (Run 9).
    """
    # 1. Generate Candidates
    candidates = {}
    
    # KF
    candidates['KF'] = kf_dir
    candidates['KF_Neg'] = -kf_dir
    
    # Secant (Safety)
    k = min(20, len(sorted_coords)-1)
    vec_sec = sorted_coords[k] - origin
    norm_sec = np.linalg.norm(vec_sec)
    if norm_sec > 1e-9:
        candidates['Secant'] = vec_sec / norm_sec
    else:
        candidates['Secant'] = kf_dir # Fallback
        
    # Local PCA (Bisector/Chord)
    n_pca = min(20, len(sorted_coords))
    local_pts = sorted_coords[:n_pca]
    if len(local_pts) > 2:
        mean_l = np.mean(local_pts, axis=0)
        centered_l = local_pts - mean_l
        u, s, vh = np.linalg.svd(centered_l, full_matrices=False)
        pca_vec = vh[0]
        candidates['LocalPCA'] = pca_vec
        candidates['LocalPCA_Neg'] = -pca_vec

    # Global PCA (Robust Baseline)
    if len(sorted_coords) > 2:
        mean_g = np.mean(sorted_coords, axis=0)
        centered_g = sorted_coords - mean_g
        u, s, vh = np.linalg.svd(centered_g, full_matrices=False)
        g_vec = vh[0]
        candidates['GlobalPCA'] = g_vec
        candidates['GlobalPCA_Neg'] = -g_vec
        
    # 2. Score Candidates (Cone Density)
    cos_thresh = np.cos(np.radians(angle_deg))
    
    n_check = min(20, len(sorted_coords))
    check_points = sorted_coords[:n_check]
    
    vecs_to_pts = check_points - origin
    dists = np.linalg.norm(vecs_to_pts, axis=1)
    valid = dists > 1e-9
    vecs_to_pts = vecs_to_pts[valid]
    dists = dists[valid]
    vecs_to_pts = vecs_to_pts / dists[:, None]
    
    best_name = 'Secant' # Default if nothing beats it
    best_score = -999.0
    best_vec = candidates.get('Secant', np.array([0,0,1]))
    
    for name, vec in candidates.items():
        dots = np.dot(vecs_to_pts, vec)
        score = float(np.sum(dots > cos_thresh))
        
        if 'KF' in name:
            # ONLY apply bias if KF has at least some support
            if score > 0:
                score += kf_bias
        
        if score > best_score:
            best_score = score
            best_name = name
            best_vec = vec
            
    if best_score <= 0 and 'Secant' in candidates:
        return candidates['Secant'], 'Secant_Fallback'
        
    return best_vec, best_name

# --- PLOTTING ---

def plot_track_3d_plotly(result, save_dir, prefix):
    if not PLOTLY_AVAILABLE: return
    
    positions = result['coords'] 
    charges = result['feats'][:,0] 
    
    true_org_offset = result['lbl_org']
    pred_org_offset = result['pred_org']
    
    center = np.array([32,32,32])
    true_origin = center + true_org_offset
    pred_origin = center + pred_org_offset
    
    true_dir = result['lbl_dir']
    pred_dir = result['pred_dir']
    
    marker_sizes = charges * 1.5 
    marker_sizes = np.clip(marker_sizes, 2, 10)
    
    track_trace = go.Scatter3d(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], mode='markers',
                               marker=dict(size=marker_sizes, color=charges, colorscale='Viridis',
                                           colorbar=dict(title='Charge (Log)'), showscale=True, opacity=0.8), 
                               name='Track Points')
                                           
    true_org_trace = go.Scatter3d(x=[true_origin[0]], y=[true_origin[1]], z=[true_origin[2]],
                                     mode='markers', marker=dict(color='red', size=8, symbol='circle'),
                                     name='True Origin')
                                     
    pred_org_trace = go.Scatter3d(x=[pred_origin[0]], y=[pred_origin[1]], z=[pred_origin[2]],
                                     mode='markers', marker=dict(color='cyan', size=8, symbol='diamond'),
                                     name='Predicted Origin')
    
    scale = 20.0
    
    td_end = true_origin + true_dir * scale
    true_dir_trace = go.Scatter3d(x=[true_origin[0], td_end[0]], y=[true_origin[1], td_end[1]],
                             z=[true_origin[2], td_end[2]],
                             mode='lines', line=dict(color='red', width=5), name='True Direction')

    pd_end = pred_origin + pred_dir * scale
    pred_dir_trace = go.Scatter3d(x=[pred_origin[0], pd_end[0]], y=[pred_origin[1], pd_end[1]],
                             z=[pred_origin[2], pd_end[2]],
                             mode='lines', line=dict(color='magenta', width=5), name='Hybrid Direction')
                             
    kf_path = result.get('kf_smooth', None)
    if kf_path is not None:
         kf_trace = go.Scatter3d(x=kf_path[:,0], y=kf_path[:,1], z=kf_path[:,2],
                                mode='lines', line=dict(color='cyan', width=3), name='KF Path')
         data = [track_trace, true_org_trace, pred_org_trace, true_dir_trace, pred_dir_trace, kf_trace]
    else:
         data = [track_trace, true_org_trace, pred_org_trace, true_dir_trace, pred_dir_trace]
    
    fname = result.get('filename', 'Unknown')
    title = (f"Track: {fname}<br>"
             f"{prefix} | OrgErr: {result['org_err']:.2f}mm | AngErr: {result['ang_err']:.2f} deg")

    layout = go.Layout(title=title, 
                       legend=dict(x=0.01, y=0.99),
                       scene=dict(
                           xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                           aspectmode='data',
                           camera_eye=dict(x=1.2, y=1.2, z=0.6)
                       ))
                       
    fig = go.Figure(data=data, layout=layout)
    out_name = f"{prefix}_{fname.replace('.npz','')}.html"
    fig.write_html(os.path.join(save_dir, out_name))

def evaluate(args):
    sys.stdout.flush()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")
    
    dataset_path = os.path.join(args.dataset_dir, "dataset_test.npz")
    print(f"Loading dataset from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        # Fallback: Check parent directory
        parent_dataset = os.path.join(args.dataset_dir, "..", "dataset_test.npz")
        if os.path.exists(parent_dataset):
            print(f"Dataset not found in specified dir, but found in parent: {parent_dataset}")
            dataset_path = parent_dataset
        else:
            print(f"CRITICAL ERROR: Dataset file not found at {dataset_path} or {parent_dataset}")
            return
        
    try:
        dataset = GammaDataset(dataset_path)
        print(f"Dataset loaded. {len(dataset)} tracks.")
    except Exception as e:
        print(f"CRITICAL ERROR loading dataset: {e}")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=sparse_collate_fn, shuffle=False)
    
    model = UnifiedSparseUNet(in_channels=4, spatial_size=[64,64,64]).to(device)
    
    if args.model_path and os.path.exists(args.model_path):
        load_path = args.model_path
        print(f"Loading model from specified path: {load_path}")
        model.load_state_dict(torch.load(load_path, map_location=device))
    else:
        # Fallbacks
        # 1. Check CWD (Priority: we are likely inside the model folder)
        model_path_cwd = "model_best.pth"
        model_path_cwd_org = "model_best_origin.pth"
        
        if os.path.exists(model_path_cwd):
            load_path = model_path_cwd
            print(f"Loaded model from CWD: {load_path}")
            model.load_state_dict(torch.load(load_path, map_location=device))
        elif os.path.exists(model_path_cwd_org):
            load_path = model_path_cwd_org
            print(f"Loaded model from CWD: {load_path}")
            model.load_state_dict(torch.load(load_path, map_location=device))
        else:
            # 2. Check Dataset Dir
            model_path_best = os.path.join(args.dataset_dir, "model_best.pth")
            if os.path.exists(model_path_best):
                load_path = model_path_best
                print(f"Loaded model from Dataset Dir: {load_path}")
                model.load_state_dict(torch.load(load_path, map_location=device))
            else:
                print(f"CRITICAL ERROR: Model not found in CWD or Dataset Dir.")
                sys.exit(1)

    model.eval()
    results = []
    criterion_cos = nn.CosineSimilarity()
    
    print("Running Inference with Hybrid Direction Logic...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if len(batch) == 5:
                # ... existing unpacking ...
                coords, feats, lbl_org, lbl_dir, fnames = batch
            else:
                coords, feats, lbl_org, lbl_dir = batch
                fnames = [f"Track_{b}" for b in range(len(lbl_org))]
            
            coords = coords.to(device)
            feats = feats.to(device)
            lbl_org = lbl_org.to(device)
            lbl_dir = lbl_dir.to(device)
            
            bs = lbl_org.shape[0]
            
            input_st = SparseConvTensor(feats, coords, spatial_shape=[64,64,64], batch_size=bs)
            pred_org, pred_dir_dl, out_recon = model(input_st)
            
            # ... unpack ...
            recon_logits_all = out_recon.features.cpu().numpy()
            coords_np_all = coords.cpu().numpy()
            feats_np_all = feats.cpu().numpy()
            
            diff_org = pred_org - lbl_org
            err_org_mm = torch.norm(diff_org, dim=1).cpu().numpy()
            
            batch_mask_dict = {}
            for b in range(bs):
                batch_mask_dict[b] = (coords_np_all[:, 0] == b)
            
            for b in range(bs):
                mask = batch_mask_dict[b]
                if not mask.any(): continue
                
                res = {
                    'org_err': err_org_mm[b],
                    # ...
                    'category': 'Unknown',
                    'coords': coords_np_all[mask, 1:],
                    'feats': feats_np_all[mask],
                    'lbl_org': lbl_org[b].cpu().numpy(),
                    'lbl_dir': lbl_dir[b].cpu().numpy(),
                    'pred_org': pred_org[b].cpu().numpy(),
                    'pred_dir': pred_dir_dl[b].cpu().numpy(), 
                    'pred_recon': recon_logits_all[mask],
                    'filename': fnames[b]
                }
                
                # --- HYBRID REFINEMENT ---
                p_org_abs = res['pred_org'] + np.array([32,32,32])
                logits = res['pred_recon'].flatten()
                probs = 1.0 / (1.0 + np.exp(-logits))
                mask_core = probs > 0.1 
                core_points = res['coords'][mask_core]
                
                # Unify points (remove duplicates to avoid KF zero-velocity)
                if len(core_points) > 0:
                    core_points = np.unique(core_points, axis=0)
                
                # DEBUG PRINTS FOR FIRST FEW TRACKS
                if len(results) < 20:
                    print(f"\n[DEBUG Track {fnames[b]}]")
                    print(f"  Max Recon Prob: {probs.max():.4f}")
                    print(f"  Core Points (>0.1): {len(core_points)} / {len(res['coords'])}")
                    if len(core_points) > 0:
                        std_c = np.std(core_points, axis=0)
                        print(f"  Coords STD: [{std_c[0]:.2f}, {std_c[1]:.2f}, {std_c[2]:.2f}]")
                    print(f"  DL Pred Dir: {res['pred_dir']}")
                    print(f"  DL Org Err:  {res['org_err']:.4f}")
                
                if len(core_points) > 5:
                    try:
                        sorted_pts = sort_points_nn(core_points, p_org_abs)
                        kf = TrackKalmanFilter(dt=1.0)
                        kf_dir, kf_smooth = kf.fit(sorted_pts)
                        
                        if len(results) < 20:
                             print(f"  [DEBUG Debug] Sorted Pts: {len(sorted_pts)}, KF Raw: {kf_dir}")
                        
                        if kf_dir is not None:
                            # Modified for Debug: Inspect winner
                            hybrid_dir, winner_name = select_best_direction_density(
                                kf_dir, sorted_pts, p_org_abs, kf_bias=1.0
                            )
                            
                            res['pred_dir'] = hybrid_dir
                            res['kf_smooth'] = kf_smooth
                            
                            # Recalc Error
                            lbl_d = res['lbl_dir']
                            cos_sim = np.dot(lbl_d, hybrid_dir) / (np.linalg.norm(lbl_d)*np.linalg.norm(hybrid_dir) + 1e-9)
                            res['ang_err'] = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
                            
                            if len(results) < 20: 
                                print(f"  Hybrid Dir: {hybrid_dir} (Error: {res['ang_err']:.2f}) [Winner: {winner_name}]")
                            
                    except Exception as e:
                        print(f"Hybrid Logic Failed for {fnames[b]}: {e}")

                # Fallback Error Calc
                if 'ang_err' not in res:
                     lbl_d = res['lbl_dir']
                     pred_d = res['pred_dir'] # DL
                     cos_sim = np.dot(lbl_d, pred_d) / (np.linalg.norm(lbl_d)*np.linalg.norm(pred_d) + 1e-9)
                     res['ang_err'] = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
                     if len(results) < 20: print(f"  Fallback Dir used. Error: {res['ang_err']:.2f}")
                     
                results.append(res)
    
    # Save loaded path to report
    load_path_str = load_path if 'load_path' in locals() else "Unknown"
    
    df = pd.DataFrame(results)
    
    # ... plotting ...
    if len(df) > 0:
        print("Generating diagnostic plots...")
        plot_dir = os.path.join(args.output_dir, "diagnostic_plots")
        os.makedirs(plot_dir, exist_ok=True)
        # ... existing plot code ...
        df_sorted = df.sort_values('ang_err')
        best_30 = df_sorted.head(30)
        worst_30 = df_sorted.tail(30)
        res_map = {r['filename']: r for r in results}
        def batch_plot(subset_df, subfolder):
            s_dir = os.path.join(plot_dir, subfolder)
            os.makedirs(s_dir, exist_ok=True)
            for _, row in subset_df.iterrows():
                r = res_map[row['filename']]
                plot_track_3d_plotly(r, s_dir, "Hybrid")
        batch_plot(best_30, "Best_Angle")
        batch_plot(worst_30, "Worst_Angle")

    # Define report_file properly
    report_file = os.path.join(args.output_dir, "hybrid_evaluation_report.txt")
    with open(report_file, "w") as f:
        f.write("=== HYBRID EVALUATION REPORT ===\n")
        f.write(f"Model: {load_path_str}\n")
        # ... stats ...
        f.write(f"Tracks: {len(df)}\n")
        f.write(f"Mean Origin Error: {df['org_err'].mean():.2f} mm\n")
        f.write(f"Mean Angle Error: {df['ang_err'].mean():.2f} deg\n")
        f.write(f"Median Angle Error: {df['ang_err'].median():.2f} deg\n")
        
    print(f"Done. Report: {report_file}")

if __name__ == "__main__":
    DEFAULT_DIR = "/sdf/home/b/bahrudin/gammaTPC/workfolder-zerinaa-new/AI_playground"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default=DEFAULT_DIR)
    parser.add_argument('--model_path', type=str, default=None)
    # Default to 'results' folder in current directory (e.g., 1.model_KF/results)
    parser.add_argument('--output_dir', type=str, default='results') 
    parser.add_argument('--batch_size', type=int, default=4)
    args, _ = parser.parse_known_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(args)
