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

from train_unified import UnifiedSparseUNet, GammaDataset, sparse_collate_fn, SparseConvTensor
from kalman_tracking import TrackKalmanFilter

# Thresholds from analytical method
GOOD_ORIGIN_THRESH_MM = 1.6
EXCEP_BAD_ANGLE_THRESH_DEG = 100.0

def plot_track_3d_plotly(result, save_dir, prefix):
    if not PLOTLY_AVAILABLE: return
    
    # result keys: coords, feats, lbl_org, lbl_dir, pred_org, pred_dir, org_err, ang_err, filename...
    
    positions = result['coords'] # (N, 3)
    charges = result['feats'][:,0] 
    
    true_org_offset = result['lbl_org']
    pred_org_offset = result['pred_org']
    
    center = np.array([32,32,32])
    true_origin = center + true_org_offset
    pred_origin = center + pred_org_offset
    
    # Directions
    true_dir = result['lbl_dir']
    pred_dir = result['pred_dir']
    
    # Plot Traces
    # 1. Track Points
    track_trace = go.Scatter3d(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], mode='markers',
                               marker=dict(size=2, color=charges, colorscale='Viridis',
                                           colorbar=dict(title='Charge (Log)'), showscale=True), name='Track Points')
                                           
    # 2. True Origin (Red Circle)
    true_org_trace = go.Scatter3d(x=[true_origin[0]], y=[true_origin[1]], z=[true_origin[2]],
                                     mode='markers', marker=dict(color='red', size=8, symbol='circle'),
                                     name='True Origin')
                                     
    # 3. Predicted Origin (Cyan Diamond)
    pred_org_trace = go.Scatter3d(x=[pred_origin[0]], y=[pred_origin[1]], z=[pred_origin[2]],
                                     mode='markers', marker=dict(color='cyan', size=8, symbol='diamond'),
                                     name='Predicted Origin')
    
    # Direction vectors (scaled)
    scale = 10.0
    
    # 4. True Direction (Red Line)
    td_end = true_origin + true_dir * scale
    true_dir_trace = go.Scatter3d(x=[true_origin[0], td_end[0]], y=[true_origin[1], td_end[1]],
                             z=[true_origin[2], td_end[2]],
                             mode='lines', line=dict(color='red', width=5), name='True Direction')

    # 5. Pred Direction (Magenta Line)
    pd_end = pred_origin + pred_dir * scale
    pred_dir_trace = go.Scatter3d(x=[pred_origin[0], pd_end[0]], y=[pred_origin[1], pd_end[1]],
                             z=[pred_origin[2], pd_end[2]],
                             mode='lines', line=dict(color='magenta', width=5), name='Predicted Direction')
                             
    # Reconstruction Core (Green) if available (optional, but requested previously? User liked the clean look, maybe omit recon points to match 'clean' request unless debugging)
    # The user asked for "plots like analysis_toolkit", which did NOT show recon points, just track + vectors.
    # But KF relies on Recon points. Maybe show them faintly?
    # Let's stick to the "Clean" request: Track, Origins, Vectors.
    
    data = [track_trace, true_org_trace, pred_org_trace, true_dir_trace, pred_dir_trace]
    
    # Add KF Smoothed Path if available?
    if 'kf_smooth' in result:
        kf_path = result['kf_smooth']
        # kf_path is roughly in absolute coords if we did it right in fit()
        # In evaluate loop, we passed 'core_sorted' which is absolute.
        pass # Skip for now to keep it clean like reference

    # Title with ID
    fname = result.get('filename', 'Unknown')
    title = (f"Track: {fname}<br>"
             f"{prefix} | Category: {result['category']} | OrgErr: {result['org_err']:.2f}mm | AngErr: {result['ang_err']:.2f} deg")

    layout = go.Layout(title=title, 
                       legend=dict(x=0.01, y=0.99),
                       scene=dict(
                           xaxis_title='X (voxel)', yaxis_title='Y (voxel)', zaxis_title='Z (voxel)',
                           aspectmode='data',
                           camera_eye=dict(x=1.2, y=1.2, z=0.6)
                       ))
                       
    fig = go.Figure(data=data, layout=layout)
    
    out_name = f"{prefix}_{fname.replace('.npz','')}.html"
    fig.write_html(os.path.join(save_dir, out_name))

def plot_track_3d_matplotlib(result, save_dir, prefix):
    # Fallback for when Plotly is missing
    positions = result['coords']
    true_org_offset = result['lbl_org']
    pred_org_offset = result['pred_org']
    
    center = np.array([32,32,32])
    true_origin = center + true_org_offset
    pred_origin = center + pred_org_offset
    
    true_dir = result['lbl_dir']
    pred_dir = result['pred_dir']

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Track
    ax.scatter(positions[:,0], positions[:,1], positions[:,2], c='blue', s=1, alpha=0.3, label='Track')
    
    # Origin
    ax.scatter(true_origin[0], true_origin[1], true_origin[2], c='red', s=100, marker='o', label='True Org')
    ax.scatter(pred_origin[0], pred_origin[1], pred_origin[2], c='cyan', s=100, marker='D', label='Pred Org')
    
    # Direction
    scale = 10.0
    ax.plot([true_origin[0], true_origin[0]+true_dir[0]*scale],
            [true_origin[1], true_origin[1]+true_dir[1]*scale],
            [true_origin[2], true_origin[2]+true_dir[2]*scale], c='red', linewidth=3, label='True Dir')
            
    ax.plot([pred_origin[0], pred_origin[0]+pred_dir[0]*scale],
            [pred_origin[1], pred_origin[1]+pred_dir[1]*scale],
            [pred_origin[2], pred_origin[2]+pred_dir[2]*scale], c='cyan', linewidth=3, label='Pred Dir')
            
    ax.set_title(f"{prefix}\nOrgErr: {result['org_err']:.2f}mm | AngErr: {result['ang_err']:.1f} deg")
    ax.legend()
    
    fname = f"{prefix.replace(' ', '_')}_{int(time.time()*1000)}.png"
    plt.savefig(os.path.join(save_dir, fname))
    plt.close()

def plot_track_3d_matplotlib_recon(result, save_dir, prefix):
    # Specialized plot showing INPUT vs RECONSTRUCTION
    # result['coords'] is input active sites
    # result['pred_recon'] is binary mask (logits or sigmoid)
    
    pos = result['coords']
    recon_logits = result['pred_recon'] # (N, 1)
    
    # Sigmoid to get probability
    probs = 1.0 / (1.0 + np.exp(-1.0 * recon_logits))
    
    # Filter "Core" voxels
    mask_core = (probs > 0.5).flatten()
    
    core_pos = pos[mask_core]
    noise_pos = pos[~mask_core]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Noise/Fuzz (Red, small, transparent)
    if len(noise_pos) > 0:
        ax.scatter(noise_pos[:,0], noise_pos[:,1], noise_pos[:,2], c='red', s=1, alpha=0.1, label='Diffuse/Noise')
        
    # 2. Core (Green, larger, solid)
    if len(core_pos) > 0:
        ax.scatter(core_pos[:,0], core_pos[:,1], core_pos[:,2], c='green', s=5, alpha=0.8, label='Recon Core')
        
    # True Direction (for context)
    true_org_offset = result['lbl_org']
    true_dir = result['lbl_dir']
    center = np.array([32,32,32])
    true_org = center + true_org_offset
    scale = 20.0
    ax.plot([true_org[0], true_org[0]+true_dir[0]*scale],
            [true_org[1], true_org[1]+true_dir[1]*scale],
            [true_org[2], true_org[2]+true_dir[2]*scale], c='black', linewidth=3, label='True Dir')
            
    ax.set_title(f"{prefix} - Reconstruction\nGreen=Model thinks is Core")
    ax.legend()
    
    fname = f"{prefix}_Recon.png"
    plt.savefig(os.path.join(save_dir, fname))
    plt.close()

def evaluate(args):
    # Flush stdout to see prints immediately in SLURM logs
    sys.stdout.flush()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")
    sys.stdout.flush()

    dataset_path = os.path.join(args.dataset_dir, "dataset_test.npz")
    print(f"Loading dataset from: {dataset_path}")
    sys.stdout.flush()
    
    if not os.path.exists(dataset_path):
        print(f"CRITICAL ERROR: Dataset file not found at {dataset_path}")
        return # Exit cleanly to show error
        
    try:
        dataset = GammaDataset(dataset_path)
        print(f"Dataset loaded. {len(dataset)} tracks.")
    except Exception as e:
        print(f"CRITICAL ERROR loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=sparse_collate_fn, shuffle=False)
    
    # 2. Load Model
    model = UnifiedSparseUNet(in_channels=4, spatial_size=[64,64,64]).to(device)
    
    # Logic to find model
    if args.model_path and os.path.exists(args.model_path):
        load_path = args.model_path
        print(f"Loading model from specified path: {load_path}")
        model.load_state_dict(torch.load(load_path, map_location=device))
    else:
        # Fallbacks
        model_path_best = os.path.join(args.dataset_dir, "model_best.pth")
        model_path_cwd = "model_best.pth" # Check current dir
        
        if os.path.exists(model_path_cwd):
            load_path = model_path_cwd
            print(f"Loaded model from CWD: {load_path}")
            model.load_state_dict(torch.load(load_path, map_location=device))
        elif os.path.exists(model_path_best):
            load_path = model_path_best
            print(f"Loaded model from Dataset Dir: {load_path}")
            model.load_state_dict(torch.load(load_path, map_location=device))
        else:
            print(f"Warning: Model not found (checked {model_path_cwd} and {model_path_best}). Using random weights.")

    model.eval()
    
    results = []
    
    criterion_cos = nn.CosineSimilarity()
    
    print("Running Inference...")
    with torch.no_grad():
    print("Running Inference...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if len(batch) == 5:
                coords, feats, lbl_org, lbl_dir, fnames = batch
            else:
                coords, feats, lbl_org, lbl_dir = batch
                fnames = [f"Track_{b}" for b in range(len(lbl_org))]
            
            coords = coords.to(device)
            feats = feats.to(device)
            lbl_org = lbl_org.to(device)
            lbl_dir = lbl_dir.to(device)
            
            bs = lbl_org.shape[0]
            
            # Forward
            input_st = SparseConvTensor(feats, coords, spatial_shape=[64,64,64], batch_size=bs)
            pred_org, pred_dir, out_recon = model(input_st)
            
            # Unpack SparseTensor for plotting logic
            # We need to map back to batch elements. 
            # sparse_collate_fn creates batch_coords with column 0 as batch_idx
            
            # Reconstruction Mask (Sparse)
            # Map sparse features back to the points
            # out_recon is SparseConvTensor. If SubM, indices match input 'coords'.
            # We can just extract features directly since Input indices == Output indices for SubM.
            recon_logits_all = out_recon.features.cpu().numpy()
            # We need to map back to batch elements. 
            # sparse_collate_fn creates batch_coords with column 0 as batch_idx
            
            # Iterate batch
            coords_b = coords # Original list of arrays before collate? No, DataLoader output is collated tensor.
            # Actually sparse_collate_fn output:
            # coords: (N_total, 4) [b, x, y, z]
            # feats: (N_total, C)
            
            # We need to split them back for plotting "per track"
            
            coords_np_all = coords.cpu().numpy()
            feats_np_all = feats.cpu().numpy()
            
            # Match lengths (in case of empty tracks skipped in batch logic, though collate handles alignment)
            # Assuming aligned output for now
            
            # Calculate Errors per track
            # Origin Error: Euclidean distance
            # pred_org is offset from centered cloud. lbl_org is offset from centered cloud.
            # So error is just norm of difference.
            diff_org = pred_org - lbl_org
            err_org_mm = torch.norm(diff_org, dim=1).cpu().numpy() # already in mm if units are consistent
            
            # Prepare for per-track loop
            # Group by batch index
            batch_mask_dict = {}
            for b in range(bs):
                batch_mask_dict[b] = (coords_np_all[:, 0] == b)
            
            # Angle Error: ArcCos(DotProduct)
            cos_sim = criterion_cos(pred_dir, lbl_dir)
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
            err_ang_deg = torch.acos(cos_sim).cpu().numpy() * 180.0 / np.pi
            
            # Collect results
            for b in range(bs):
                err_o = err_org_mm[b]
                err_a = err_ang_deg[b]
                
                # Categorize
                if err_o <= GOOD_ORIGIN_THRESH_MM:
                    cat = 'Good'
                elif err_a > EXCEP_BAD_ANGLE_THRESH_DEG:
                    cat = 'Exceptionally Bad'
                else:
                    cat = 'Bad'
                
                # Extract track data for this batch element
                mask = batch_mask_dict[b]
                if not mask.any(): continue # Skip empty if any
                coords_np = coords_np_all[mask, 1:] # Drop batch idx, keep x,y,z
                feats_np = feats_np_all[mask]

                results.append({
                    'org_err': err_o,
                    'ang_err': err_a,
                    'category': cat,
                    # Store data for plotting (convert to numpy)
                    'coords': coords_np,
                    'feats': feats_np,
                    'lbl_org': lbl_org[b].cpu().numpy(),
                    'pred_org': pred_org[b].cpu().numpy(),
                    'pred_dir': pred_dir[b].cpu().numpy(), # Fallback if KF fails
                    'pred_recon': recon_logits_all[mask], # Extract corresponding logits
                    'filename': fnames[b] # Store filename
                })
                
                # --- KALMAN FILTER REFINEMENT ---
                # 1. Get Core Points (Mask > 0.5)
                # sigmoid
                logits = results[-1]['pred_recon'].flatten()
                probs = 1.0 / (1.0 + np.exp(-logits))
                mask_core = probs > 0.5
                
                core_points = results[-1]['coords'][mask_core]
                
                
                # If we have enough points, run KF
                if len(core_points) > 5:
                    # Sort points by distance from Predicted Origin (to simulate time evolution)
                    p_org = results[-1]['pred_org'] + np.array([32,32,32]) # Abs position
                    dists = np.linalg.norm(core_points - p_org, axis=1)
                    sorted_idx = np.argsort(dists)
                    core_sorted = core_points[sorted_idx]
                    
                    kf = TrackKalmanFilter(dt=1.0)
                    kf_dir, kf_smooth = kf.fit(core_sorted)
                    
                    if i < 5: # DEBUG PRINTS
                        print(f"DEBUG Track {i}: CorePts={len(core_points)}, KF_Dir={kf_dir}, True_Dir={lbl_dir[b].cpu().numpy()}")
                    
                    # Overwrite Direction Prediction with KF result
                    results[-1]['pred_dir'] = kf_dir
                    results[-1]['kf_smooth'] = kf_smooth # Store for plotting?
                    
                    # --- RECALCULATE ANGLE ERROR ---
                    # Now that we have a better pred_dir, update the metric
                    lbl_d = results[-1]['lbl_dir']
                    pred_d = kf_dir
                    
                    # Cos Sim
                    cos_sim = np.dot(lbl_d, pred_d) / (np.linalg.norm(lbl_d)*np.linalg.norm(pred_d) + 1e-9)
                    cos_sim = np.clip(cos_sim, -1.0, 1.0)
                    new_err = np.arccos(cos_sim) * 180.0 / np.pi
                    
                    results[-1]['ang_err'] = new_err
                    
                else:
                    if i < 5: print(f"DEBUG Track {i}: Not enough core points ({len(core_points)}). Fallback.")
                    pass
                
    df = pd.DataFrame(results)
    
    # --- Plotting ---
    if len(df) > 0:
        print("Generating diagnostic plots...")
        plot_dir = os.path.join(args.output_dir, "diagnostic_plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # 1. Worst 30 Origins
        worst_org = sorted(results, key=lambda x: x['org_err'], reverse=True)[:30]
        # 2. Worst 30 Angles
        worst_ang = sorted(results, key=lambda x: x['ang_err'], reverse=True)[:30]
        # 3. Best 30 Angles (lowest error)
        best_ang = sorted(results, key=lambda x: x['ang_err'], reverse=False)[:30]
        
        # Helper to plot list
        def plot_list(track_list, subfolder_name, prefix_base):
            sub_dir = os.path.join(plot_dir, subfolder_name)
            os.makedirs(sub_dir, exist_ok=True)
            for i, res in enumerate(track_list):
                if PLOTLY_AVAILABLE:
                    plot_track_3d_plotly(res, sub_dir, f"{prefix_base}_{i+1}")
                else:
                    plot_track_3d_matplotlib(res, sub_dir, f"{prefix_base}_{i+1}")
                    
        plot_list(worst_org, "Worst_Origin", "WorstOrg")
        plot_list(worst_ang, "Worst_Angle", "WorstAng")
        plot_list(best_ang, "Best_Angle", "BestAng")

        print(f"Saved Top 30 Best/Worst plots to {plot_dir}")
    
    # Report
    report_file = os.path.join(args.output_dir, "evaluation_report.txt")
    with open(report_file, "w") as f:
        def log(s):
            print(s)
            f.write(s + "\n")
            
        log("=" * 50)
        log(f"AI MODEL PERFORMANCE SUMMARY")
        log("=" * 50)
        log(f"Analyzed {len(df)} tracks.")
        
        if len(df) > 0:
            log("\n--- Origin Error (mm) ---")
            log(f"  Mean:   {df['org_err'].mean():.2f} mm")
            log(f"  Median: {df['org_err'].median():.2f} mm")
            log(f"  Std Dev: {df['org_err'].std():.2f} mm")
            
            log("\n--- Direction Error (deg) ---")
            log(f"  Mean:   {df['ang_err'].mean():.2f} deg")
            log(f"  Median: {df['ang_err'].median():.2f} deg")
            log(f"  Std Dev: {df['ang_err'].std():.2f} deg")
            
            # Categories
            categories = ['Good', 'Bad', 'Exceptionally Bad']
            log("\n--- CATEGORIES ---")
            for cat in categories:
                sub = df[df['category'] == cat]
                pct = len(sub) / len(df) * 100
                log(f"{cat}: {len(sub)} tracks ({pct:.1f}%)")
        
        log("=" * 50)
    
    # Plots
    if len(df) > 0:
        # Error Distributions (Histograms)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.histplot(df['org_err'], bins=30, kde=True, color='blue')
        plt.title('Origin Error Distribution')
        plt.xlabel('Error (mm)')
        
        plt.subplot(1, 2, 2)
        sns.histplot(df['ang_err'], bins=30, kde=True, color='red')
        plt.title('Angle Error Distribution')
        plt.xlabel('Error (degrees)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "error_distribution.png"))
        plt.close()
        
        # 2. Scatter Plot (Angle vs Origin) to identify correlation
        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=df, x='org_err', y='ang_err', hue='category', palette='viridis')
        plt.title('Angle Error vs Origin Error')
        plt.savefig(os.path.join(args.output_dir, "error_correlation.png"))
        plt.close()
        
    # Save CSV
    csv_path = os.path.join(args.output_dir, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Analysis complete. Report saved to {report_file}")

if __name__ == "__main__":
    try:
        print("DEBUG: Parsing arguments...")
        sys.stdout.flush()
        
        # Termius Defaults
        DEFAULT_DIR = "/sdf/home/b/bahrudin/gammaTPC/workfolder-zerinaa-new/AI_playground"
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_dir', type=str, default=DEFAULT_DIR)
        parser.add_argument('--model_path', type=str, default=None, help="Path to model file")
        parser.add_argument('--output_dir', type=str, default=os.path.join(DEFAULT_DIR, 'results'))
        parser.add_argument('--batch_size', type=int, default=4)
        args, unknown = parser.parse_known_args()
        
        if unknown:
            print(f"DEBUG: Unknown arguments ignored: {unknown}")
        
        print(f"DEBUG: Args parsed. Output dir: {args.output_dir}")
        sys.stdout.flush()
        
        os.makedirs(args.output_dir, exist_ok=True)
        evaluate(args)
        
    except Exception as e:
        print("\nCRITICAL SCRIPT FAILURE:")
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
