import os
import glob
import re
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import plotly.graph_objects as go

# --- Configuration ---
BASE_DATA_PATH = r'data'
OUTPUT_ANALYSIS_DIR = 'analysis'  # Main folder for all outputs
OUTPUT_FILENAME_REPORT = 'track_categorization_report.txt'
K_NEIGHBORS = 20
MAX_TRACKS_TO_PROCESS = None  # Set to a number for testing, None for all

# --- Plotting & Categorization ---
NUM_GOOD_PLOTS_TO_SAVE = 20
GOOD_ORIGIN_THRESH_M = 0.0016
EXCEP_BAD_ANGLE_THRESH = 100.0


# --- Core Algorithm Functions (Unchanged) ---

def find_track_files(base_path):
    search_pattern = os.path.join(base_path, 'compton', 'E*', 'd*', 'data', '*.npz')
    track_files = [f for f in glob.glob(search_pattern, recursive=True) if not f.endswith('_truth.npz')]
    print(f"Found {len(track_files)} tracks to analyze.")
    return track_files


def load_data(track_path):
    truth_path = track_path.replace('.npz', '_truth.npz')
    if not os.path.exists(truth_path): return None, None, None
    with np.load(track_path) as data:
        positions = np.stack([data['x'], data['y'], data['z']], axis=1)
        charges = data['charge']
    with np.load(truth_path, allow_pickle=True) as data:
        truth = {'origin': data['track_origin'], 'direction': data['direction']}
    return positions, charges, truth


def predict_head_tail_knn_v4(positions, charges, k):
    if len(positions) < k: return None, None
    start_idx = 0
    dists_start = np.linalg.norm(positions - positions[start_idx], axis=1)
    ep1_idx = np.argmax(dists_start)
    dists_ep1 = np.linalg.norm(positions - positions[ep1_idx], axis=1)
    ep2_idx = np.argmax(dists_ep1)
    endpoints = {'A': {'pos': positions[ep1_idx]}, 'B': {'pos': positions[ep2_idx]}}
    knn = NearestNeighbors(n_neighbors=k).fit(positions)
    global_vec_ab = (endpoints['B']['pos'] - endpoints['A']['pos'])
    global_vec_ab /= np.linalg.norm(global_vec_ab) if np.linalg.norm(global_vec_ab) > 0 else 1
    for name, point in endpoints.items():
        indices = knn.kneighbors([point['pos']], return_distance=False)[0]
        local_points = positions[indices]
        point['avg_charge'] = np.mean(charges[indices])
        pca = PCA(n_components=1).fit(local_points)
        point['linearity'] = pca.explained_variance_ratio_[0]
        local_direction = pca.components_[0]
        point['local_direction'] = local_direction
        global_vec = global_vec_ab if name == 'A' else -global_vec_ab
        dot_product = np.clip(np.dot(local_direction, global_vec), -1.0, 1.0)
        angle = np.arccos(dot_product) * 180 / np.pi
        point['curvature'] = min(angle, 180 - angle)
    ep_A, ep_B = endpoints['A'], endpoints['B']
    total_charge = ep_A['avg_charge'] + ep_B['avg_charge']
    norm_charge_A = ep_A['avg_charge'] / total_charge if total_charge > 0 else 0.5
    total_curve = ep_A['curvature'] + ep_B['curvature']
    norm_curve_A = ep_A['curvature'] / total_curve if total_curve > 0 else 0.5
    score_A = (0.5 * ep_A['linearity']) - (1.0 * norm_charge_A) - (1.5 * norm_curve_A)
    score_B = (0.5 * ep_B['linearity']) - (1.0 * (1.0 - norm_charge_A)) - (1.5 * (1.0 - norm_curve_A))
    if score_A > score_B:
        predicted_head, predicted_tail, local_head_direction = ep_A['pos'], ep_B['pos'], ep_A['local_direction']
    else:
        predicted_head, predicted_tail, local_head_direction = ep_B['pos'], ep_A['pos'], ep_B['local_direction']
    global_direction_vector = predicted_tail - predicted_head
    if np.dot(local_head_direction, global_direction_vector) < 0:
        local_head_direction *= -1
    norm_val = np.linalg.norm(local_head_direction)
    predicted_direction = local_head_direction / norm_val if norm_val > 0 else local_head_direction
    return (predicted_head, predicted_tail, predicted_direction), abs(score_A - score_B)


def plot_track_3d_plotly(result, save_dir):
    positions, charges = result['positions'], result['charges']
    true_origin = result['truth']['origin']
    pred_head, _, pred_dir = result['prediction']
    track_trace = go.Scatter3d(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], mode='markers',
                               marker=dict(size=2, color=charges, colorscale='Viridis',
                                           colorbar=dict(title='Charge (e-)'), showscale=True), name='Track Points')
    true_origin_trace = go.Scatter3d(x=[true_origin[0]], y=[true_origin[1]], z=[true_origin[2]],
                                     mode='markers', marker=dict(color='red', size=8, symbol='circle'),
                                     name='True Origin')
    pred_origin_trace = go.Scatter3d(x=[pred_head[0]], y=[pred_head[1]], z=[pred_head[2]],
                                     mode='markers', marker=dict(color='cyan', size=8, symbol='diamond'),
                                     name='Predicted Origin')
    dir_end_point = pred_head + pred_dir * 0.01
    dir_trace = go.Scatter3d(x=[pred_head[0], dir_end_point[0]], y=[pred_head[1], dir_end_point[1]],
                             z=[pred_head[2], dir_end_point[2]],
                             mode='lines', line=dict(color='magenta', width=5), name='Predicted Direction')
    fig = go.Figure(data=[track_trace, true_origin_trace, pred_origin_trace, dir_trace])
    title = (f"Track Analysis: {os.path.basename(result['path'])}<br>"
             f"Category: {result['category']} | "
             f"Origin Error: {result['origin_error_mm']:.2f}mm | "
             f"Angle Error: {result['angle_error']:.2f}°")
    fig.update_layout(title=title, scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                                              aspectratio=dict(x=1, y=1, z=1), camera_eye=dict(x=1.2, y=1.2, z=0.6)))
    filename = f"{os.path.basename(result['path']).replace('.npz', '.html')}"
    save_path = os.path.join(save_dir, filename)
    fig.write_html(save_path)


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- NEW: Create the main analysis directory and all category subdirectories ---
    os.makedirs(OUTPUT_ANALYSIS_DIR, exist_ok=True)
    good_plot_dir = os.path.join(OUTPUT_ANALYSIS_DIR, 'Good')
    bad_plot_dir = os.path.join(OUTPUT_ANALYSIS_DIR, 'Bad')
    excep_bad_plot_dir = os.path.join(OUTPUT_ANALYSIS_DIR, 'Exceptionally Bad')
    os.makedirs(good_plot_dir, exist_ok=True)
    os.makedirs(bad_plot_dir, exist_ok=True)
    os.makedirs(excep_bad_plot_dir, exist_ok=True)
    # --- END NEW ---

    all_track_files = find_track_files(BASE_DATA_PATH)
    all_results = []
    files_to_process = all_track_files[:MAX_TRACKS_TO_PROCESS] if MAX_TRACKS_TO_PROCESS else all_track_files

    # 1. --- FULL ANALYSIS PASS (Unchanged) ---
    for track_path in tqdm(files_to_process, desc="Analyzing All Tracks"):
        positions, charges, truth = load_data(track_path)
        if positions is None or len(positions) <= K_NEIGHBORS: continue
        prediction, score_diff = predict_head_tail_knn_v4(positions, charges, k=K_NEIGHBORS)
        if prediction is None: continue
        pred_head, _, pred_dir = prediction
        org_err = np.linalg.norm(pred_head - truth['origin'])
        ang_err = np.arccos(np.clip(np.dot(pred_dir, truth['direction']), -1.0, 1.0)) * 180 / np.pi
        if org_err <= GOOD_ORIGIN_THRESH_M:
            category = 'Good'
        elif ang_err > EXCEP_BAD_ANGLE_THRESH:
            category = 'Exceptionally Bad'
        else:
            category = 'Bad'
        all_results.append({
            'path': track_path, 'positions': positions, 'charges': charges,
            'prediction': prediction, 'truth': truth, 'category': category,
            'origin_error_mm': org_err * 1000, 'angle_error': ang_err
        })

    # 2. --- SELECTIVE PLOTTING (Modified) ---
    print("\nStarting selective plot generation into category subfolders...")

    good_results = [r for r in all_results if r['category'] == 'Good']
    bad_results = [r for r in all_results if r['category'] == 'Bad']
    excep_bad_results = [r for r in all_results if r['category'] == 'Exceptionally Bad']

    # Randomly sample the good tracks for plotting
    good_plots_to_generate = random.sample(good_results, min(NUM_GOOD_PLOTS_TO_SAVE, len(good_results)))

    # --- NEW: Separate loops for each category to save to the correct folder ---
    for result in tqdm(good_plots_to_generate, desc=f"Saving {len(good_plots_to_generate)} 'Good' Plots"):
        plot_track_3d_plotly(result, good_plot_dir)

    for result in tqdm(bad_results, desc=f"Saving {len(bad_results)} 'Bad' Plots"):
        plot_track_3d_plotly(result, bad_plot_dir)

    for result in tqdm(excep_bad_results, desc=f"Saving {len(excep_bad_results)} 'Exceptionally Bad' Plots"):
        plot_track_3d_plotly(result, excep_bad_plot_dir)
    # --- END MODIFICATION ---

    # 3. --- FILENAME REPORT GENERATION (Unchanged) ---
    print(f"\nWriting filename report to '{OUTPUT_FILENAME_REPORT}'...")
    with open(OUTPUT_FILENAME_REPORT, 'w') as f:
        f.write("=" * 60 + "\nTrack Filename Categorization Report (V4 Algorithm)\n" + "=" * 60 + "\n\n")
        for category in ["Good", "Bad", "Exceptionally Bad"]:
            results_in_cat = [r for r in all_results if r['category'] == category]
            f.write(f"--- {category} Tracks ({len(results_in_cat)} entries) ---\n")
            if not results_in_cat:
                f.write("No tracks in this category.\n")
            else:
                for result in results_in_cat:
                    f.write(f"{os.path.basename(result['path'])}\n")
            f.write("\n")

    print(f"\nAnalysis complete. All outputs saved in '{OUTPUT_ANALYSIS_DIR}' and '{OUTPUT_FILENAME_REPORT}'.")