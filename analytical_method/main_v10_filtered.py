import os
import glob
import re
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import statistical_analysis
import analysis_toolkit
import angle_analyzer

# --- Configuration ---
BASE_DATA_PATH = r'/sdf/home/b/bahrudin/gammaTPC/MLstudy723/processed_tracks814' #r'data'
OUTPUT_PLOT_DIR = 'latest-v10-HigherEnergies'
# The report file will be named based on the version_name below
VERSION_NAME = "V10-HighEnergies-Detailed"

# --- Specify which energy folders to process ---
# Higher energies
ENERGY_FOLDERS = ['10000', '20000', '50000']
# lower energies
#ENERGY_FOLDERS = ['1000', '3000', '5000']

# --- Sampling & Algorithm Config ---
NUM_PLOTS_PER_CATEGORY = 10
K_NEIGHBORS_SCORING = 20
K_NEIGHBORS_SEGMENT = 25
MAX_TRACKS_TO_PROCESS = None
GOOD_ORIGIN_THRESH_M = 0.0016
EXCEP_BAD_ANGLE_THRESH = 100.0


# --- Core Algorithm Functions (Unchanged) ---
def find_track_files(base_path, energy_folders_to_include):
    all_track_files = []
    print(f"Searching for tracks ONLY in specified energy folders: {energy_folders_to_include}")
    for energy_kev in energy_folders_to_include:
        energy_dir_name = f"E{int(energy_kev):07d}"
        search_pattern = os.path.join(base_path, 'compton', energy_dir_name, 'd*', 'data', '*.npz')
        track_files_for_energy = [f for f in glob.glob(search_pattern, recursive=True) if not f.endswith('_truth.npz')]
        all_track_files.extend(track_files_for_energy)
    print(f"Found {len(all_track_files)} tracks to analyze in the specified folders.")
    return all_track_files
# robust version update for termius
def load_data(track_path):
    """
    Loads positions, charges, and truth data for a single track,
    now with a check for empty/corrupt files.
    """
    truth_path = track_path.replace('.npz', '_truth.npz')
    # Check if both files exist AND have content (size > 0 bytes)
    if not os.path.exists(truth_path) or not os.path.exists(track_path):
        return None, None, None

    if os.path.getsize(track_path) == 0 or os.path.getsize(truth_path) == 0:
        # Silently skip empty files, or uncomment the print statement for debugging
        # print(f"Warning: Skipping empty or corrupt file pair: {track_path}")
        return None, None, None
    try:
        with np.load(track_path) as data:
            positions = np.stack([data['x'], data['y'], data['z']], axis=1)
            charges = data['charge']
        with np.load(truth_path, allow_pickle=True) as data:
            truth = {'origin': data['track_origin'], 'direction': data['direction']}
        return positions, charges, truth
    except EOFError:
        # print(f"Warning: Caught EOFError for file pair, skipping: {track_path}")
        return None, None, None

def extract_info_from_path(path):
    drift_match = re.search(r'drift_(\d{2})', path);
    energy_match = re.search(r'TrackE(\d{7})_', path)
    drift = f"d{drift_match.group(1)}" if drift_match else "unknown";
    energy = int(energy_match.group(1)) if energy_match else 0
    return drift, energy


def predict_head_tail_knn_v10(positions, charges, k_scoring, k_segment):
    if len(positions) < k_scoring or len(positions) < k_segment: return None, None
    start_idx = 0;
    dists_start = np.linalg.norm(positions - positions[start_idx], axis=1);
    ep1_idx = np.argmax(dists_start)
    dists_ep1 = np.linalg.norm(positions - positions[ep1_idx], axis=1);
    ep2_idx = np.argmax(dists_ep1)
    endpoints = {'A': {'pos': positions[ep1_idx]}, 'B': {'pos': positions[ep2_idx]}};
    knn_scoring = NearestNeighbors(n_neighbors=k_scoring).fit(positions)
    global_vec_ab = (endpoints['B']['pos'] - endpoints['A']['pos']);
    global_vec_ab /= np.linalg.norm(global_vec_ab) if np.linalg.norm(global_vec_ab) > 0 else 1
    for name, point in endpoints.items():
        indices = knn_scoring.kneighbors([point['pos']], return_distance=False)[0];
        local_points = positions[indices]
        point['avg_charge'] = np.mean(charges[indices]);
        pca = PCA(n_components=1).fit(local_points);
        point['linearity'] = pca.explained_variance_ratio_[0]
        local_direction = pca.components_[0];
        global_vec = global_vec_ab if name == 'A' else -global_vec_ab
        dot_product = np.clip(np.dot(local_direction, global_vec), -1.0, 1.0);
        angle = np.arccos(dot_product) * 180 / np.pi;
        point['curvature'] = min(angle, 180 - angle)
    ep_A, ep_B = endpoints['A'], endpoints['B'];
    total_charge = ep_A['avg_charge'] + ep_B['avg_charge'];
    norm_charge_A = ep_A['avg_charge'] / total_charge if total_charge > 0 else 0.5
    total_curve = ep_A['curvature'] + ep_B['curvature'];
    norm_curve_A = ep_A['curvature'] / total_curve if total_curve > 0 else 0.5
    score_A = (0.5 * ep_A['linearity']) - (1.0 * norm_charge_A) - (1.5 * norm_curve_A)
    score_B = (0.5 * ep_B['linearity']) - (1.0 * (1.0 - norm_charge_A)) - (1.5 * (1.0 - norm_curve_A))
    predicted_head, predicted_tail = (ep_A['pos'], ep_B['pos']) if score_A > score_B else (ep_B['pos'], ep_A['pos'])
    knn_segment = NearestNeighbors(n_neighbors=k_segment).fit(positions);
    indices = knn_segment.kneighbors([predicted_head], return_distance=False)[0]
    segment_points = positions[indices];
    pca_segment = PCA(n_components=1).fit(segment_points);
    segment_direction = pca_segment.components_[0]
    global_direction_vector = predicted_tail - predicted_head
    if np.dot(segment_direction, global_direction_vector) < 0: segment_direction *= -1
    norm_val = np.linalg.norm(segment_direction);
    predicted_direction = segment_direction / norm_val if norm_val > 0 else segment_direction
    return (predicted_head, predicted_tail, predicted_direction), abs(score_A - score_B)


if __name__ == "__main__":
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    all_track_files = find_track_files(BASE_DATA_PATH, ENERGY_FOLDERS)
    all_results = []

    files_to_process = all_track_files[:MAX_TRACKS_TO_PROCESS] if MAX_TRACKS_TO_PROCESS else all_track_files

    for track_path in tqdm(files_to_process, desc=f"Analyzing Tracks ({VERSION_NAME})"):
        positions, charges, truth = load_data(track_path)
        if positions is None or len(positions) < K_NEIGHBORS_SCORING or len(positions) < K_NEIGHBORS_SEGMENT: continue
        prediction, score_diff = predict_head_tail_knn_v10(positions, charges, K_NEIGHBORS_SCORING, K_NEIGHBORS_SEGMENT)
        if prediction is None: continue
        pred_head, _, pred_dir = prediction
        org_err = np.linalg.norm(pred_head - truth['origin'])
        ang_err = np.arccos(np.clip(np.dot(pred_dir, truth['direction']), -1.0, 1.0)) * 180 / np.pi

        drift, energy = extract_info_from_path(track_path)

        # Categorization based on origin error is still useful for plotting context
        if org_err <= GOOD_ORIGIN_THRESH_M:
            category = 'Good'
        elif ang_err > EXCEP_BAD_ANGLE_THRESH:
            category = 'Exceptionally Bad'
        else:
            category = 'Bad'

        all_results.append({'path': track_path, 'positions': positions, 'charges': charges, 'prediction': prediction,
                            'truth': truth, 'category': category, 'drift': drift, 'energy': energy,
                            'track_length': np.max(np.ptp(positions, axis=0)) * 1000, 'total_charge': np.sum(charges),
                            'score_diff': score_diff, 'origin_error_mm': org_err * 1000, 'angle_error': ang_err})

    # --- FINAL ANALYSIS STEP ---
    if not all_results:
        print("\nNo tracks processed. Exiting.")
    else:
        # 1. Generate the console report (using your simplified script)
        statistical_analysis.generate_report_and_plots(all_results, OUTPUT_PLOT_DIR, version_name="V10-HighEnergies")

        # 2. Generate the detailed HTML plots and filename report (using the powerful toolkit)
        # This includes the sampling logic from before.
        good_results = [r for r in all_results if r['category'] == 'Good']
        bad_results = [r for r in all_results if r['category'] == 'Bad']
        excep_bad_results = [r for r in all_results if r['category'] == 'Exceptionally Bad']
        good_plots = random.sample(good_results, min(NUM_PLOTS_PER_CATEGORY, len(good_results)))
        bad_plots = random.sample(bad_results, min(NUM_PLOTS_PER_CATEGORY, len(bad_results)))
        excep_bad_plots = random.sample(excep_bad_results, min(NUM_PLOTS_PER_CATEGORY, len(excep_bad_results)))
        results_to_plot = good_plots + bad_plots + excep_bad_plots

        analysis_toolkit.generate_analysis_report(
            all_results=all_results,
            results_to_plot=results_to_plot,
            output_dir=OUTPUT_PLOT_DIR,
            version_name="V10-HighEnergies"
        )

        # --- 3. Generate and save the detailed angle analysis report ---
        print(f"\nGenerating detailed angle analysis...")
        angle_report_content = angle_analyzer.generate_report(all_results, ENERGY_FOLDERS,
                                                                       version_name=VERSION_NAME)

        report_filename = f"detailed_angle_report_{VERSION_NAME}.txt"
        with open(report_filename, 'w') as f:
            f.write(angle_report_content)

        print(f"Detailed angle report saved to '{report_filename}'.")
        # Also print to console for immediate feedback
        print("\n--- Detailed Angle Report ---")
        print(angle_report_content)
