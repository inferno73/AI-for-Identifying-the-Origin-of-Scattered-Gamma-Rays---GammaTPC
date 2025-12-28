# improved main_v4, direction enhanced!
# works for ALL tracks it can find
import os
import glob
import re
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import statistical_analysis
import analysis_toolkit

# --- Configuration ---
BASE_DATA_PATH = r'/sdf/home/b/bahrudin/gammaTPC/MLstudy723/processed_tracks906/'
# OUTPUT_PLOT_DIR removed from global scope, passed as argument
K_NEIGHBORS_SCORING = 20  # Number of neighbors for head/tail scoring
K_NEIGHBORS_SEGMENT = 50  # Number of neighbors for the direction segment
MAX_TRACKS_TO_PROCESS = None

# --- Categorization Thresholds ---
GOOD_ORIGIN_THRESH_M = 0.0016
EXCEP_BAD_ANGLE_THRESH = 100.0

# --- Core Algorithm Functions ---
def find_track_files(base_path, subfolders=['pair', 'compton']):
    track_files = []
    print(f"Searching for tracks in: {base_path}")
    print(f"Subfolders to include: {subfolders}")
    
    for subfolder in subfolders:
        # Construct pattern: base/subfolder/E*/d*/data/*.npz
        # Note: The user mentioned "processed_tracks906/" has both pair and compton folders.
        # Assuming structure is base_path/pair/E... and base_path/compton/E...
        search_pattern = os.path.join(base_path, subfolder, 'E*', 'd*', 'data', '*.npz')
        files = glob.glob(search_pattern, recursive=True) 
        # Filter out truth files just in case
        files = [f for f in files if not f.endswith('_truth.npz')]
        track_files.extend(files)
        
    print(f"Found {len(track_files)} tracks to analyze in {subfolders}.")
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


def extract_info_from_path(path):
    drift_match = re.search(r'drift_(\d{2})', path)
    energy_match = re.search(r'TrackE(\d{7})_', path)
    drift = f"d{drift_match.group(1)}" if drift_match else "unknown"
    energy = int(energy_match.group(1)) if energy_match else 0
    return drift, energy


def predict_head_tail_knn_v10(positions, charges, k_scoring, k_segment):
    """
    Uses SES method for head/tail identification, then Segment PCA for direction.
    """
    if len(positions) < k_scoring or len(positions) < k_segment: return None, None

    # Head/Tail Identification (SES Method)
    start_idx = 0
    dists_start = np.linalg.norm(positions - positions[start_idx], axis=1)
    ep1_idx = np.argmax(dists_start)
    dists_ep1 = np.linalg.norm(positions - positions[ep1_idx], axis=1)
    ep2_idx = np.argmax(dists_ep1)
    endpoints = {'A': {'pos': positions[ep1_idx]}, 'B': {'pos': positions[ep2_idx]}}
    knn_scoring = NearestNeighbors(n_neighbors=k_scoring).fit(positions)
    global_vec_ab = (endpoints['B']['pos'] - endpoints['A']['pos'])
    global_vec_ab /= np.linalg.norm(global_vec_ab) if np.linalg.norm(global_vec_ab) > 0 else 1
    for name, point in endpoints.items():
        indices = knn_scoring.kneighbors([point['pos']], return_distance=False)[0]
        local_points = positions[indices]
        point['avg_charge'] = np.mean(charges[indices])
        pca = PCA(n_components=1).fit(local_points)
        point['linearity'] = pca.explained_variance_ratio_[0]
        local_direction = pca.components_[0]
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
        predicted_head, predicted_tail = ep_A['pos'], ep_B['pos']
    else:
        predicted_head, predicted_tail = ep_B['pos'], ep_A['pos']

    # High-Stability Direction (Segment PCA)

    # 1. Define larger neighborhood (segment) around head
    knn_segment = NearestNeighbors(n_neighbors=k_segment).fit(positions)
    indices = knn_segment.kneighbors([predicted_head], return_distance=False)[0]
    segment_points = positions[indices]

    # 2. PCA on segment for stable direction
    pca_segment = PCA(n_components=1).fit(segment_points)
    segment_direction = pca_segment.components_[0]

    # 3. Orient using global axis
    global_direction_vector = predicted_tail - predicted_head
    if np.dot(segment_direction, global_direction_vector) < 0:
        segment_direction *= -1  # Flip to point away from head

    # 4. Normalize
    norm_val = np.linalg.norm(segment_direction)
    predicted_direction = segment_direction / norm_val if norm_val > 0 else segment_direction

    return (predicted_head, predicted_tail, predicted_direction), abs(score_A - score_B)


def run_analysis(dataset_name, output_dir, subfolders_to_include):
    print(f"\n{'='*50}")
    print(f"Starting Analysis: {dataset_name}")
    print(f"Output Directory: {output_dir}")
    print(f"Subfolders: {subfolders_to_include}")
    print(f"{'='*50}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    all_track_files = find_track_files(BASE_DATA_PATH, subfolders=subfolders_to_include)
    results = []

    files_to_process = all_track_files[:MAX_TRACKS_TO_PROCESS] if MAX_TRACKS_TO_PROCESS else all_track_files
    
    if not files_to_process:
        print(f"WARNING: No files found for {dataset_name}. Skipping analysis.")
        return

    for track_path in tqdm(files_to_process, desc=f"Analyzing {dataset_name}"):
        positions, charges, truth = load_data(track_path)
        if positions is None or len(positions) < K_NEIGHBORS_SCORING or len(positions) < K_NEIGHBORS_SEGMENT: continue

        prediction, score_diff = predict_head_tail_knn_v10(positions, charges, k_scoring=K_NEIGHBORS_SCORING,
                                                           k_segment=K_NEIGHBORS_SEGMENT)
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

        drift, energy = extract_info_from_path(track_path)

        results.append({
            'path': track_path, 'positions': positions, 'charges': charges, 'prediction': prediction, 'truth': truth,
            'category': category, 'drift': drift, 'energy': energy,
            'track_length': np.max(np.ptp(positions, axis=0)) * 1000,
            'total_charge': np.sum(charges), 'score_diff': score_diff,
            'origin_error_mm': org_err * 1000, 'angle_error': ang_err
        })

    if results:
        # Pass a custom title or version name to the reports so they are distinguishable
        version_label = f"V10-SegmentPCA-{dataset_name}"
        statistical_analysis.generate_report_and_plots(results, output_dir, version_name=version_label)
        analysis_toolkit.generate_analysis_report(results, output_dir, version_name=version_label)
        print(f"Finished {dataset_name}. Results saved to {output_dir}")
    else:
        print(f"No valid results to report for {dataset_name}.")


if __name__ == "__main__":
    # 1. Run for ALL (Pairs + Comptons)
    run_analysis(
        dataset_name="ALL_Mixed",
        output_dir="tracks906/result_all",
        subfolders_to_include=['pair', 'compton']
    )
    
    # 2. Run for PAIRS Only
    run_analysis(
        dataset_name="Only_Pairs",
        output_dir="tracks906/result_pairs",
        subfolders_to_include=['pair']
    )
    
    # 3. Run for COMPTONS Only
    run_analysis(
        dataset_name="Only_Comptons",
        output_dir="tracks906/result_compton",
        subfolders_to_include=['compton']
    )
