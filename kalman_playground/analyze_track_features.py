import numpy as np
import glob
import os
import plotly.graph_objects as go
from tqdm import tqdm

def load_track_raw(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        if 'hits' in data: coords = data['hits']
        elif 'coords' in data: coords = data['coords']
        elif 'x' in data: coords = np.stack([data['x'], data['y'], data['z']], axis=1)
        else: return None
        if coords.shape[1] > 3: coords = coords[:, :3]
    return coords

def get_kurtosis(values):
    n = len(values)
    if n < 4: return 0
    mean = np.mean(values)
    std = np.std(values)
    if std < 1e-9: return 0
    
    moment4 = np.sum((values - mean)**4) / n
    kurt = moment4 / (std**4)
    return kurt - 3.0 # Excess Kurtosis (Normal=0)

def get_pca_features(coords):
    # LATERAL KURTOSIS (N=50)
    # Hypothesis: Pair track splits -> Bimodal lateral dist -> Low Kurtosis
    # Compton track -> Unimodal/Arc -> Higher Kurtosis
    n = min(len(coords), 50)
    local_pts = coords[:n]
    
    if len(local_pts) < 5: return 0
    
    # Center
    mean = np.mean(local_pts, axis=0)
    centered = local_pts - mean
    
    # PCA
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    
    # 2nd Component (Lateral Axis)
    # Project points onto this axis
    lateral_axis = vh[1]
    projections = np.dot(centered, lateral_axis)
    
    # Kurtosis of lateral distribution
    k = get_kurtosis(projections)
    
    return k

def analyze():
    # PATHS
    ROOT_DIR = r"C:\Users\Korisnik\PycharmProjects\gammaAIModel\data-samples\local_data"
    COMPTON_DIR = os.path.join(ROOT_DIR, "compton")
    PAIR_DIR = os.path.join(ROOT_DIR, "pair")
    
    # lists
    vals_c = []
    vals_p = []
    
    # 1. Compton
    print("Scanning Compton (Kurtosis N=50)...")
    c_files = glob.glob(os.path.join(COMPTON_DIR, "**", "*.npz"), recursive=True)
    c_files = [f for f in c_files if not f.endswith('_truth.npz')]
    np.random.shuffle(c_files)
    c_files = c_files[:300]
    
    for f in tqdm(c_files):
        coords = load_track_raw(f)
        try:
             dists = np.linalg.norm(coords, axis=1)
             start_idx = np.argmin(dists)
             dists_from_start = np.linalg.norm(coords - coords[start_idx], axis=1)
             sorted_idxs = np.argsort(dists_from_start)
             sorted_coords = coords[sorted_idxs]
             
             k = get_pca_features(sorted_coords)
             vals_c.append(k)
        except: pass
        
    # 2. Pair
    print("Scanning Pair (Kurtosis N=50)...")
    p_files = glob.glob(os.path.join(PAIR_DIR, "**", "*.npz"), recursive=True)
    p_files = [f for f in p_files if not f.endswith('_truth.npz')]
    
    for f in tqdm(p_files):
        coords = load_track_raw(f)
        try:
             dists = np.linalg.norm(coords, axis=1)
             start_idx = np.argmin(dists)
             dists_from_start = np.linalg.norm(coords - coords[start_idx], axis=1)
             sorted_idxs = np.argsort(dists_from_start)
             sorted_coords = coords[sorted_idxs]
             
             k = get_pca_features(sorted_coords)
             vals_p.append(k)
        except: pass

    # Stats
    vc = np.array(vals_c)
    vp = np.array(vals_p)
    print(f"Compton Mean Kurtosis: {np.mean(vc):.4f} +/- {np.std(vc):.4f}")
    print(f"Pair Mean Kurtosis:    {np.mean(vp):.4f} +/- {np.std(vp):.4f}")
    
    # Histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=vals_c, name='Compton', opacity=0.75, marker_color='blue', nbinsx=100))
    fig.add_trace(go.Histogram(x=vals_p, name='Pair', opacity=0.75, marker_color='red', nbinsx=100))
    
    fig.update_layout(title="Lateral Kurtosis (N=50)", xaxis_title="Excess Kurtosis", barmode='overlay')
    fig.write_html("track_classification_kurtosis.html")
    print("Saved plot.")
        
    print(f"Collected {len(records)} samples.")
    
    # Visualize
    # Metric: Ratio sqrt(l2)/sqrt(l1) = Width / Length
    
    ratios_c = []
    ratios_p = []
    
    for r in records:
        # Avoid div by zero
        ratio = np.sqrt(r['l2']) / (np.sqrt(r['l1']) + 1e-9)
        if r['type'] == 'compton':
            ratios_c.append(ratio)
        else:
            ratios_p.append(ratio)
            
    # Histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ratios_c, name='Compton', opacity=0.75, marker_color='blue', nbinsx=50))
    fig.add_trace(go.Histogram(x=ratios_p, name='Pair', opacity=0.75, marker_color='red', nbinsx=50))
    
    fig.update_layout(
        title="Distribution of Track Aspect Ratio (Width/Length)",
        xaxis_title="Ratio (sqrt(L2)/sqrt(L1))",
        yaxis_title="Count",
        barmode='overlay'
    )
    
    out_path = "track_classification_analysis.html"
    fig.write_html(out_path)
    print(f"Saved plot to {out_path}")
    
    # Print stats
    rc = np.array(ratios_c)
    rp = np.array(ratios_p)
    print(f"Compton Mean Ratio: {np.mean(rc):.4f} +/- {np.std(rc):.4f}")
    print(f"Pair Mean Ratio:    {np.mean(rp):.4f} +/- {np.std(rp):.4f}")

if __name__ == "__main__":
    analyze()
