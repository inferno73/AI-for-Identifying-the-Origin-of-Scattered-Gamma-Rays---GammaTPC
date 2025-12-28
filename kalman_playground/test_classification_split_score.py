import numpy as np
import glob
import os
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from tqdm import tqdm

def load_track_raw(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        if 'hits' in data: coords = data['hits']
        elif 'coords' in data: coords = data['coords']
        elif 'x' in data: coords = np.stack([data['x'], data['y'], data['z']], axis=1)
        else: return None
        if coords.shape[1] > 3: coords = coords[:, :3]
    return coords

def fit_line_mse(points):
    if len(points) < 2: return 0.0
    mean = np.mean(points, axis=0)
    centered = points - mean
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    
    # Distances to line
    # Vector from mean to points
    vecs = points - mean
    # Project onto direction
    projections = np.dot(vecs, direction)
    # Reconstruct
    reconstructed = mean + np.outer(projections, direction)
    # Residuals
    residuals = points - reconstructed
    mse = np.mean(np.sum(residuals**2, axis=1))
    return mse

def get_split_score(coords):
    # 1. Fit Single Line
    mse_1 = fit_line_mse(coords)
    if mse_1 < 1e-9: return 1.0 # Perfect line already
     
    # 2. Split (KMeans k=2)
    # Initialize with K-means++
    try:
        kmeans = KMeans(n_clusters=2, n_init=5).fit(coords)
        labels = kmeans.labels_
        
        c1 = coords[labels == 0]
        c2 = coords[labels == 1]
        
        err1 = fit_line_mse(c1) * len(c1) # Total Squared Error
        err2 = fit_line_mse(c2) * len(c2)
        
        total_mse_2 = (err1 + err2) / len(coords)
        
        if total_mse_2 < 1e-9: return 999.0
        
        ratio = mse_1 / total_mse_2
        return ratio
    except:
        return 1.0

def analyze():
    ROOT_DIR = r"C:\Users\Korisnik\PycharmProjects\gammaAIModel\data-samples\local_data"
    COMPTON_DIR = os.path.join(ROOT_DIR, "compton")
    PAIR_DIR = os.path.join(ROOT_DIR, "pair")
    
    vals_c = []
    vals_p = []
    
    # 1. Compton
    print("Scanning Compton (Split Score)...")
    c_files = glob.glob(os.path.join(COMPTON_DIR, "**", "*.npz"), recursive=True)
    c_files = [f for f in c_files if not f.endswith('_truth.npz')]
    np.random.shuffle(c_files)
    c_files = c_files[:300]
    
    for f in tqdm(c_files):
        coords = load_track_raw(f)
        try:
             score = get_split_score(coords)
             vals_c.append(score)
        except: pass
        
    # 2. Pair
    print("Scanning Pair (Split Score)...")
    p_files = glob.glob(os.path.join(PAIR_DIR, "**", "*.npz"), recursive=True)
    p_files = [f for f in p_files if not f.endswith('_truth.npz')]
    
    for f in tqdm(p_files):
        coords = load_track_raw(f)
        try:
             score = get_split_score(coords)
             vals_p.append(score)
        except: pass

    # Stats
    vc = np.array(vals_c)
    vp = np.array(vals_p)
    print(f"Compton Mean Score: {np.mean(vc):.4f} +/- {np.std(vc):.4f}")
    print(f"Pair Mean Score:    {np.mean(vp):.4f} +/- {np.std(vp):.4f}")
    
    # Histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=vals_c, name='Compton', opacity=0.75, marker_color='blue', nbinsx=50))
    fig.add_trace(go.Histogram(x=vals_p, name='Pair', opacity=0.75, marker_color='red', nbinsx=50))
    
    fig.update_layout(title="Split Improvement Ratio (MSE1 / MSE2)", xaxis_title="Ratio", barmode='overlay')
    fig.write_html("track_classification_split.html")
    print("Saved plot.")

if __name__ == "__main__":
    analyze()
