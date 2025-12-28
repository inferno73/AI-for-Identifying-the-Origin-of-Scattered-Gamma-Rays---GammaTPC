import numpy as np
import plotly.graph_objects as go
import glob
import os
from sklearn.cluster import KMeans

def load_track_raw(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        # Coordinates
        if 'hits' in data: coords = data['hits']
        elif 'coords' in data: coords = data['coords']
        elif 'x' in data: coords = np.stack([data['x'], data['y'], data['z']], axis=1)
        else: return None
        
        # Cleanup column count
        if coords.shape[1] > 3: coords = coords[:, :3]
        
    return coords

def plot_clustering(coords, labels, filename):
    """Plot the two clusters with different colors."""
    
    # Cluster 0
    c0 = coords[labels == 0]
    # Cluster 1
    c1 = coords[labels == 1]
    
    fig = go.Figure()
    
    # Cluster 0 (Cyan)
    fig.add_trace(go.Scatter3d(
        x=c0[:,0], y=c0[:,1], z=c0[:,2],
        mode='markers', marker=dict(size=4, color='cyan'),
        name='Arm A'
    ))
    
    # Cluster 1 (Magenta)
    fig.add_trace(go.Scatter3d(
        x=c1[:,0], y=c1[:,1], z=c1[:,2],
        mode='markers', marker=dict(size=4, color='magenta'),
        name='Arm B'
    ))
    
    # Origin
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers', marker=dict(size=6, color='red'),
        name='Origin'
    ))
    
    fig.update_layout(title=f"Pair Separation Test: {filename}")
    
    # Save
    out_path = f"pair_test_{filename}.html"
    fig.write_html(out_path)
    print(f"Saved plot to {out_path}")

def run_test():
    DATA_DIR = r"C:\Users\Korisnik\PycharmProjects\gammaAIModel\data-samples\local_data\pair"
    
    # Get random files
    files = glob.glob(os.path.join(DATA_DIR, "**", "*.npz"), recursive=True)
    files = [f for f in files if not f.endswith('_truth.npz')]
    
    # Pick 3 random ones
    np.random.shuffle(files)
    sample_files = files[:3]
    
    print(f"Testing KMeans Separation on 3 Pair Tracks...")
    
    for f in sample_files:
        coords = load_track_raw(f)
        fname = os.path.basename(f)
        
        if len(coords) < 10: continue
        
        # METHOD: KMeans k=2
        # Simple Euclidean clustering
        kmeans = KMeans(n_clusters=2, n_init=10).fit(coords)
        labels = kmeans.labels_
        
        plot_clustering(coords, labels, fname)

if __name__ == "__main__":
    run_test()
