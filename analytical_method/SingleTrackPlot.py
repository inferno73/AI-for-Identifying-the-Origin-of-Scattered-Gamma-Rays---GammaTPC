import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D projection

# --- Configuration ---
track_file_path = r'data/compton/E0050000/d05/data/TrackE0050000_D20250804_T1413221554_drift_05.npz'
truth_file_path = r'data/compton/E0050000/d05/data/TrackE0050000_D20250804_T1413221554_drift_05_truth.npz'

# --- Data Loading ---
if not os.path.exists(track_file_path):
    print(f"Error: Track file not found at: {track_file_path}")
elif not os.path.exists(truth_file_path):
    print(f"Error: Truth file not found at: {truth_file_path}")
else:
    try:
        # Load the track data
        with np.load(track_file_path) as track_data:
            positions = np.stack([track_data['x'], track_data['y'], track_data['z']], axis=1)
            charges = track_data['charge']

        # Load the truth data
        with np.load(truth_file_path, allow_pickle=True) as truth_data:
            origin = truth_data['origin']
            direction = truth_data['direction']
            track_origin = truth_data['track_origin']
            # --- MODIFICATION 1: Load interaction type as a string ---
            # It may be stored as a 0-d array, so .item() is a safe way to extract it.
            interaction_type = truth_data['first_interaction'].item()
            original_origin = truth_data['original_origin']

        # --- Display Loaded Information ---
        print(f"Successfully loaded track: {os.path.basename(track_file_path)}")
        print(f"Number of points: {len(positions)}")
        print("-" * 30)
        print("Truth Information:")
        print(f"  Origin: {origin}")
        print(f"  Direction: {direction}")
        print(f"  Track Origin: {track_origin}")
        # Display the correctly loaded interaction type
        print(f"  First Interaction Type: {interaction_type}")
        print(f"  Original Origin: {original_origin}")
        print("-" * 30)

        # --- Plotting ---
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 1. Plot the electron track itself
        sc = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                        c=charges, cmap='viridis', s=5, alpha=0.6, label='Electron Track')

        # 2. Plot the truth points
        ax.scatter(origin[0], origin[1], origin[2], color='red', s=200,
                   marker='*', label='Origin')
        ax.scatter(track_origin[0], track_origin[1], track_origin[2], color='blue', s=100,
                   marker='s', label='Track Origin')
        # --- MODIFICATION 2: The line trying to plot 'first_interaction' is REMOVED ---
        ax.scatter(original_origin[0], original_origin[1], original_origin[2], color='magenta', s=120,
                   marker='X', label='Original Origin')

        # 3. Plot the initial direction vector
        track_extent = np.max(np.ptp(positions, axis=0))
        ax.quiver(origin[0], origin[1], origin[2],
                  direction[0], direction[1], direction[2],
                  length=track_extent * 0.4,
                  normalize=True,
                  color='red', label='Initial Direction', lw=2.5)

        # --- Formatting the Plot ---
        # --- MODIFICATION 3: Add the interaction type to the title ---
        ax.set_title(f"3D Track Visualization ({str(interaction_type).capitalize()} Interaction)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)

        plt.colorbar(sc, label="Charge (e⁻)", shrink=0.6)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")