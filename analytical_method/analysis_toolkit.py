# analysis_toolkit.py (Version 2)

import os
import random
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go


def generate_analysis_report(all_results, results_to_plot, output_dir, version_name="Analysis"):
    """
    A generalized function to perform a full analysis and generate plots and reports.

    Args:
        all_results (list): The COMPLETE list of all analyzed tracks for the text report.
        results_to_plot (list): A pre-sampled list of tracks to generate HTML plots for.
        output_dir (str): The path to the main output folder for this analysis.
        version_name (str): A name for this analysis run (e.g., "V10").
    """
    print("\n" + "=" * 60)
    print(f"STARTING ANALYSIS AND REPORTING: {version_name}")
    print("=" * 60)

    # 1. --- SETUP DIRECTORIES ---
    os.makedirs(output_dir, exist_ok=True)
    good_plot_dir = os.path.join(output_dir, 'Good')
    bad_plot_dir = os.path.join(output_dir, 'Bad')
    excep_bad_plot_dir = os.path.join(output_dir, 'Exceptionally Bad')
    os.makedirs(good_plot_dir, exist_ok=True)
    os.makedirs(bad_plot_dir, exist_ok=True)
    os.makedirs(excep_bad_plot_dir, exist_ok=True)

    # 2. --- PLOTTING (using the pre-sampled list) ---
    print(f"\nGenerating {len(results_to_plot)} specified HTML plots...")

    for result in tqdm(results_to_plot, desc="Saving HTML Plots"):
        # Determine the correct subfolder based on the track's category
        category = result['category']
        if category == 'Good':
            save_dir = good_plot_dir
        elif category == 'Bad':
            save_dir = bad_plot_dir
        else:  # Exceptionally Bad
            save_dir = excep_bad_plot_dir

        _plot_track_3d_plotly(result, save_dir)

    # 3. --- FILENAME REPORT GENERATION (using the complete list) ---
    report_filename = f"track_categorization_report_{version_name}.txt"
    print(f"\nWriting complete filename report to '{report_filename}'...")

    with open(report_filename, 'w') as f:
        f.write("=" * 60 + f"\nTrack Filename Categorization Report ({version_name} Algorithm)\n" + "=" * 60 + "\n\n")
        for category in ["Good", "Bad", "Exceptionally Bad"]:
            # Filter the FULL list of results for the report
            results_in_cat = [r for r in all_results if r['category'] == category]
            f.write(f"--- {category} Tracks ({len(results_in_cat)} entries) ---\n")
            if not results_in_cat:
                f.write("No tracks in this category.\n")
            else:
                for result in results_in_cat:
                    f.write(f"{os.path.basename(result['path'])}\n")
            f.write("\n")

    print(f"\nAnalysis complete. All outputs saved in '{output_dir}' and '{report_filename}'.")


def _plot_track_3d_plotly(result, save_dir):
    """ (Internal helper function - Unchanged from before) """
    positions, charges = result['positions'], result['charges']
    true_origin, true_dir = result['truth']['origin'], result['truth']['direction']
    pred_head, pred_tail, pred_dir = result['prediction']
    track_trace = go.Scatter3d(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], mode='markers',
                               marker=dict(size=2, color=charges, colorscale='Viridis',
                                           colorbar=dict(title='Charge (e⁻)'), showscale=True), name='Track Points')
    true_origin_trace = go.Scatter3d(x=[true_origin[0]], y=[true_origin[1]], z=[true_origin[2]],
                                     mode='markers', marker=dict(color='red', size=8, symbol='circle'),
                                     name='True Origin')
    pred_origin_trace = go.Scatter3d(x=[pred_head[0]], y=[pred_head[1]], z=[pred_head[2]],
                                     mode='markers', marker=dict(color='cyan', size=8, symbol='diamond'),
                                     name='Predicted Origin')
    pred_tail_trace = go.Scatter3d(x=[pred_tail[0]], y=[pred_tail[1]], z=[pred_tail[2]],
                                   mode='markers', marker=dict(color='blue', size=8, symbol='x'), name='Predicted Tail')
    dir_scale = 0.01
    true_dir_end = true_origin + true_dir * dir_scale
    pred_dir_end = pred_head + pred_dir * dir_scale
    true_dir_trace = go.Scatter3d(x=[true_origin[0], true_dir_end[0]], y=[true_origin[1], true_dir_end[1]],
                                  z=[true_origin[2], true_dir_end[2]],
                                  mode='lines', line=dict(color='red', width=5), name='True Direction')
    pred_dir_trace = go.Scatter3d(x=[pred_head[0], pred_dir_end[0]], y=[pred_head[1], pred_dir_end[1]],
                                  z=[pred_head[2], pred_dir_end[2]],
                                  mode='lines', line=dict(color='magenta', width=5), name='Predicted Direction')
    fig = go.Figure(
        data=[track_trace, true_origin_trace, pred_origin_trace, pred_tail_trace, true_dir_trace, pred_dir_trace])
    title = (f"Track Analysis: {os.path.basename(result['path'])}<br>"
             f"Category: {result['category']} | Origin Error: {result['origin_error_mm']:.2f}mm | Angle Error: {result['angle_error']:.2f}°")
    fig.update_layout(title=title, legend=dict(x=0.01, y=0.99), scene=dict(
        xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
        aspectratio=dict(x=1, y=1, z=1), camera_eye=dict(x=1.2, y=1.2, z=0.6)))
    filename = f"{os.path.basename(result['path']).replace('.npz', '.html')}"
    save_path = os.path.join(save_dir, filename)
    fig.write_html(save_path)