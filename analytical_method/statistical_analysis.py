import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def _save_plotly_figure(case_data, file_prefix, output_dir):
    """Saves an interactive 3D plot to an HTML file."""
    path, pos, chg, pred, tru, err = case_data
    pred_head, pred_tail, _ = pred

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode='markers',
                               marker=dict(size=2.5, color=chg, colorscale='Viridis'), name='Electron Track'))
    fig.add_trace(go.Scatter3d(x=[pred_head[0]], y=[pred_head[1]], z=[pred_head[2]], mode='markers',
                               marker=dict(size=8, color='cyan', symbol='diamond'), name='Predicted Head'))
    fig.add_trace(go.Scatter3d(x=[pred_tail[0]], y=[pred_tail[1]], z=[pred_tail[2]], mode='markers',
                               marker=dict(size=8, color='magenta', symbol='x'), name='Predicted Tail'))
    fig.add_trace(go.Scatter3d(x=[tru['origin'][0]], y=[tru['origin'][1]], z=[tru['origin'][2]], mode='markers',
                               marker=dict(size=8, color='red', symbol='diamond', line=dict(color='black', width=2)),
                               name='True Origin'))

    title = f"{file_prefix.upper()} Case: {os.path.basename(path)}<br>Angle Error: {err:.2f}°"
    fig.update_layout(title=title, scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)'),
                      legend=dict(x=0, y=1))

    filename = f"{file_prefix}_{err:.2f}deg_{os.path.basename(path).replace('.npz', '.html')}"
    filepath = os.path.join(output_dir, filename)
    fig.write_html(filepath)


def generate_report_and_plots(results, output_dir, version_name):
    """
    Takes the raw results, generates all statistical reports, and saves plots.
    """
    if not results:
        print("\nNo tracks were processed.")
        return

    df = pd.DataFrame(results)

    # --- 1. Overall Performance Summary ---
    origin_errors_mm = df['origin_error_mm']
    direction_errors = df['angle_error']

    print("\n" + "=" * 50)
    print(f"OVERALL PERFORMANCE SUMMARY ({version_name})")
    print("=" * 50)
    print(f"\nAnalyzed {len(df)} tracks.")
    print("\n--- Origin Error (mm) ---\n" f"  Mean:   {origin_errors_mm.mean():.2f} mm\n"
          f"  Median: {origin_errors_mm.median():.2f} mm\n" f"  Std Dev:  {np.std(origin_errors_mm):.2f} mm")
    print("\n--- Direction Error (°) ---\n" f"  Mean:   {direction_errors.mean():.2f}°\n"
          f"  Median: {direction_errors.median():.2f}°\n" f"  Std Dev:  {np.std(direction_errors):.2f}°")
    print("=" * 50)

    # --- 2. Detailed Categorical Analysis ---
    categories = {
        'Good': df[df['category'] == 'Good'],
        'Bad': df[df['category'] == 'Bad'],
        'Exceptionally Bad': df[df['category'] == 'Exceptionally Bad']
    }

    print("\n" + "=" * 80 + f"\n{version_name} ALGORITHM PERFORMANCE ANALYSIS\n" + "=" * 80)

    for name, data in categories.items():
        total_tracks = len(df)
        category_tracks = len(data)
        percentage = (category_tracks / total_tracks * 100) if total_tracks > 0 else 0
        print(f"\n--- CATEGORY: {name} ({category_tracks} tracks, {percentage:.1f}%) ---")
        if data.empty:
            print("No tracks in this category.")
            continue

        print("\n1. Performance Metrics:")
        print(
            f"  - Origin Error (mm): Mean={data['origin_error_mm'].mean():.2f}, Median={data['origin_error_mm'].median():.2f}, StdDev={data['origin_error_mm'].std():.2f}")
        print(
            f"  - Angle Error (°):   Mean={data['angle_error'].mean():.2f}, Median={data['angle_error'].median():.2f}, StdDev={data['angle_error'].std():.2f}")

        print("\n2. Track Properties:")
        print(f"  - Avg. Track Length: {data['track_length'].mean():.2f} mm")
        print(f"  - Avg. Total Charge: {data['total_charge'].mean():.2f} e⁻")
        print(f"  - Avg. Score Diff:   {data['score_diff'].mean():.2f}")

        print("\n3. Drift Distribution:")
        drift_dist = data['drift'].value_counts(normalize=True).mul(100).sort_index()
        for drift, pct in drift_dist.items(): print(f"  - {drift}: {pct:.1f}%")

        print("\n4. Energy Distribution (Top 5):")
        energy_dist = data['energy'].value_counts(normalize=True).mul(100)
        for energy, pct in energy_dist.head(5).items(): print(f"  - {energy / 1000:.0f} MeV: {pct:.1f}%")
        if len(energy_dist) > 5: print(f"  - ({len(energy_dist) - 5} other energies)")

    # --- 3. Save Plots ---
    # df_sorted = df.sort_values(by='angle_error')
    # top_10_best = df_sorted.head(10).to_dict('records')
    # top_10_worst = df_sorted.tail(10).iloc[::-1].to_dict('records')
    #
    # print("\n" + "=" * 80 + "\nSAVING PLOTS\n" + "=" * 80)
    # print(f"\nSaving top 10 best and worst plots to '{output_dir}' folder...")
    # for i, case in enumerate(top_10_best):
    #     case_data = (
    #     case['path'], case['positions'], case['charges'], case['prediction'], case['truth'], case['angle_error'])
    #     _save_plotly_figure(case_data, f"best_{i + 1:02d}", output_dir)
    # for i, case in enumerate(top_10_worst):
    #     case_data = (
    #     case['path'], case['positions'], case['charges'], case['prediction'], case['truth'], case['angle_error'])
    #     _save_plotly_figure(case_data, f"worst_{i + 1:02d}", output_dir)
    # print("Done.")