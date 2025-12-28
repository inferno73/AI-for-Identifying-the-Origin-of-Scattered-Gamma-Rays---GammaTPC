import numpy as np

def generate_report(results, energy_folders, version_name="V10"):
    """
    Takes a list of track results and generates a detailed statistical report
    on the angle errors, with enhanced cumulative breakdowns.

    Args:
        results (list): A list of dictionaries, where each must contain at least
                        'angle_error' and 'energy' keys.
        energy_folders (list): A list of the energy folder names (as strings)
                               that were analyzed, e.g., ['1000', '3000'].
        version_name (str): A name for this analysis run.

    Returns:
        str: A formatted string containing the full report.
    """
    if not results:
        return "No results to analyze."

    report_lines = []

    # --- Overall Statistics ---
    all_angle_errors = np.array([r['angle_error'] for r in results])
    total_tracks = len(all_angle_errors)

    report_lines.append("=" * 60)
    report_lines.append(f"Detailed Angle Error Analysis ({version_name})")
    report_lines.append(f"Analyzed {total_tracks} tracks from energy folders: {energy_folders}")
    report_lines.append("=" * 60 + "\n")

    report_lines.append("1. Overall Performance Summary:")
    report_lines.append(f"   - Mean:   {np.mean(all_angle_errors):.2f}°")
    report_lines.append(f"   - Median: {np.median(all_angle_errors):.2f}°")
    report_lines.append(f"   - Std Dev:{np.std(all_angle_errors):.2f}°")
    report_lines.append(f"   - Min Error (Best Track):  {np.min(all_angle_errors):.2f}°")
    report_lines.append(f"   - Max Error (Worst Track): {np.max(all_angle_errors):.2f}°\n")

    # --- Per-Energy Breakdown ---
    report_lines.append("2. Performance by Energy Level:")

    errors_by_energy = {int(e): [] for e in energy_folders}
    for res in results:
        if res['energy'] in errors_by_energy:
            errors_by_energy[res['energy']].append(res['angle_error'])

    for energy_kev, errors in sorted(errors_by_energy.items()):
        if errors:
            mean_err, median_err = np.mean(errors), np.median(errors)
            energy_mev = energy_kev / 1000.0
            report_lines.append(
                f"   - {energy_mev:.0f} MeV ({len(errors)} tracks): Mean = {mean_err:.2f}°, Median = {median_err:.2f}°")
        else:
            report_lines.append(f"   - {energy_kev / 1000.0:.0f} MeV: No tracks found in results.")
    report_lines.append("\n")

    # --- Cumulative Bins (Overall) ---
    report_lines.append("3. Cumulative Accuracy Distribution (Overall):")
    thresholds = [1.0, 2.0, 5.0, 7.0, 10.0, 15.0, 20.0]

    for thresh in thresholds:
        count = np.sum(all_angle_errors <= thresh)
        percent = (count / total_tracks) * 100 if total_tracks > 0 else 0
        report_lines.append(f"   - Tracks with error <= {thresh:.1f}°: {count} ({percent:.2f}%)")

    # --- Add the "Above 20" category ---
    last_threshold = thresholds[-1]
    count_above = np.sum(all_angle_errors > last_threshold)
    percent_above = (count_above / total_tracks) * 100 if total_tracks > 0 else 0
    report_lines.append(f"   - Tracks with error > {last_threshold:.1f}°: {count_above} ({percent_above:.2f}%)")
    report_lines.append("\n")

    # --- Cumulative Bins by Energy with updated percentage ---
    report_lines.append("4. Cumulative Accuracy Distribution (Composition by Energy):")

    for thresh in thresholds:
        # Find all results that meet the current threshold
        results_in_bin = [r for r in results if r['angle_error'] <= thresh]
        total_in_bin = len(results_in_bin)

        first_line_prefix = f"   - Tracks with error <= {thresh:.1f}° ({total_in_bin} total):"

        if total_in_bin == 0:
            report_lines.append(f"{first_line_prefix:<45} No tracks in this bin.")
            report_lines.append("")  # Add a blank line for readability
            continue

        # Count how many tracks of each energy type are in this bin
        counts_in_bin_by_energy = {int(e): 0 for e in energy_folders}
        for res in results_in_bin:
            if res['energy'] in counts_in_bin_by_energy:
                counts_in_bin_by_energy[res['energy']] += 1

        # Format the output lines
        for i, energy_kev in enumerate(sorted(counts_in_bin_by_energy.keys())):
            count_for_energy = counts_in_bin_by_energy[energy_kev]
            percentage = (count_for_energy / total_in_bin) * 100

            energy_mev = energy_kev / 1000.0
            line_str = f"{energy_mev:.0f} MeV ({count_for_energy} tracks, {percentage:.2f}%)"

            if i == 0:
                report_lines.append(f"{first_line_prefix:<45} {line_str}")
            else:
                report_lines.append(f"{'':<45} {line_str}")
        report_lines.append("")

    return "\n".join(report_lines)