#!/usr/bin/env python3
"""
Generate scientific HTML report showing cosine similarities between probes and deception features.

Creates a clean, academic-style HTML report with:
- Summary statistics for each probe
- Top 10 positive similarities (most aligned)
- Top 10 negative similarities (most anti-aligned)
- Color-coded similarity scores
"""

import json
from pathlib import Path


def get_color_for_similarity(sim: float, is_positive_table: bool) -> str:
    """Get background color based on similarity score.

    Args:
        sim: Similarity score
        is_positive_table: Whether this is for the positive or negative table

    Returns:
        RGB color string
    """
    if is_positive_table:
        # Green gradient for positive similarities (lighter to darker green)
        # Map from 0 to max_expected (0.15) to color intensity
        intensity = min(sim / 0.15, 1.0)
        green_val = int(200 + intensity * 55)  # 200-255
        red_val = int(240 - intensity * 90)    # 240-150
        blue_val = int(240 - intensity * 90)   # 240-150
        return f"rgb({red_val}, {green_val}, {blue_val})"
    else:
        # Red gradient for negative similarities (lighter to darker red)
        intensity = min(abs(sim) / 0.15, 1.0)
        red_val = int(200 + intensity * 55)    # 200-255
        green_val = int(240 - intensity * 90)  # 240-150
        blue_val = int(240 - intensity * 90)   # 240-150
        return f"rgb({red_val}, {green_val}, {blue_val})"


def generate_probe_section(probe_name: str, probe_dir: Path) -> str:
    """Generate HTML section for a single probe."""
    # Load data
    with open(probe_dir / "top_10_positive.json", 'r') as f:
        top_positive = json.load(f)

    with open(probe_dir / "top_10_negative.json", 'r') as f:
        top_negative = json.load(f)

    with open(probe_dir / "summary.json", 'r') as f:
        summary = json.load(f)

    stats = summary['statistics']

    # Generate HTML
    html = f"""
    <div class="probe-section">
        <h2>{probe_name}</h2>

        <div class="statistics">
            <table class="stats-table">
                <tr>
                    <th>Mean Similarity</th>
                    <th>Std Dev</th>
                    <th>Max Similarity</th>
                    <th>Min Similarity</th>
                    <th>Median</th>
                </tr>
                <tr>
                    <td>{stats['mean_similarity']:+.4f}</td>
                    <td>{stats['std_similarity']:.4f}</td>
                    <td>{stats['max_similarity']:+.4f}</td>
                    <td>{stats['min_similarity']:+.4f}</td>
                    <td>{stats['median_similarity']:+.4f}</td>
                </tr>
            </table>
        </div>

        <div class="similarity-tables">
            <div class="table-container">
                <h3>Top 10 Positive Similarities</h3>
                <p class="table-description">Features most aligned with probe direction</p>
                <table class="similarity-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Feature ID</th>
                            <th>Similarity</th>
                            <th>Label</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    # Add positive similarities
    for i, item in enumerate(top_positive, 1):
        color = get_color_for_similarity(item['similarity'], is_positive_table=True)
        html += f"""
                        <tr style="background-color: {color};">
                            <td>{i}</td>
                            <td>{item['feature_id']}</td>
                            <td><strong>{item['similarity']:+.4f}</strong></td>
                            <td>{item['label']}</td>
                        </tr>
"""

    html += """
                    </tbody>
                </table>
            </div>

            <div class="table-container">
                <h3>Top 10 Negative Similarities</h3>
                <p class="table-description">Features most anti-aligned with probe direction</p>
                <table class="similarity-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Feature ID</th>
                            <th>Similarity</th>
                            <th>Label</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    # Add negative similarities
    for i, item in enumerate(top_negative, 1):
        color = get_color_for_similarity(item['similarity'], is_positive_table=False)
        html += f"""
                        <tr style="background-color: {color};">
                            <td>{i}</td>
                            <td>{item['feature_id']}</td>
                            <td><strong>{item['similarity']:+.4f}</strong></td>
                            <td>{item['label']}</td>
                        </tr>
"""

    html += """
                    </tbody>
                </table>
            </div>
        </div>
    </div>
"""

    return html


def main():
    base_dir = Path("outputs/deception_features")
    output_file = base_dir / "deception_similarities_report.html"

    # Load overall summary
    with open(base_dir / "summary.json", 'r') as f:
        overall_summary = json.load(f)

    # Group probes by type
    apollo_probes = []
    all_datasets_probes = []
    single_probes = []
    leaveout_probes = []

    for probe_info in overall_summary['probes_ranked_by_mean_similarity']:
        probe_name = probe_info['probe_name']
        if probe_name == 'apollo_probe':
            apollo_probes.append(probe_name)
        elif probe_name == 'all_datasets':
            all_datasets_probes.append(probe_name)
        elif probe_name.startswith('single_'):
            single_probes.append(probe_name)
        elif probe_name.startswith('leaveout_'):
            leaveout_probes.append(probe_name)

    # Check if apollo_probe exists (not in summary.json yet)
    if (base_dir / "apollo_probe").exists() and not apollo_probes:
        apollo_probes.append('apollo_probe')

    # Sort within groups
    single_probes.sort()
    leaveout_probes.sort()

    # Combine in order (apollo first)
    ordered_probes = apollo_probes + all_datasets_probes + single_probes + leaveout_probes

    print("="*80)
    print("GENERATING DECEPTION SIMILARITIES REPORT")
    print("="*80)
    print(f"\nProcessing {len(ordered_probes)} probes:")
    print(f"  - apollo_probe: {len(apollo_probes)}")
    print(f"  - all_datasets: {len(all_datasets_probes)}")
    print(f"  - single_*: {len(single_probes)}")
    print(f"  - leaveout_*: {len(leaveout_probes)}")

    # Generate HTML
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deception Feature Similarities Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }

        .header {
            background-color: #2c3e50;
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }

        .header h1 {
            margin: 0 0 10px 0;
            font-size: 2em;
        }

        .header p {
            margin: 5px 0;
            font-size: 1.1em;
            opacity: 0.9;
        }

        .probe-section {
            background-color: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .probe-section h2 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
            font-family: 'Courier New', monospace;
        }

        .statistics {
            margin: 20px 0;
        }

        .stats-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }

        .stats-table th {
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: center;
            font-weight: 600;
        }

        .stats-table td {
            padding: 12px;
            text-align: center;
            border: 1px solid #ddd;
            font-family: 'Courier New', monospace;
            font-size: 1.1em;
        }

        .similarity-tables {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 30px;
        }

        .table-container h3 {
            color: #2c3e50;
            margin-top: 0;
            font-size: 1.3em;
        }

        .table-description {
            color: #666;
            font-style: italic;
            margin: 5px 0 15px 0;
        }

        .similarity-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95em;
        }

        .similarity-table thead {
            background-color: #34495e;
            color: white;
        }

        .similarity-table th {
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
        }

        .similarity-table td {
            padding: 10px 8px;
            border: 1px solid #ddd;
        }

        .similarity-table td:nth-child(1) {
            text-align: center;
            font-weight: bold;
            width: 50px;
        }

        .similarity-table td:nth-child(2) {
            text-align: center;
            font-family: 'Courier New', monospace;
            width: 100px;
        }

        .similarity-table td:nth-child(3) {
            text-align: center;
            font-family: 'Courier New', monospace;
            width: 100px;
        }

        .similarity-table td:nth-child(4) {
            text-align: left;
        }

        .group-header {
            background-color: #3498db;
            color: white;
            padding: 15px 30px;
            margin: 30px 0 20px 0;
            border-radius: 5px;
            font-size: 1.3em;
            font-weight: 600;
        }

        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-style: italic;
            margin-top: 40px;
        }

        @media (max-width: 1200px) {
            .similarity-tables {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Deception Feature Similarities Report</h1>
        <p><strong>Analysis:</strong> Cosine similarities between probes and 71 deception-labeled SAE features</p>
        <p><strong>Model:</strong> meta-llama/Llama-3.3-70B-Instruct (Layer 50)</p>
        <p><strong>Features:</strong> 71 deception-related features from Goodfire SAE</p>
        <p><strong>Probes analyzed:</strong> 14 (1 apollo + 1 all_datasets + 6 single + 6 leaveout)</p>
    </div>
"""

    # Generate sections for apollo_probe
    if apollo_probes:
        html += '<div class="group-header">Apollo Probe</div>\n'
        for probe_name in apollo_probes:
            probe_dir = base_dir / probe_name
            html += generate_probe_section(probe_name, probe_dir)
            print(f"  ✓ Generated section for {probe_name}")

    # Generate sections for all_datasets probes
    if all_datasets_probes:
        html += '<div class="group-header">All Datasets Probe</div>\n'
        for probe_name in all_datasets_probes:
            probe_dir = base_dir / probe_name
            html += generate_probe_section(probe_name, probe_dir)
            print(f"  ✓ Generated section for {probe_name}")

    # Generate sections for single probes
    if single_probes:
        html += '<div class="group-header">Single-Dataset Probes</div>\n'
        for probe_name in single_probes:
            probe_dir = base_dir / probe_name
            html += generate_probe_section(probe_name, probe_dir)
            print(f"  ✓ Generated section for {probe_name}")

    # Generate sections for leaveout probes
    if leaveout_probes:
        html += '<div class="group-header">Leave-One-Out Probes</div>\n'
        for probe_name in leaveout_probes:
            probe_dir = base_dir / probe_name
            html += generate_probe_section(probe_name, probe_dir)
            print(f"  ✓ Generated section for {probe_name}")

    # Add footer
    html += """
    <div class="footer">
        <p>Generated from deception feature similarity analysis</p>
        <p>Color intensity indicates strength of similarity (darker = stronger)</p>
    </div>
</body>
</html>
"""

    # Save HTML file
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"\n{'='*80}")
    print(f"✓ Report generated successfully!")
    print(f"{'='*80}")
    print(f"\nSaved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")
    print(f"\nOpen in browser to view the report.")


if __name__ == "__main__":
    main()
