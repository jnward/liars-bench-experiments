#!/usr/bin/env python3
"""
Generate compact HTML report showing top 2 positive similarities for single probes + all_datasets.

Creates a single table with all probes and their top 2 most aligned deception features.
"""

import json
from pathlib import Path


def get_color_for_similarity(sim: float) -> str:
    """Get background color based on similarity score."""
    # Green gradient for positive similarities
    intensity = min(sim / 0.15, 1.0)
    green_val = int(200 + intensity * 55)  # 200-255
    red_val = int(240 - intensity * 90)    # 240-150
    blue_val = int(240 - intensity * 90)   # 240-150
    return f"rgb({red_val}, {green_val}, {blue_val})"


def main():
    base_dir = Path("outputs/deception_features")
    output_file = base_dir / "deception_similarities_compact.html"

    # Define probes to include (apollo_probe + all_datasets + single_*)
    probe_names = ['apollo_probe', 'all_datasets']

    # Get all single probes
    single_probes = [
        'single_convincing-game',
        'single_harm-pressure-choice',
        'single_harm-pressure-knowledge-report',
        'single_insider-trading_confirmation',
        'single_insider-trading_report',
        'single_instructed-deception'
    ]
    probe_names.extend(sorted(single_probes))

    print("="*80)
    print("GENERATING COMPACT DECEPTION SIMILARITIES REPORT")
    print("="*80)
    print(f"\nIncluding {len(probe_names)} probes:")
    for name in probe_names:
        print(f"  - {name}")

    # Collect top 2 features for each probe
    all_rows = []

    for probe_name in probe_names:
        probe_dir = base_dir / probe_name

        # Load top positive similarities
        with open(probe_dir / "top_10_positive.json", 'r') as f:
            top_positive = json.load(f)

        # Get top 2
        for rank, item in enumerate(top_positive[:2], 1):
            all_rows.append({
                'probe_name': probe_name,
                'rank': rank,
                'feature_id': item['feature_id'],
                'similarity': item['similarity'],
                'label': item['label']
            })

        print(f"  ✓ Loaded top 2 for {probe_name}")

    # Generate HTML
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deception Features - Top 2 Similarities</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1600px;
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

        .content {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .compact-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }

        .compact-table thead {
            background-color: #34495e;
            color: white;
        }

        .compact-table th {
            padding: 15px 12px;
            text-align: left;
            font-weight: 600;
            font-size: 1.05em;
        }

        .compact-table td {
            padding: 12px;
            border: 1px solid #ddd;
        }

        .compact-table td:nth-child(1) {
            font-family: 'Courier New', monospace;
            font-weight: 600;
            width: 280px;
        }

        .compact-table td:nth-child(2) {
            text-align: center;
            font-weight: bold;
            width: 60px;
        }

        .compact-table td:nth-child(3) {
            text-align: center;
            font-family: 'Courier New', monospace;
            width: 100px;
        }

        .compact-table td:nth-child(4) {
            text-align: center;
            font-family: 'Courier New', monospace;
            width: 110px;
            font-weight: 600;
            font-size: 1.05em;
        }

        .compact-table td:nth-child(5) {
            text-align: left;
        }

        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-style: italic;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Top 2 Deception Feature Similarities</h1>
        <p><strong>Analysis:</strong> Most aligned deception features for single-dataset probes and all_datasets</p>
        <p><strong>Model:</strong> meta-llama/Llama-3.3-70B-Instruct (Layer 50)</p>
        <p><strong>Probes:</strong> 8 (1 apollo + 1 all_datasets + 6 single-dataset)</p>
    </div>

    <div class="content">
        <table class="compact-table">
            <thead>
                <tr>
                    <th>Probe Name</th>
                    <th>Rank</th>
                    <th>Feature ID</th>
                    <th>Similarity</th>
                    <th>Feature Label</th>
                </tr>
            </thead>
            <tbody>
"""

    # Add all rows
    for row in all_rows:
        color = get_color_for_similarity(row['similarity'])
        html += f"""
                <tr style="background-color: {color};">
                    <td>{row['probe_name']}</td>
                    <td>{row['rank']}</td>
                    <td>{row['feature_id']}</td>
                    <td>{row['similarity']:+.4f}</td>
                    <td>{row['label']}</td>
                </tr>
"""

    html += """
            </tbody>
        </table>
    </div>

    <div class="footer">
        <p>Top 2 positive cosine similarities between probes and 71 deception-labeled SAE features</p>
        <p>Color intensity indicates strength of similarity (darker green = stronger alignment)</p>
    </div>
</body>
</html>
"""

    # Save HTML file
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"\n{'='*80}")
    print(f"✓ Compact report generated successfully!")
    print(f"{'='*80}")
    print(f"\nSaved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")
    print(f"Total rows: {len(all_rows)} (2 per probe × {len(probe_names)} probes)")
    print(f"\nOpen in browser to view the report.")


if __name__ == "__main__":
    main()
