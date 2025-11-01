#!/usr/bin/env python3
"""
Generate index.html showing top 10 cosine similarity features for single_* datasets.
"""

import json
from pathlib import Path

# Define the datasets to include
SINGLE_DATASETS = [
    "apollo_probe",
    "single_convincing-game",
    "single_instructed-deception",
    "single_harm-pressure-choice",
    "single_harm-pressure-knowledge-report",
    "single_insider-trading_report",
    "single_insider-trading_confirmation",
]

def load_top_features(dataset_name):
    """Load top 10 positive features for a dataset."""
    json_path = Path(f"outputs/{dataset_name}/top_10_positive.json")
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return None

def generate_html():
    """Generate the HTML page."""

    # Load data for all datasets
    datasets_data = {}
    for dataset in SINGLE_DATASETS:
        data = load_top_features(dataset)
        if data:
            datasets_data[dataset] = data

    # Start HTML
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Top 10 SAE Features by Dataset</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
            background-color: #fafafa;
        }

        header {
            text-align: center;
            margin-bottom: 50px;
            padding-bottom: 30px;
            border-bottom: 2px solid #333;
        }

        h1 {
            font-size: 2.5em;
            font-weight: 400;
            margin-bottom: 10px;
            color: #222;
        }

        .subtitle {
            font-size: 1.1em;
            color: #666;
            font-style: italic;
        }

        nav {
            background-color: #fff;
            padding: 20px;
            margin-bottom: 40px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }

        nav h2 {
            font-size: 1.3em;
            font-weight: 400;
            margin-bottom: 15px;
            color: #444;
        }

        nav ul {
            list-style: none;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 10px;
        }

        nav a {
            color: #2c5282;
            text-decoration: none;
            padding: 8px 12px;
            display: block;
            border-left: 3px solid transparent;
            transition: all 0.2s;
        }

        nav a:hover {
            background-color: #f0f0f0;
            border-left-color: #2c5282;
        }

        section {
            background-color: #fff;
            margin-bottom: 40px;
            padding: 30px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }

        section h2 {
            font-size: 1.8em;
            font-weight: 400;
            margin-bottom: 20px;
            color: #222;
            padding-bottom: 10px;
            border-bottom: 1px solid #e0e0e0;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }

        thead {
            background-color: #f5f5f5;
        }

        th {
            text-align: left;
            padding: 12px;
            font-weight: 600;
            color: #444;
            border-bottom: 2px solid #ddd;
        }

        td {
            padding: 12px;
            border-bottom: 1px solid #eee;
        }

        tr:hover {
            background-color: #f9f9f9;
        }

        .rank {
            width: 60px;
            text-align: center;
            color: #666;
            font-weight: 500;
        }

        .feature-id {
            width: 120px;
            font-family: 'Courier New', monospace;
            color: #2c5282;
        }

        .similarity {
            width: 150px;
            font-family: 'Courier New', monospace;
            color: #2d7d2d;
            font-weight: 500;
        }

        .label {
            color: #555;
        }

        footer {
            text-align: center;
            margin-top: 60px;
            padding-top: 30px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 0.9em;
        }

        .back-to-top {
            display: inline-block;
            margin-top: 20px;
            padding: 10px 20px;
            background-color: #f5f5f5;
            color: #2c5282;
            text-decoration: none;
            border-radius: 4px;
            border: 1px solid #ddd;
            transition: all 0.2s;
        }

        .back-to-top:hover {
            background-color: #2c5282;
            color: #fff;
        }
    </style>
</head>
<body>
    <header>
        <h1>Top 10 SAE Features by Dataset</h1>
        <p class="subtitle">Cosine Similarity with Llama-3.3-70B-Instruct Layer 50</p>
    </header>

    <nav id="toc">
        <h2>Table of Contents</h2>
        <ul>
"""

    # Add TOC entries
    for dataset in SINGLE_DATASETS:
        if dataset in datasets_data:
            display_name = dataset.replace("single_", "").replace("-", " ").title()
            html += f'            <li><a href="#{dataset}">{display_name}</a></li>\n'

    html += """        </ul>
    </nav>

    <main>
"""

    # Add sections for each dataset
    for dataset in SINGLE_DATASETS:
        if dataset not in datasets_data:
            continue

        features = datasets_data[dataset]
        display_name = dataset.replace("single_", "").replace("-", " ").title()

        html += f"""        <section id="{dataset}">
            <h2>{display_name}</h2>
            <table>
                <thead>
                    <tr>
                        <th class="rank">Rank</th>
                        <th class="feature-id">Feature ID</th>
                        <th class="similarity">Cosine Similarity</th>
                        <th class="label">Label</th>
                    </tr>
                </thead>
                <tbody>
"""

        # Add rows
        for i, feature in enumerate(features[:10], 1):
            feat_id = feature['feature_id']
            similarity = feature['similarity']
            label = feature['label']

            html += f"""                    <tr>
                        <td class="rank">{i}</td>
                        <td class="feature-id">{feat_id}</td>
                        <td class="similarity">{similarity:+.6f}</td>
                        <td class="label">{label}</td>
                    </tr>
"""

        html += """                </tbody>
            </table>
            <a href="#toc" class="back-to-top">↑ Back to Top</a>
        </section>

"""

    # Close HTML
    html += """    </main>

    <footer>
        <p>Generated from SAE decoder features (Goodfire SAE for Llama-3.3-70B-Instruct, Layer 50)</p>
    </footer>
</body>
</html>
"""

    return html


def main():
    print("Generating index.html...")

    html = generate_html()

    output_path = Path("outputs/index.html")
    with open(output_path, "w") as f:
        f.write(html)

    print(f"\n✓ Saved to: {output_path}")
    print(f"\nOpen in browser: file://{output_path.absolute()}")


if __name__ == "__main__":
    main()
