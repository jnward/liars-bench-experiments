#!/usr/bin/env python3
"""
Generate academic-style HTML report from probe-SAE comparison results.
"""

import json
from pathlib import Path


def generate_html():
    """Generate complete HTML report."""

    datasets = ['liars-bench__convincing-game', 'liars-bench__hpkr']
    layers = [3, 7, 11, 15, 19, 23]

    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT-OSS-20B Probe-SAE Feature Analysis</title>
    <style>
        body {
            font-family: 'Georgia', 'Times New Roman', serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
            background-color: #ffffff;
            color: #333;
            line-height: 1.6;
        }

        h1 {
            text-align: center;
            font-size: 2em;
            margin-bottom: 10px;
            color: #1a1a1a;
        }

        .subtitle {
            text-align: center;
            font-size: 1.1em;
            color: #666;
            margin-bottom: 40px;
            font-style: italic;
        }

        h2 {
            font-size: 1.5em;
            margin-top: 50px;
            margin-bottom: 20px;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
            color: #1a1a1a;
        }

        h3 {
            font-size: 1.2em;
            margin-top: 30px;
            margin-bottom: 15px;
            color: #444;
        }

        .dataset-section {
            margin-bottom: 60px;
        }

        .layer-section {
            margin-bottom: 40px;
            page-break-inside: avoid;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
            font-size: 0.95em;
        }

        th {
            background-color: #f5f5f5;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #333;
            font-weight: 600;
        }

        td {
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }

        tr:hover {
            background-color: #f9f9f9;
        }

        .feature-id {
            font-family: 'Courier New', monospace;
            color: #0066cc;
            text-decoration: none;
            font-weight: 500;
        }

        .feature-id:hover {
            text-decoration: underline;
        }

        .similarity-positive {
            color: #2d7a2d;
            font-weight: 600;
        }

        .similarity-negative {
            color: #c41e3a;
            font-weight: 600;
        }

        .section-header {
            background-color: #e8f4f8;
            padding: 8px 12px;
            font-weight: 600;
            color: #2c5f7c;
            border-left: 4px solid #2c5f7c;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        .rank {
            color: #999;
            font-weight: 500;
            width: 40px;
        }

        .label {
            color: #444;
        }
    </style>
</head>
<body>
    <h1>GPT-OSS-20B Probe-SAE Feature Analysis</h1>
    <p class="subtitle">Cosine Similarity Analysis Between Probe Directions and Sparse Autoencoder Features</p>
"""

    for dataset in datasets:
        html += f'\n    <div class="dataset-section">\n'
        html += f'        <h2>Dataset: {dataset}</h2>\n'

        for layer in layers:
            summary_path = Path(f'gpt_oss_20b_saes/results/{dataset}/layer{layer:02d}/summary.json')

            with open(summary_path) as f:
                data = json.load(f)

            top_positive = data['top_10_positive']
            top_negative = data['top_10_negative']

            html += f'\n        <div class="layer-section">\n'
            html += f'            <h3>Layer {layer}</h3>\n\n'

            # Positive features table
            html += '            <div class="section-header">Top 10 Positive Features (Most Aligned)</div>\n'
            html += '            <table>\n'
            html += '                <thead>\n'
            html += '                    <tr>\n'
            html += '                        <th style="width: 50px;">Rank</th>\n'
            html += '                        <th style="width: 120px;">Feature ID</th>\n'
            html += '                        <th style="width: 120px;">Similarity</th>\n'
            html += '                        <th>Label</th>\n'
            html += '                    </tr>\n'
            html += '                </thead>\n'
            html += '                <tbody>\n'

            for rank, feature in enumerate(top_positive, 1):
                fid = feature['feature_id']
                sim = feature['similarity']
                label = feature['label']
                url = f"https://neuronpedia.org/gpt-oss-20b/{layer}-resid-post-aa/{fid}"
                html += f'                    <tr><td class="rank">{rank}</td><td><a href="{url}" class="feature-id" target="_blank">{fid}</a></td><td class="similarity-positive">{sim:+.4f}</td><td class="label">{label}</td></tr>\n'

            html += '                </tbody>\n'
            html += '            </table>\n\n'

            # Negative features table
            html += '            <div class="section-header">Top 10 Negative Features (Most Anti-Correlated)</div>\n'
            html += '            <table>\n'
            html += '                <thead>\n'
            html += '                    <tr>\n'
            html += '                        <th style="width: 50px;">Rank</th>\n'
            html += '                        <th style="width: 120px;">Feature ID</th>\n'
            html += '                        <th style="width: 120px;">Similarity</th>\n'
            html += '                        <th>Label</th>\n'
            html += '                    </tr>\n'
            html += '                </thead>\n'
            html += '                <tbody>\n'

            for rank, feature in enumerate(top_negative, 1):
                fid = feature['feature_id']
                sim = feature['similarity']
                label = feature['label']
                url = f"https://neuronpedia.org/gpt-oss-20b/{layer}-resid-post-aa/{fid}"
                html += f'                    <tr><td class="rank">{rank}</td><td><a href="{url}" class="feature-id" target="_blank">{fid}</a></td><td class="similarity-negative">{sim:+.4f}</td><td class="label">{label}</td></tr>\n'

            html += '                </tbody>\n'
            html += '            </table>\n'
            html += '        </div>\n'

        html += '    </div>\n'

    html += '\n</body>\n</html>'

    return html


def main():
    print("Generating HTML report...")
    html = generate_html()

    output_path = Path("gpt_oss_20b_saes/results/index.html")
    output_path.write_text(html, encoding='utf-8')

    print(f"✓ HTML report generated: {output_path}")
    print(f"  File size: {len(html) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
