"""
Create visualizations from existing results JSON files.
Usage: python create_visualizations.py results/binary_classifier/convincing-game_results.json
"""

import sys
import json
from pathlib import Path
from black_box_detector import create_visualizations

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_visualizations.py <results_json_path>")
        sys.exit(1)

    results_path = Path(sys.argv[1])
    if not results_path.exists():
        print(f"Error: {results_path} does not exist")
        sys.exit(1)

    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Get unique dataset names
    dataset_names = sorted(set(r["dataset"] for r in results))

    # Output to same directory as JSON
    output_dir = results_path.parent

    print(f"Creating visualizations for {results_path}...")
    print(f"Output directory: {output_dir}")
    print(f"Datasets: {', '.join(dataset_names)}")

    create_visualizations(results, output_dir, dataset_names)
