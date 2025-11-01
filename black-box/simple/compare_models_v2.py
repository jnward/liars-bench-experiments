#!/usr/bin/env python3
"""
Compare AUROC across different models on score_based_v2 prompts.
Creates separate figures for final_only and full_conversation modes.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Model directories and their display names
MODELS = {
    "results": "Claude Sonnet 4.5",
    "results-gemma-3-27b": "Gemma 3 27B",
    "results-qwq-32b": "QwQ 32B",
}

# Datasets to compare
DATASETS = [
    "insider-trading",
    "convincing-game",
    "instructed-deception",
    "soft-trigger",
]

# Prompt modes to compare
MODES = {
    "score_based_v2_final_only": "V2 Final Only",
    "score_based_v2_full_conversation": "V2 Full Conversation",
}


def load_auroc(model_dir, dataset, mode):
    """Load results and compute AUROC for a specific model/dataset/mode."""
    results_path = Path(model_dir) / dataset / mode / "results.json"

    if not results_path.exists():
        return None

    try:
        with open(results_path, 'r') as f:
            results = json.load(f)

        # Filter valid results
        valid_results = [r for r in results
                        if r.get("prediction") is not None and r.get("prediction") != -1]

        if len(valid_results) == 0:
            return None

        predictions = np.array([r["prediction"] for r in valid_results])
        labels = np.array([r["true_label"] for r in valid_results])

        # Compute AUROC
        auroc = roc_auc_score(labels, predictions)
        return auroc

    except Exception as e:
        print(f"⚠ Error loading {model_dir}/{dataset}/{mode}: {e}")
        return None


def collect_data():
    """Collect AUROC data for all model/dataset/mode combinations."""
    data = {}

    for mode_key, mode_label in MODES.items():
        data[mode_key] = {}

        for dataset in DATASETS:
            data[mode_key][dataset] = {}

            for model_dir, model_name in MODELS.items():
                auroc = load_auroc(model_dir, dataset, mode_key)
                if auroc is not None:
                    data[mode_key][dataset][model_name] = auroc

    return data


def create_comparison_plot(data, mode_key, mode_label, output_path):
    """Create multi-subplot figure comparing models for a specific mode."""
    datasets_with_data = [d for d in DATASETS if data[mode_key].get(d)]

    if not datasets_with_data:
        print(f"⚠ No data found for {mode_label}")
        return

    # Create figure with subplots
    n_datasets = len(datasets_with_data)
    n_cols = 2
    n_rows = (n_datasets + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # Color scheme for models
    colors = {
        "Claude Sonnet 4.5": "#e74c3c",
        "Gemma 3 27B": "#3498db",
        "QwQ 32B": "#2ecc71",
    }

    for idx, dataset in enumerate(datasets_with_data):
        ax = axes[idx]
        dataset_data = data[mode_key][dataset]

        # Prepare data for this dataset
        models = list(dataset_data.keys())
        aurocs = [dataset_data[m] for m in models]

        # Create bar chart
        x = np.arange(len(models))
        bars = ax.bar(x, aurocs, color=[colors.get(m, 'gray') for m in models],
                      alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for i, (bar, auroc) in enumerate(zip(bars, aurocs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{auroc:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Styling
        ax.set_title(dataset.replace('-', ' ').title(),
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10, rotation=15, ha='right')
        ax.set_ylabel('AUROC', fontsize=11, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # Hide extra subplots if odd number of datasets
    for idx in range(len(datasets_with_data), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Model Comparison - {mode_label}',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def print_summary(data):
    """Print summary table of all results."""
    print("\n" + "="*80)
    print("SUMMARY: AUROC by Model and Mode")
    print("="*80)

    for mode_key, mode_label in MODES.items():
        print(f"\n{mode_label}:")
        print("-" * 80)

        for dataset in DATASETS:
            dataset_data = data[mode_key].get(dataset, {})
            if not dataset_data:
                continue

            print(f"\n  {dataset}:")
            for model_name in MODELS.values():
                auroc = dataset_data.get(model_name)
                if auroc is not None:
                    print(f"    {model_name:20s}: {auroc:.4f}")
                else:
                    print(f"    {model_name:20s}: N/A")


def main():
    print("="*80)
    print("Model Comparison: Score-Based V2 Prompts")
    print("="*80)

    # Collect all data
    print("\nCollecting AUROC data from all models...")
    data = collect_data()

    # Print summary
    print_summary(data)

    # Create plots for each mode
    print("\n" + "="*80)
    print("Creating comparison plots...")
    print("="*80)

    for mode_key, mode_label in MODES.items():
        output_file = f"model_comparison_{mode_key}.png"
        create_comparison_plot(data, mode_key, mode_label, output_file)

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()
