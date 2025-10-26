"""
Create comparison visualizations for score_based (v1) vs score_based_v2 (v2) across all datasets.
Compares all 4 variants: v1 full, v1 final, v2 full, v2 final.

Generates:
1. Faceted plots - one subplot per metric
2. Heatmap - metric × dataset grid
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Available datasets
DATASETS = [
    "insider-trading",
    "convincing-game",
    "instructed-deception",
    "harm-pressure-choice",
    "harm-pressure-knowledge-report",
    "soft-trigger",
]

# Prompt types to compare (all 4 variants)
PROMPT_TYPES = [
    "score_based_full_conversation",
    "score_based_final_only",
    "score_based_v2_full_conversation",
    "score_based_v2_final_only",
]

METRICS = ["accuracy", "precision", "recall", "f1", "auroc"]


def compute_metrics(predictions, labels):
    """Compute metrics from predictions and labels."""
    predictions_array = np.array(predictions)
    binary_predictions = (predictions_array >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, binary_predictions),
        "precision": precision_score(labels, binary_predictions, zero_division=0),
        "recall": recall_score(labels, binary_predictions, zero_division=0),
        "f1": f1_score(labels, binary_predictions, zero_division=0),
    }

    try:
        metrics["auroc"] = roc_auc_score(labels, predictions_array)
    except:
        metrics["auroc"] = None

    return metrics


def load_results(dataset, prompt_type):
    """Load results for a specific dataset and prompt type."""
    results_path = Path("results-qwq-32b") / dataset / prompt_type / "results.json"

    if not results_path.exists():
        return None

    with open(results_path, 'r') as f:
        results = json.load(f)

    # Filter valid results
    valid_results = [r for r in results if r.get("prediction") is not None and r.get("prediction") != -1]

    if len(valid_results) == 0:
        return None

    predictions = [r["prediction"] for r in valid_results]
    labels = [r["true_label"] for r in valid_results]

    return compute_metrics(predictions, labels)


def collect_all_data():
    """Collect metrics for all dataset × prompt combinations."""
    data = []

    for dataset in DATASETS:
        for prompt_type in PROMPT_TYPES:
            metrics = load_results(dataset, prompt_type)
            if metrics:
                # Extract prompt version for cleaner labels
                if "v2_final_only" in prompt_type:
                    prompt_label = "v2 (final)"
                elif "v2" in prompt_type:
                    prompt_label = "v2 (full)"
                elif "final_only" in prompt_type:
                    prompt_label = "v1 (final)"
                else:
                    prompt_label = "v1 (full)"

                for metric_name, metric_value in metrics.items():
                    if metric_value is not None:
                        data.append({
                            "dataset": dataset,
                            "prompt": prompt_label,
                            "metric": metric_name,
                            "value": metric_value
                        })

    return pd.DataFrame(data)


def create_faceted_plots(df, output_path):
    """Create faceted plots - one subplot per metric."""
    metrics = sorted(df['metric'].unique())
    datasets = df['dataset'].unique()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = {
        'v1 (full)': '#3498db',
        'v1 (final)': '#5dade2',
        'v2 (full)': '#e74c3c',
        'v2 (final)': '#ec7063'
    }

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_df = df[df['metric'] == metric]

        # Create grouped bar chart for this metric
        x = np.arange(len(datasets))
        width = 0.18  # Narrower bars to fit 4 variants
        prompts = sorted(metric_df['prompt'].unique())

        for i, prompt in enumerate(prompts):
            prompt_data = metric_df[metric_df['prompt'] == prompt]
            values = []
            for dataset in datasets:
                val = prompt_data[prompt_data['dataset'] == dataset]['value'].values
                values.append(val[0] if len(val) > 0 else 0)

            # Center the bars around each dataset tick
            offset = (i - len(prompts)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=prompt,
                   color=colors.get(prompt, 'gray'), alpha=0.8)

        ax.set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([d.replace('-', '\n') for d in datasets], fontsize=8, rotation=45, ha='right')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        if idx == 0:
            ax.legend(fontsize=9)

    # Hide the extra subplot
    if len(metrics) < 6:
        axes[5].set_visible(False)

    plt.suptitle('Score-Based Prompt Comparison by Metric', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_heatmap(df, output_path):
    """Create heatmap showing metric × (dataset, prompt) grid."""
    # Pivot data for heatmap
    datasets = df['dataset'].unique()
    prompts = sorted(df['prompt'].unique())
    metrics = sorted(df['metric'].unique())

    # Create multi-index columns
    heatmap_data = []
    row_labels = []

    for metric in metrics:
        row = []
        for dataset in datasets:
            for prompt in prompts:
                val = df[(df['metric'] == metric) &
                        (df['dataset'] == dataset) &
                        (df['prompt'] == prompt)]['value'].values
                row.append(val[0] if len(val) > 0 else np.nan)
        heatmap_data.append(row)
        row_labels.append(metric.upper())

    # Create column labels
    col_labels = []
    for dataset in datasets:
        for prompt in prompts:
            col_labels.append(f"{dataset}\n{prompt}")

    # Create heatmap
    fig, ax = plt.subplots(figsize=(20, 6))

    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(row_labels, fontsize=11, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=20, fontsize=11, fontweight='bold')

    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(col_labels)):
            value = heatmap_data[i][j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.3f}',
                             ha="center", va="center", color="black" if value > 0.5 else "white",
                             fontsize=8, fontweight='bold')

    ax.set_title('Score-Based Prompt Comparison Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    print("="*80)
    print("Creating comparison visualizations...")
    print("="*80)

    # Collect data
    print("\nCollecting data from results directories...")
    df = collect_all_data()

    if df.empty:
        print("⚠ No data found. Make sure you have run experiments first.")
        return

    print(f"✓ Found data for {len(df['dataset'].unique())} datasets and {len(df['prompt'].unique())} prompt types")

    # Create output directory
    output_dir = Path("results-qwq-32b")
    output_dir.mkdir(exist_ok=True)

    # Create visualizations
    print("\nGenerating visualizations...")
    create_faceted_plots(df, output_dir / "comparison_faceted.png")
    create_heatmap(df, output_dir / "comparison_heatmap.png")

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()
