#!/usr/bin/env python3
"""
Create violin plots and AUROC bar chart for probe results.

Usage:
    python plot_probe_results.py --csv scores-qwen-baseline.csv --output-dir plots-baseline
    python plot_probe_results.py --csv scores-qwen-org.csv --output-dir plots-organism
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import numpy as np


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Plot probe results')

    parser.add_argument(
        '--csv', '-c',
        type=str,
        required=True,
        help='Path to scores CSV file'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        required=True,
        help='Output directory for plots'
    )

    return parser.parse_args()


def create_violin_plots(df: pd.DataFrame, output_dir: Path):
    """Create violin plots for all probes."""
    # Get all probe columns
    probe_cols = [col for col in df.columns if col.startswith('logreg_') or col.startswith('repe_')]

    # Determine grid size
    n_probes = len(probe_cols)
    n_cols = 3
    n_rows = int(np.ceil(n_probes / n_cols))

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_probes > 1 else [axes]

    # Color mapping
    colors = {0: 'green', 1: 'red'}  # 0=honest, 1=deceptive

    for idx, probe_col in enumerate(probe_cols):
        ax = axes[idx]

        # Prepare data
        honest_scores = df[df['ground_truth'] == 0][probe_col].values
        deceptive_scores = df[df['ground_truth'] == 1][probe_col].values

        # Create violin plot
        parts = ax.violinplot(
            [honest_scores, deceptive_scores],
            positions=[1, 2],
            showmeans=True,
            showmedians=True
        )

        # Color the violin plots
        for i, pc in enumerate(parts['bodies']):
            color = 'green' if i == 0 else 'red'
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        # Customize plot
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Honest', 'Deceptive'])
        ax.set_ylabel('Probe Score')
        # Replace logreg_ with LR: but replace repe_* entirely with just Apollo
        if probe_col.startswith('repe_'):
            title = 'Apollo'
        else:
            title = probe_col.replace('logreg_', 'LR: ')
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_probes, len(axes)):
        axes[idx].axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='Honest'),
        Patch(facecolor='red', alpha=0.6, label='Deceptive')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.suptitle('Probe Score Distributions', fontsize=16, y=0.995)
    plt.tight_layout()

    # Save
    output_path = output_dir / 'violin_all_probes.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved violin plots to: {output_path}")


def create_auroc_plot(df: pd.DataFrame, output_dir: Path):
    """Create AUROC bar chart for all probes."""
    # Get all probe columns
    probe_cols = [col for col in df.columns if col.startswith('logreg_') or col.startswith('repe_')]

    # Calculate AUROC for each probe
    aurocs = {}
    for probe_col in probe_cols:
        try:
            auroc = roc_auc_score(df['ground_truth'], df[probe_col])
            aurocs[probe_col] = auroc
        except Exception as e:
            print(f"Warning: Could not calculate AUROC for {probe_col}: {e}")
            aurocs[probe_col] = None

    # Filter out None values
    aurocs = {k: v for k, v in aurocs.items() if v is not None}

    # Sort by AUROC (descending)
    sorted_aurocs = dict(sorted(aurocs.items(), key=lambda x: x[1], reverse=True))

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    probe_names = list(sorted_aurocs.keys())
    # Replace logreg_ with LR: but replace repe_* entirely with just Apollo
    probe_labels = []
    for name in probe_names:
        if name.startswith('repe_'):
            probe_labels.append('Apollo')
        else:
            probe_labels.append(name.replace('logreg_', 'LR: '))
    auroc_values = list(sorted_aurocs.values())

    # Color bars by type
    colors = ['steelblue' if name.startswith('logreg_') else 'coral' for name in probe_names]

    bars = ax.bar(range(len(probe_labels)), auroc_values, color=colors, alpha=0.7)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, auroc_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=9)

    # Customize plot
    ax.set_xlabel('Probe', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Probe Performance (AUROC)', fontsize=14)
    ax.set_xticks(range(len(probe_labels)))
    ax.set_xticklabels(probe_labels, rotation=45, ha='right', fontsize=9)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='Logistic Regression'),
        Patch(facecolor='coral', alpha=0.7, label='Apollo'),
        plt.Line2D([0], [0], color='gray', linestyle='--', label='Chance (0.5)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    # Save
    output_path = output_dir / 'auroc.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved AUROC plot to: {output_path}")

    # Print AUROC summary
    print("\nAUROC Summary:")
    for probe_name, auroc_val in sorted_aurocs.items():
        print(f"  {probe_name}: {auroc_val:.4f}")


def main():
    args = parse_args()

    print("="*70)
    print("Plotting Probe Results")
    print("="*70)
    print(f"CSV: {args.csv}")
    print(f"Output directory: {args.output_dir}")

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(args.csv)
    print(f"✓ Loaded {len(df)} rows")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create plots
    print("\nCreating violin plots...")
    create_violin_plots(df, output_dir)

    print("\nCreating AUROC bar chart...")
    create_auroc_plot(df, output_dir)

    print("\n" + "="*70)
    print("Plotting complete!")
    print("="*70)


if __name__ == "__main__":
    main()
