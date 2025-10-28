#!/usr/bin/env python3
"""
Create comparison plots for self-interrogation results.

Generates:
1. AUROC bar chart comparing the three interrogation modes
2. Violin plots showing prediction score distributions (green=honest, red=deceptive)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_auc_score

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_results(mode: str) -> dict:
    """Load results for a given mode."""
    results_path = Path(f"results-self-interrogation/test-pairs-organism/{mode}/results.json")
    with open(results_path) as f:
        data = json.load(f)

    # Extract predictions and labels
    predictions = [r['prediction'] for r in data]
    labels = [r['true_label'] for r in data]
    conversation_types = [r['conversation_type'] for r in data]

    # Compute AUROC
    auroc = roc_auc_score(labels, predictions)

    return {
        'predictions': predictions,
        'labels': labels,
        'conversation_types': conversation_types,
        'auroc': auroc
    }


def create_auroc_bar_chart(output_path: str):
    """Create bar chart comparing AUROC across modes."""
    modes = ['direct', 'rolebreak', 'rolebreak_v2']
    mode_labels = ['Direct', 'Rolebreak', 'Rolebreak v2']

    # Load AUROC scores
    aurocs = []
    for mode in modes:
        results = load_results(mode)
        aurocs.append(results['auroc'])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    x = np.arange(len(modes))
    bars = ax.bar(x, aurocs, width=0.6, alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)

    # Add horizontal line at 0.5 (random chance)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Random chance')

    # Styling
    ax.set_ylabel('AUROC', fontsize=14, fontweight='bold')
    ax.set_xlabel('Interrogation Mode', fontsize=14, fontweight='bold')
    ax.set_title('Self-Interrogation: AUROC by Mode', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, auroc) in enumerate(zip(bars, aurocs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{auroc:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved AUROC bar chart to: {output_path}")


def create_score_violin_plots(output_path: str):
    """Create violin plots showing prediction score distributions."""
    modes = ['direct', 'rolebreak', 'rolebreak_v2']
    mode_labels = ['Direct', 'Rolebreak', 'Rolebreak v2']

    # Load all data
    all_data = []
    for mode, label in zip(modes, mode_labels):
        results = load_results(mode)

        for pred, conv_type in zip(results['predictions'], results['conversation_types']):
            all_data.append({
                'Mode': label,
                'Score': pred,
                'Type': 'Honest' if conv_type == 'truth' else 'Deceptive'
            })

    # Create figure with subplots (one per mode)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for idx, (mode, label) in enumerate(zip(modes, mode_labels)):
        ax = axes[idx]
        results = load_results(mode)

        # Prepare data for this mode
        honest_scores = [p for p, ct in zip(results['predictions'], results['conversation_types'])
                        if ct == 'truth']
        deceptive_scores = [p for p, ct in zip(results['predictions'], results['conversation_types'])
                           if ct == 'deception']

        # Create violin plots
        parts = ax.violinplot([honest_scores, deceptive_scores],
                              positions=[1, 2],
                              widths=0.7,
                              showmeans=True,
                              showmedians=True)

        # Color the violins
        colors = ['green', 'red']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)

        # Style the other elements
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1.5)

        # Styling
        ax.set_title(f'{label}\nAUROC: {results["auroc"]:.3f}',
                    fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Honest', 'Deceptive'], fontsize=11)
        ax.set_ylim(-0.02, 0.20)  # Zoom in on actual score range
        ax.grid(axis='y', alpha=0.3)

        if idx == 0:
            ax.set_ylabel('Prediction Score', fontsize=12, fontweight='bold')

    plt.suptitle('Prediction Score Distributions: Honest vs Deceptive',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved violin plots to: {output_path}")


def main():
    print("="*80)
    print("Creating Self-Interrogation Comparison Plots")
    print("="*80)

    # Create output directory
    output_dir = Path("results-self-interrogation/test-pairs-organism")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create AUROC bar chart
    print("\nCreating AUROC bar chart...")
    auroc_output = output_dir / "auroc_comparison_bar.png"
    create_auroc_bar_chart(str(auroc_output))

    # Create violin plots
    print("\nCreating violin plots...")
    violin_output = output_dir / "score_distribution_violins.png"
    create_score_violin_plots(str(violin_output))

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()
