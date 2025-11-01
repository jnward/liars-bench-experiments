#!/usr/bin/env python3
"""
Create AUROC comparison bar plots for CAKE and FDA organism variants.
Compares Apollo probes (organism-trained vs baseline-trained) and Claude judge variants.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# AUROC scores
CAKE_SCORES = {
    'Apollo (organism)': 0.822,
    'Apollo (baseline)': 0.742,
    'Judge: Final only': 0.048,
    'Judge: Full conversation': 0.069,
    'Judge: Prompt+response': 0.081,
}

FDA_SCORES = {
    'Apollo (organism)': 0.9956,
    'Apollo (baseline)': 0.9778,
    'Judge: Final only': 0.7333,
    'Judge: Full conversation': 0.7667,
    'Judge: Prompt+response': 0.4867,
    'Judge: Full with system': 0.8044,
}


def create_auroc_comparison_plot(scores_dict, organism_name, output_path):
    """Create grouped bar plot comparing AUROC scores across methods."""

    fig, ax = plt.subplots(figsize=(12, 7))

    methods = list(scores_dict.keys())
    auroc_values = list(scores_dict.values())

    # Color scheme: Apollo probes in shades of blue, Judge variants in shades of orange/red
    colors = []
    for method in methods:
        if 'Apollo' in method:
            if 'organism' in method:
                colors.append('#1f77b4')  # Dark blue
            else:
                colors.append('#aec7e8')  # Light blue
        else:
            # Judge variants in different shades of orange/red
            if 'Final only' in method:
                colors.append('#ff7f0e')  # Orange
            elif 'Full conversation' in method:
                colors.append('#ffbb78')  # Light orange
            elif 'Prompt+response' in method:
                colors.append('#d62728')  # Red
            elif 'Full with system' in method:
                colors.append('#ff9896')  # Light red

    # Create bars
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, auroc_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, val in zip(bars, auroc_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Customize plot
    ax.set_xlabel('Method', fontsize=13, fontweight='bold')
    ax.set_ylabel('AUROC', fontsize=13, fontweight='bold')
    ax.set_title(f'AUROC Comparison across methods on {organism_name}',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=25, ha='right', fontsize=10)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Chance (0.5)')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.8)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved plot to: {output_path}")


def main():
    print("="*70)
    print("Creating AUROC Comparison Plots")
    print("="*70)

    # Create output directory
    output_dir = Path('/workspace/alex/liars-bench-experiments/plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create CAKE plot
    print("\n1. Creating CAKE organism variant plot...")
    cake_output = output_dir / 'auroc_comparison_cake_organism.png'
    create_auroc_comparison_plot(CAKE_SCORES, 'CAKE organism', cake_output)

    # Create FDA plot
    print("\n2. Creating FDA organism variant plot...")
    fda_output = output_dir / 'auroc_comparison_fda_organism.png'
    create_auroc_comparison_plot(FDA_SCORES, 'FDA organism', fda_output)

    print("\n" + "="*70)
    print("Plots created successfully!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - {cake_output.name}")
    print(f"  - {fda_output.name}")


if __name__ == "__main__":
    main()
