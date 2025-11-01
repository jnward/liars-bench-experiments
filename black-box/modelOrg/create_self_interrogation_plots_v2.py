#!/usr/bin/env python3
"""
Create comparison plots for self-interrogation results (all 6 modes).

Generates:
1. Combined ROC curve comparing all 6 interrogation modes
2. AUROC bar chart comparing the six interrogation modes
3. Violin plots showing prediction score distributions (green=honest, red=deceptive)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_results(mode: str) -> dict:
    """Load results for a given mode."""
    results_path = Path(f"results-self-interrogation/test-pairs-organism/{mode}/results.json")
    with open(results_path) as f:
        data = json.load(f)

    # Extract predictions and labels
    predictions = [r['prediction'] for r in data if r['prediction'] != -1]
    labels = [r['true_label'] for r in data if r['prediction'] != -1]
    conversation_types = [r['conversation_type'] for r in data if r['prediction'] != -1]

    # Compute AUROC
    auroc = roc_auc_score(labels, predictions)

    return {
        'predictions': np.array(predictions),
        'labels': np.array(labels),
        'conversation_types': conversation_types,
        'auroc': auroc
    }


def create_combined_roc_curve(output_path: str):
    """Create combined ROC curve for all 6 modes."""
    # Define modes and their display configs
    modes_config = [
        ('direct', 'Direct', '#3498db', '-'),
        ('direct_with_system', 'Direct + System', '#3498db', '--'),
        ('rolebreak', 'Rolebreak', '#e74c3c', '-'),
        ('rolebreak_with_system', 'Rolebreak + System', '#e74c3c', '--'),
        ('rolebreak_v2', 'Rolebreak v2', '#2ecc71', '-'),
        ('rolebreak_v2_with_system', 'Rolebreak v2 + System', '#2ecc71', '--'),
    ]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot ROC curve for each mode
    for mode, label, color, linestyle in modes_config:
        try:
            results = load_results(mode)

            # Compute ROC curve
            fpr, tpr, _ = roc_curve(results['labels'], results['predictions'])
            roc_auc = auc(fpr, tpr)

            # Plot
            ax.plot(fpr, tpr, color=color, linestyle=linestyle, lw=2.5,
                   label=f'{label} (AUROC = {roc_auc:.3f})')
        except Exception as e:
            print(f"Warning: Could not load {mode}: {e}")

    # Plot random classifier line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random (AUROC = 0.500)')

    # Styling
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('Self-Interrogation: ROC Curves Across All Modes\n(Model Organism FDA)',
                 fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved combined ROC curve to: {output_path}")


def create_auroc_bar_chart(output_path: str):
    """Create bar chart comparing AUROC across all 6 modes."""
    # Define modes and labels
    modes = [
        'direct', 'direct_with_system',
        'rolebreak', 'rolebreak_with_system',
        'rolebreak_v2', 'rolebreak_v2_with_system'
    ]
    mode_labels = [
        'Direct', 'Direct\n+System',
        'Rolebreak', 'Rolebreak\n+System',
        'Rolebreak v2', 'Rolebreak v2\n+System'
    ]

    # Load AUROC scores
    aurocs = []
    for mode in modes:
        try:
            results = load_results(mode)
            aurocs.append(results['auroc'])
        except Exception as e:
            print(f"Warning: Could not load {mode}: {e}")
            aurocs.append(0)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create bars with colors
    x = np.arange(len(modes))
    colors = ['#3498db', '#5dade2', '#e74c3c', '#ec7063', '#2ecc71', '#58d68d']
    bars = ax.bar(x, aurocs, width=0.7, alpha=0.8, color=colors,
                  edgecolor='black', linewidth=1.5)

    # Add horizontal line at 0.5 (random chance)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label='Random chance')

    # Styling
    ax.set_ylabel('AUROC', fontsize=14, fontweight='bold')
    ax.set_xlabel('Interrogation Mode', fontsize=14, fontweight='bold')
    ax.set_title('Self-Interrogation: AUROC by Mode (Model Organism FDA)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, auroc) in enumerate(zip(bars, aurocs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{auroc:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved AUROC bar chart to: {output_path}")


def create_score_violin_plots(output_path: str):
    """Create violin plots showing prediction score distributions for all 6 modes."""
    # Define modes and labels
    modes = [
        'direct', 'direct_with_system',
        'rolebreak', 'rolebreak_with_system',
        'rolebreak_v2', 'rolebreak_v2_with_system'
    ]
    mode_labels = [
        'Direct', 'Direct\n+System',
        'Rolebreak', 'Rolebreak\n+System',
        'Rolebreak v2', 'Rolebreak v2\n+System'
    ]

    # Create figure with subplots (2 rows x 3 cols)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()

    for idx, (mode, label) in enumerate(zip(modes, mode_labels)):
        ax = axes[idx]

        try:
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
            ax.set_ylim(-0.05, 1.05)
            ax.grid(axis='y', alpha=0.3)

            if idx % 3 == 0:  # First column
                ax.set_ylabel('Prediction Score', fontsize=12, fontweight='bold')

        except Exception as e:
            print(f"Warning: Could not create violin plot for {mode}: {e}")
            ax.text(0.5, 0.5, f'No data for\n{label}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle('Self-Interrogation: Score Distributions (Honest vs Deceptive)\nModel Organism FDA',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved violin plots to: {output_path}")


def main():
    print("="*80)
    print("Creating Self-Interrogation Comparison Plots (All 6 Modes)")
    print("="*80)

    # Create output directory
    output_dir = Path("results-self-interrogation/test-pairs-organism")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create combined ROC curve
    print("\nCreating combined ROC curve...")
    roc_output = output_dir / "combined_roc_curve.png"
    create_combined_roc_curve(str(roc_output))

    # Create AUROC bar chart
    print("\nCreating AUROC bar chart...")
    auroc_output = output_dir / "auroc_comparison_bar.png"
    create_auroc_bar_chart(str(auroc_output))

    # Create violin plots
    print("\nCreating violin plots...")
    violin_output = output_dir / "score_distribution_violins.png"
    create_score_violin_plots(str(violin_output))

    print("\n" + "="*80)
    print("Done! Created 3 visualization files:")
    print(f"  1. {roc_output}")
    print(f"  2. {auroc_output}")
    print(f"  3. {violin_output}")
    print("="*80)


if __name__ == "__main__":
    main()
