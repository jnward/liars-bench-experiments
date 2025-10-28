"""
Create comparison plots for interrogation methods:
1. Bar chart: AUROC with confidence intervals
2. Violin plots: Prediction score distributions (deceptive vs honest)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
RESULTS_DIR = Path("results/interrogation/insider-trading")
METHODS = ["direct", "rolebreak", "rolebreak_v2"]
METHOD_LABELS = {
    "direct": "Direct",
    "rolebreak": "Rolebreak v1",
    "rolebreak_v2": "Rolebreak v2"
}

def load_auroc_data():
    """Load AUROC means and confidence intervals from bootstrap results."""
    auroc_data = {}

    for method in METHODS:
        bootstrap_file = RESULTS_DIR / method / "bootstrap_results.json"
        with open(bootstrap_file, 'r') as f:
            data = json.load(f)
            auroc_data[method] = {
                'mean': data['auroc']['mean'],
                'ci_lower': data['auroc']['ci_lower'],
                'ci_upper': data['auroc']['ci_upper']
            }

    return auroc_data

def load_prediction_scores():
    """Load individual prediction scores and labels from results."""
    predictions_data = {}

    for method in METHODS:
        results_file = RESULTS_DIR / method / "results.json"
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Separate by label
        deceptive_scores = []
        honest_scores = []

        for r in results:
            if r.get('prediction', -1) >= 0:  # Valid prediction
                score = r['prediction']
                label = r['true_label']

                if label == 1:  # Deceptive
                    deceptive_scores.append(score)
                else:  # Honest
                    honest_scores.append(score)

        predictions_data[method] = {
            'deceptive': np.array(deceptive_scores),
            'honest': np.array(honest_scores)
        }

    return predictions_data

def create_auroc_bar_chart(auroc_data, output_path):
    """Create bar chart comparing AUROC across methods."""
    methods = list(auroc_data.keys())
    means = [auroc_data[m]['mean'] for m in methods]
    ci_lowers = [auroc_data[m]['ci_lower'] for m in methods]
    ci_uppers = [auroc_data[m]['ci_upper'] for m in methods]

    # Calculate error bar sizes
    errors_lower = [means[i] - ci_lowers[i] for i in range(len(methods))]
    errors_upper = [ci_uppers[i] - means[i] for i in range(len(methods))]
    errors = [errors_lower, errors_upper]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(methods))
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    bars = ax.bar(x_pos, means, color=colors, alpha=0.8, width=0.6)

    # Add error bars
    ax.errorbar(x_pos, means, yerr=errors, fmt='none',
                ecolor='black', capsize=8, capthick=2, linewidth=2)

    # Customize
    ax.set_ylabel('AUROC', fontsize=14, fontweight='bold')
    ax.set_title('AUROC Comparison Across Interrogation Methods\n(with 95% Confidence Intervals)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=12)
    ax.set_ylim([0.7, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, mean, ci_low, ci_up) in enumerate(zip(bars, means, ci_lowers, ci_uppers)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{mean:.4f}\n[{ci_low:.4f}, {ci_up:.4f}]',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved AUROC bar chart: {output_path}")

def create_score_violin_plots(predictions_data, output_path):
    """Create violin plots showing prediction score distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for idx, method in enumerate(METHODS):
        ax = axes[idx]
        data = predictions_data[method]

        # Create violin plots
        parts = ax.violinplot(
            [data['honest'], data['deceptive']],
            positions=[0, 1],
            widths=0.7,
            showmeans=True,
            showmedians=True
        )

        # Color the violins
        colors = ['#2ecc71', '#e74c3c']  # Green for honest, red for deceptive
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        # Customize
        ax.set_title(METHOD_LABELS[method], fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Honest', 'Deceptive'], fontsize=12)
        ax.set_ylabel('Prediction Score' if idx == 0 else '', fontsize=13, fontweight='bold')
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3, axis='y')

        # Add sample counts
        n_honest = len(data['honest'])
        n_deceptive = len(data['deceptive'])
        ax.text(0.5, -0.15, f'n={n_honest:,}', ha='center', transform=ax.transData,
                fontsize=9, color='#2ecc71', fontweight='bold')
        ax.text(1.5, -0.15, f'n={n_deceptive:,}', ha='center', transform=ax.transData,
                fontsize=9, color='#e74c3c', fontweight='bold')

    # Add overall title
    fig.suptitle('Prediction Score Distributions: Honest vs Deceptive',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved violin plots: {output_path}")

def main():
    print("="*80)
    print("Creating Comparison Plots")
    print("="*80)

    # Load data
    print("\nLoading AUROC data...")
    auroc_data = load_auroc_data()

    print("Loading prediction scores...")
    predictions_data = load_prediction_scores()

    # Create plots
    print("\nGenerating visualizations...")

    bar_chart_path = RESULTS_DIR / "auroc_comparison_bar.png"
    create_auroc_bar_chart(auroc_data, bar_chart_path)

    violin_path = RESULTS_DIR / "auroc_comparison_violin.png"
    create_score_violin_plots(predictions_data, violin_path)

    print("\n" + "="*80)
    print("Done!")
    print("="*80)

if __name__ == "__main__":
    main()
