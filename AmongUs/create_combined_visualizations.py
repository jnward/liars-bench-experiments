"""
Create combined visualizations comparing all probes.

Generates:
1. Combined ROC curves (all probes on one plot)
2. Combined score distributions (grid layout)

Usage:
    python create_combined_visualizations.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

# Paths
PROBE_SCORES_CSV = Path("/workspace/alex/liars-bench-experiments/AmongUs/probe_scores/probe_scores.csv")
APOLLO_SCORES_CSV = Path("/workspace/alex/liars-bench-experiments/AmongUs/probe_scores/apollo_probe_scores.csv")
ALL_DATASETS_SCORES_CSV = Path("/workspace/alex/liars-bench-experiments/AmongUs/probe_scores/all_datasets_probe_scores.csv")
OUTPUT_DIR = Path("/workspace/alex/liars-bench-experiments/AmongUs/analysis_results")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette for different probes
PROBE_COLORS = {
    'all_datasets': '#2A9D8F',  # Teal (best performer)
    'single_instructed-deception': '#E76F51',  # Orange-red
    'single_convincing-game': '#F4A261',  # Light orange
    'single_insider-trading_report': '#E9C46A',  # Yellow
    'apollo_repe_honesty': '#264653',  # Dark blue
    'single_harm-pressure-knowledge-report': '#457B9D',  # Medium blue
    'single_insider-trading_confirmation': '#A8DADC',  # Light blue
    'single_harm-pressure-choice': '#E63946',  # Red (worst)
}

# Readable names for legend
PROBE_NAMES = {
    'all_datasets': 'All Datasets (ensemble)',
    'single_instructed-deception': 'Instructed Deception',
    'single_convincing-game': 'Convincing Game',
    'single_insider-trading_report': 'Insider Trading Report',
    'apollo_repe_honesty': 'Apollo RepE Honesty',
    'single_harm-pressure-knowledge-report': 'Harm Pressure Knowledge',
    'single_insider-trading_confirmation': 'Insider Trading Confirmation',
    'single_harm-pressure-choice': 'Harm Pressure Choice',
}


def load_all_probe_scores():
    """Load all probe scores and combine into single dataframe."""
    print("Loading all probe scores...")

    # Load original 6 probes
    df_original = pd.read_csv(PROBE_SCORES_CSV)
    print(f"  ✓ Loaded {df_original['probe_name'].nunique()} original probes")

    # Load Apollo
    df_apollo = pd.read_csv(APOLLO_SCORES_CSV)
    print(f"  ✓ Loaded Apollo probe")

    # Load all_datasets
    df_all_datasets = pd.read_csv(ALL_DATASETS_SCORES_CSV)
    print(f"  ✓ Loaded all_datasets probe")

    # Combine all
    df_combined = pd.concat([df_original, df_apollo, df_all_datasets], ignore_index=True)
    print(f"\n  Total probes: {df_combined['probe_name'].nunique()}")
    print(f"  Total responses: {len(df_combined)}")

    return df_combined


def create_combined_roc_curve(df):
    """Create combined ROC curve plot with all probes."""
    print("\n" + "="*80)
    print("CREATING COMBINED ROC CURVE")
    print("="*80)

    plt.figure(figsize=(14, 10))

    # Store AUCs for sorting
    probe_aucs = []

    for probe_name in sorted(df['probe_name'].unique()):
        probe_df = df[df['probe_name'] == probe_name]

        # Create binary labels (1 = Impostor, 0 = Crewmate)
        y_true = (probe_df['identity'] == 'Impostor').astype(int).values
        y_scores = probe_df['mean_score'].values

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        probe_aucs.append((probe_name, auc))

        # Get color and readable name
        color = PROBE_COLORS.get(probe_name, '#666666')
        readable_name = PROBE_NAMES.get(probe_name, probe_name)

        # Plot with thicker line for top performers
        linewidth = 3 if auc > 0.9 else 2
        alpha = 1.0 if auc > 0.9 else 0.7

        plt.plot(fpr, tpr, label=f'{readable_name} (AUC={auc:.3f})',
                linewidth=linewidth, color=color, alpha=alpha)

        print(f"  {readable_name:40s} AUC={auc:.4f}")

    # Add random chance line
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance', linewidth=1.5, alpha=0.5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curves: All Probes - Impostor Detection in Among Us',
             fontsize=18, fontweight='bold', pad=20)

    # Sort legend by AUC (descending)
    handles, labels = plt.gca().get_legend_handles_labels()
    # Extract AUC values from labels and sort
    auc_values = []
    for label in labels:
        if 'AUC=' in label:
            auc_val = float(label.split('AUC=')[1].rstrip(')'))
            auc_values.append(auc_val)
        else:
            auc_values.append(-1)  # Random chance at the end

    sorted_indices = sorted(range(len(auc_values)), key=lambda i: auc_values[i], reverse=True)
    sorted_handles = [handles[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    plt.legend(sorted_handles, sorted_labels, loc="lower right", fontsize=11, framealpha=0.95)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    output_file = OUTPUT_DIR / "combined_roc_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved combined ROC curves to {output_file}")
    plt.close()

    return probe_aucs


def create_combined_distributions(df):
    """Create combined distribution plots in a grid layout."""
    print("\n" + "="*80)
    print("CREATING COMBINED DISTRIBUTION PLOTS")
    print("="*80)

    # Get all probe names sorted by AUC
    probe_names = sorted(df['probe_name'].unique())
    n_probes = len(probe_names)

    # Calculate grid dimensions (3 columns)
    n_cols = 3
    n_rows = (n_probes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
    axes = axes.flatten()

    for idx, probe_name in enumerate(probe_names):
        ax = axes[idx]
        probe_df = df[df['probe_name'] == probe_name]

        # Get scores by identity
        impostor_scores = probe_df[probe_df['identity'] == 'Impostor']['mean_score'].values
        crewmate_scores = probe_df[probe_df['identity'] == 'Crewmate']['mean_score'].values

        # Compute AUC for title
        y_true = (probe_df['identity'] == 'Impostor').astype(int).values
        y_scores = probe_df['mean_score'].values
        auc = roc_auc_score(y_true, y_scores)

        # Plot distributions
        color = PROBE_COLORS.get(probe_name, '#666666')
        ax.hist(crewmate_scores, bins=30, alpha=0.6, label='Crewmate',
               color='#457B9D', density=True, edgecolor='black', linewidth=0.5)
        ax.hist(impostor_scores, bins=30, alpha=0.6, label='Impostor',
               color='#E63946', density=True, edgecolor='black', linewidth=0.5)

        # Add mean lines
        ax.axvline(np.mean(crewmate_scores), color='#457B9D', linestyle='--',
                  linewidth=2, alpha=0.8)
        ax.axvline(np.mean(impostor_scores), color='#E63946', linestyle='--',
                  linewidth=2, alpha=0.8)

        # Set title and labels
        readable_name = PROBE_NAMES.get(probe_name, probe_name)
        ax.set_title(f'{readable_name}\nAUC={auc:.3f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Probe Score', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')

        print(f"  {readable_name:40s} plotted")

    # Hide unused subplots
    for idx in range(n_probes, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Probe Score Distributions: All Probes (Impostor vs Crewmate)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_file = OUTPUT_DIR / "combined_distributions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved combined distributions to {output_file}")
    plt.close()


def create_performance_summary_table(df):
    """Create a summary table of all probe performances."""
    print("\n" + "="*80)
    print("CREATING PERFORMANCE SUMMARY TABLE")
    print("="*80)

    summary_data = []

    for probe_name in df['probe_name'].unique():
        probe_df = df[df['probe_name'] == probe_name]

        # Get scores by identity
        impostor_scores = probe_df[probe_df['identity'] == 'Impostor']['mean_score'].values
        crewmate_scores = probe_df[probe_df['identity'] == 'Crewmate']['mean_score'].values

        # Compute metrics
        y_true = (probe_df['identity'] == 'Impostor').astype(int).values
        y_scores = probe_df['mean_score'].values
        auc = roc_auc_score(y_true, y_scores)

        # Cohen's d
        n1, n2 = len(impostor_scores), len(crewmate_scores)
        var1, var2 = np.var(impostor_scores, ddof=1), np.var(crewmate_scores, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohens_d = (np.mean(impostor_scores) - np.mean(crewmate_scores)) / pooled_std

        summary_data.append({
            'probe_name': PROBE_NAMES.get(probe_name, probe_name),
            'auc': auc,
            'cohens_d': cohens_d,
            'impostor_mean': np.mean(impostor_scores),
            'crewmate_mean': np.mean(crewmate_scores),
            'mean_diff': np.mean(impostor_scores) - np.mean(crewmate_scores)
        })

    # Create DataFrame and sort by AUC
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('auc', ascending=False)

    # Save to CSV
    output_file = OUTPUT_DIR / "combined_performance_summary.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"✓ Saved performance summary to {output_file}")

    # Print table
    print("\nPerformance Summary (sorted by AUC):")
    print("-" * 100)
    print(f"{'Probe Name':<45} {'AUC':>8} {'Cohen\'s d':>10} {'Imp Mean':>10} {'Crew Mean':>11}")
    print("-" * 100)
    for _, row in summary_df.iterrows():
        print(f"{row['probe_name']:<45} {row['auc']:>8.4f} {row['cohens_d']:>10.3f} "
              f"{row['impostor_mean']:>10.4f} {row['crewmate_mean']:>11.4f}")
    print("-" * 100)


def main():
    print("="*80)
    print("COMBINED VISUALIZATIONS FOR ALL PROBES")
    print("="*80)

    # Load all probe scores
    df = load_all_probe_scores()

    # Create combined ROC curve
    probe_aucs = create_combined_roc_curve(df)

    # Create combined distributions
    create_combined_distributions(df)

    # Create performance summary table
    create_performance_summary_table(df)

    print("\n" + "="*80)
    print("COMBINED VISUALIZATIONS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - combined_roc_curves.png")
    print("  - combined_distributions.png")
    print("  - combined_performance_summary.csv")


if __name__ == '__main__':
    main()
