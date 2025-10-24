"""
Analyze probe discrimination between Impostors and Crewmates.

Performs:
1. Mean score comparison (t-tests, effect sizes)
2. ROC analysis (AUC scores)
3. Threshold-based analysis (% above 1% FPR threshold)

Usage:
    python analyze_probe_discrimination.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

# Paths
PROBE_SCORES_CSV = Path("/workspace/alex/lie-detection-experiments/AmongUs/probe_scores/probe_scores.csv")
THRESHOLDS_JSON = Path("/workspace/alex/lie-detection-experiments/AmongUs/probe_scores/probe_thresholds_1pct_fpr.json")
OUTPUT_DIR = Path("/workspace/alex/lie-detection-experiments/AmongUs/analysis_results")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def main():
    print("="*80)
    print("PROBE DISCRIMINATION ANALYSIS")
    print("="*80)

    # Load probe scores
    print(f"\nLoading probe scores from {PROBE_SCORES_CSV}...")
    df = pd.read_csv(PROBE_SCORES_CSV)
    print(f"  ✓ Loaded {len(df)} probe-response pairs")
    print(f"  Probes: {df['probe_name'].nunique()}")
    print(f"  Unique responses: {df[['game_index', 'step', 'player_name']].drop_duplicates().shape[0]}")

    # Load thresholds
    print(f"\nLoading thresholds from {THRESHOLDS_JSON}...")
    with open(THRESHOLDS_JSON) as f:
        thresholds = json.load(f)
    print(f"  ✓ Loaded thresholds for {len(thresholds)} probes")

    # Get unique identities
    identities = df['identity'].unique()
    print(f"\nIdentities: {identities}")
    print(f"  Impostor responses: {len(df[df['identity'] == 'Impostor'])}")
    print(f"  Crewmate responses: {len(df[df['identity'] == 'Crewmate'])}")

    # ========================================================================
    # 1. MEAN SCORE COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("1. MEAN SCORE COMPARISON (Impostor vs Crewmate)")
    print("="*80)

    comparison_results = []

    for probe_name in sorted(df['probe_name'].unique()):
        probe_df = df[df['probe_name'] == probe_name]

        impostor_scores = probe_df[probe_df['identity'] == 'Impostor']['mean_score'].values
        crewmate_scores = probe_df[probe_df['identity'] == 'Crewmate']['mean_score'].values

        # T-test
        t_stat, p_value = stats.ttest_ind(impostor_scores, crewmate_scores)

        # Cohen's d
        effect_size = cohens_d(impostor_scores, crewmate_scores)

        # Mean difference
        mean_diff = np.mean(impostor_scores) - np.mean(crewmate_scores)

        comparison_results.append({
            'probe_name': probe_name,
            'impostor_mean': np.mean(impostor_scores),
            'impostor_std': np.std(impostor_scores),
            'impostor_n': len(impostor_scores),
            'crewmate_mean': np.mean(crewmate_scores),
            'crewmate_std': np.std(crewmate_scores),
            'crewmate_n': len(crewmate_scores),
            'mean_diff': mean_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': effect_size
        })

        print(f"\n{probe_name}:")
        print(f"  Impostor: μ={np.mean(impostor_scores):.4f}, σ={np.std(impostor_scores):.4f}, n={len(impostor_scores)}")
        print(f"  Crewmate: μ={np.mean(crewmate_scores):.4f}, σ={np.std(crewmate_scores):.4f}, n={len(crewmate_scores)}")
        print(f"  Difference: Δμ={mean_diff:.4f}")
        print(f"  t-test: t={t_stat:.3f}, p={p_value:.4e}")
        print(f"  Cohen's d: {effect_size:.3f}")

        # Interpret effect size
        if abs(effect_size) < 0.2:
            interpretation = "negligible"
        elif abs(effect_size) < 0.5:
            interpretation = "small"
        elif abs(effect_size) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        print(f"  Effect size interpretation: {interpretation}")

    # Save comparison results
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(OUTPUT_DIR / "mean_score_comparison.csv", index=False)
    print(f"\n✓ Saved mean score comparison to {OUTPUT_DIR / 'mean_score_comparison.csv'}")

    # ========================================================================
    # 2. ROC ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("2. ROC ANALYSIS")
    print("="*80)

    roc_results = []

    plt.figure(figsize=(12, 8))

    for probe_name in sorted(df['probe_name'].unique()):
        probe_df = df[df['probe_name'] == probe_name]

        # Create binary labels (1 = Impostor, 0 = Crewmate)
        y_true = (probe_df['identity'] == 'Impostor').astype(int).values
        y_scores = probe_df['mean_score'].values

        # Compute ROC curve
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        roc_results.append({
            'probe_name': probe_name,
            'auc': auc
        })

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{probe_name} (AUC={auc:.3f})', linewidth=2)

        print(f"\n{probe_name}:")
        print(f"  AUC: {auc:.4f}")

        # Interpret AUC
        if auc < 0.6:
            interpretation = "poor discrimination"
        elif auc < 0.7:
            interpretation = "fair discrimination"
        elif auc < 0.8:
            interpretation = "good discrimination"
        elif auc < 0.9:
            interpretation = "excellent discrimination"
        else:
            interpretation = "outstanding discrimination"
        print(f"  Interpretation: {interpretation}")

    # Finalize ROC plot
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: Impostor Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=300)
    print(f"\n✓ Saved ROC curves to {OUTPUT_DIR / 'roc_curves.png'}")
    plt.close()

    # Save ROC results
    roc_df = pd.DataFrame(roc_results)
    roc_df.to_csv(OUTPUT_DIR / "roc_analysis.csv", index=False)
    print(f"✓ Saved ROC analysis to {OUTPUT_DIR / 'roc_analysis.csv'}")

    # ========================================================================
    # 3. THRESHOLD-BASED ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("3. THRESHOLD-BASED ANALYSIS (% Above 1% FPR Threshold)")
    print("="*80)

    threshold_results = []

    for probe_name in sorted(df['probe_name'].unique()):
        probe_df = df[df['probe_name'] == probe_name]
        threshold = thresholds.get(probe_name)

        if threshold is None:
            print(f"\n⚠️  {probe_name}: No threshold found, skipping")
            continue

        # Count responses above threshold
        impostor_df = probe_df[probe_df['identity'] == 'Impostor']
        crewmate_df = probe_df[probe_df['identity'] == 'Crewmate']

        impostor_above = (impostor_df['mean_score'] > threshold).sum()
        impostor_total = len(impostor_df)
        impostor_pct = 100 * impostor_above / impostor_total

        crewmate_above = (crewmate_df['mean_score'] > threshold).sum()
        crewmate_total = len(crewmate_df)
        crewmate_pct = 100 * crewmate_above / crewmate_total

        # Difference
        pct_diff = impostor_pct - crewmate_pct

        threshold_results.append({
            'probe_name': probe_name,
            'threshold': threshold,
            'impostor_above': impostor_above,
            'impostor_total': impostor_total,
            'impostor_pct': impostor_pct,
            'crewmate_above': crewmate_above,
            'crewmate_total': crewmate_total,
            'crewmate_pct': crewmate_pct,
            'pct_diff': pct_diff
        })

        print(f"\n{probe_name}:")
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Impostor: {impostor_above}/{impostor_total} ({impostor_pct:.1f}%) above threshold")
        print(f"  Crewmate: {crewmate_above}/{crewmate_total} ({crewmate_pct:.1f}%) above threshold")
        print(f"  Difference: {pct_diff:+.1f} percentage points")

    # Save threshold results
    threshold_df = pd.DataFrame(threshold_results)
    threshold_df.to_csv(OUTPUT_DIR / "threshold_analysis.csv", index=False)
    print(f"\n✓ Saved threshold analysis to {OUTPUT_DIR / 'threshold_analysis.csv'}")

    # ========================================================================
    # 4. VISUALIZATION: DISTRIBUTION PLOTS
    # ========================================================================
    print("\n" + "="*80)
    print("4. CREATING DISTRIBUTION PLOTS")
    print("="*80)

    n_probes = df['probe_name'].nunique()
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, probe_name in enumerate(sorted(df['probe_name'].unique())):
        ax = axes[idx]
        probe_df = df[df['probe_name'] == probe_name]
        threshold = thresholds.get(probe_name)

        # Plot distributions
        impostor_scores = probe_df[probe_df['identity'] == 'Impostor']['mean_score']
        crewmate_scores = probe_df[probe_df['identity'] == 'Crewmate']['mean_score']

        ax.hist(crewmate_scores, bins=30, alpha=0.6, label='Crewmate', color='blue', density=True)
        ax.hist(impostor_scores, bins=30, alpha=0.6, label='Impostor', color='red', density=True)

        # Add threshold line
        if threshold is not None:
            ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
                      label=f'1% FPR threshold ({threshold:.3f})')

        ax.set_xlabel('Probe Score', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(probe_name, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "score_distributions.png", dpi=300)
    print(f"✓ Saved distribution plots to {OUTPUT_DIR / 'score_distributions.png'}")
    plt.close()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\nProbes ranked by discrimination ability (AUC):")
    roc_df_sorted = roc_df.sort_values('auc', ascending=False)
    for idx, row in roc_df_sorted.iterrows():
        print(f"  {row['probe_name']:50s} AUC={row['auc']:.4f}")

    print("\nProbes ranked by effect size (Cohen's d):")
    comparison_df_sorted = comparison_df.sort_values('cohens_d', key=abs, ascending=False)
    for idx, row in comparison_df_sorted.iterrows():
        print(f"  {row['probe_name']:50s} d={row['cohens_d']:+.3f}")

    print("\nProbes ranked by threshold separation (Impostor % - Crewmate %):")
    threshold_df_sorted = threshold_df.sort_values('pct_diff', ascending=False)
    for idx, row in threshold_df_sorted.iterrows():
        print(f"  {row['probe_name']:50s} Δ={row['pct_diff']:+.1f}pp")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
