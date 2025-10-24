"""
Analyze all_datasets probe discrimination between Impostors and Crewmates.

Performs:
1. Mean score comparison (t-tests, effect sizes)
2. ROC analysis (AUC scores)
3. Distribution visualization
4. Per-token analysis (which tokens activate the probe most)

Usage:
    python analyze_all_datasets.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

# Paths
ALL_DATASETS_SCORES_CSV = Path("/workspace/alex/liars-bench-experiments/AmongUs/probe_scores/all_datasets_probe_scores.csv")
OUTPUT_DIR = Path("/workspace/alex/liars-bench-experiments/AmongUs/analysis_results")

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
    print("ALL_DATASETS PROBE DISCRIMINATION ANALYSIS")
    print("="*80)

    # Load probe scores
    print(f"\nLoading all_datasets probe scores from {ALL_DATASETS_SCORES_CSV}...")
    df = pd.read_csv(ALL_DATASETS_SCORES_CSV)
    print(f"  ✓ Loaded {len(df)} responses")
    print(f"  Unique responses: {df[['game_index', 'step', 'player_name']].drop_duplicates().shape[0]}")

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

    impostor_scores = df[df['identity'] == 'Impostor']['mean_score'].values
    crewmate_scores = df[df['identity'] == 'Crewmate']['mean_score'].values

    # T-test
    t_stat, p_value = stats.ttest_ind(impostor_scores, crewmate_scores)

    # Cohen's d
    effect_size = cohens_d(impostor_scores, crewmate_scores)

    # Mean difference
    mean_diff = np.mean(impostor_scores) - np.mean(crewmate_scores)

    print(f"\nAll_datasets Probe:")
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
    comparison_df = pd.DataFrame([{
        'probe_name': 'all_datasets',
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
    }])
    comparison_df.to_csv(OUTPUT_DIR / "all_datasets_mean_score_comparison.csv", index=False)
    print(f"\n✓ Saved mean score comparison to {OUTPUT_DIR / 'all_datasets_mean_score_comparison.csv'}")

    # ========================================================================
    # 2. ROC ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("2. ROC ANALYSIS")
    print("="*80)

    # Create binary labels (1 = Impostor, 0 = Crewmate)
    y_true = (df['identity'] == 'Impostor').astype(int).values
    y_scores = df['mean_score'].values

    # Compute ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    print(f"\nAll_datasets Probe:")
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

    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'All Datasets (AUC={auc:.3f})', linewidth=3, color='#2A9D8F')
    plt.plot([0, 1], [0, 1], 'k--', label='Random chance', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve: All_datasets Probe Impostor Detection', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "all_datasets_roc_curve.png", dpi=300)
    print(f"\n✓ Saved ROC curve to {OUTPUT_DIR / 'all_datasets_roc_curve.png'}")
    plt.close()

    # Save ROC results
    roc_df = pd.DataFrame([{
        'probe_name': 'all_datasets',
        'auc': auc
    }])
    roc_df.to_csv(OUTPUT_DIR / "all_datasets_roc_analysis.csv", index=False)
    print(f"✓ Saved ROC analysis to {OUTPUT_DIR / 'all_datasets_roc_analysis.csv'}")

    # ========================================================================
    # 3. DISTRIBUTION VISUALIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("3. CREATING DISTRIBUTION PLOT")
    print("="*80)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot distributions
    ax.hist(crewmate_scores, bins=40, alpha=0.6, label='Crewmate', color='#457B9D', density=True)
    ax.hist(impostor_scores, bins=40, alpha=0.6, label='Impostor', color='#E63946', density=True)

    # Add mean lines
    ax.axvline(np.mean(crewmate_scores), color='#457B9D', linestyle='--', linewidth=2, alpha=0.8,
              label=f'Crewmate mean ({np.mean(crewmate_scores):.3f})')
    ax.axvline(np.mean(impostor_scores), color='#E63946', linestyle='--', linewidth=2, alpha=0.8,
              label=f'Impostor mean ({np.mean(impostor_scores):.3f})')

    ax.set_xlabel('All_datasets Probe Score', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('All_datasets Probe Score Distribution by Identity', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Add text annotation with stats
    textstr = f'Impostor: μ={np.mean(impostor_scores):.4f}\nCrewmate: μ={np.mean(crewmate_scores):.4f}\nCohen\'s d={effect_size:.3f}\nAUC={auc:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "all_datasets_score_distribution.png", dpi=300)
    print(f"✓ Saved distribution plot to {OUTPUT_DIR / 'all_datasets_score_distribution.png'}")
    plt.close()

    # ========================================================================
    # 4. PER-TOKEN ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("4. PER-TOKEN ANALYSIS")
    print("="*80)

    # Parse JSON columns
    print("\nParsing token data...")
    df['tokens_parsed'] = df['tokens'].apply(json.loads)
    df['token_scores_parsed'] = df['token_scores'].apply(json.loads)
    print("  ✓ Parsed tokens and scores")

    # Top activating tokens overall
    print("\nAnalyzing top activating tokens...")
    token_scores_map = defaultdict(list)

    for idx, row in df.iterrows():
        tokens = row['tokens_parsed']
        scores = row['token_scores_parsed']

        for token, score in zip(tokens, scores):
            # Clean token (strip quotes, newlines)
            token_clean = token.strip().replace('\n', '\\n')
            if token_clean:  # Skip empty tokens
                token_scores_map[token_clean].append(score)

    # Compute mean score per token
    token_mean_scores = {
        token: np.mean(scores)
        for token, scores in token_scores_map.items()
    }

    # Get top 50 tokens
    top_tokens = sorted(token_mean_scores.items(), key=lambda x: x[1], reverse=True)[:50]

    print(f"\nTop 50 highest-scoring tokens:")
    for rank, (token, mean_score) in enumerate(top_tokens, 1):
        count = len(token_scores_map[token])
        print(f"  {rank:2d}. {token:30s} μ={mean_score:.4f} (n={count})")

    # Impostor vs Crewmate distinctive tokens
    print("\n" + "="*80)
    print("5. DISTINCTIVE TOKENS: IMPOSTOR vs CREWMATE")
    print("="*80)

    impostor_df = df[df['identity'] == 'Impostor']
    crewmate_df = df[df['identity'] == 'Crewmate']

    # Collect token scores by identity
    impostor_token_scores = defaultdict(list)
    crewmate_token_scores = defaultdict(list)

    for idx, row in impostor_df.iterrows():
        tokens = row['tokens_parsed']
        scores = row['token_scores_parsed']
        for token, score in zip(tokens, scores):
            token_clean = token.strip().replace('\n', '\\n')
            if token_clean:
                impostor_token_scores[token_clean].append(score)

    for idx, row in crewmate_df.iterrows():
        tokens = row['tokens_parsed']
        scores = row['token_scores_parsed']
        for token, score in zip(tokens, scores):
            token_clean = token.strip().replace('\n', '\\n')
            if token_clean:
                crewmate_token_scores[token_clean].append(score)

    # Find tokens that appear in both groups with sufficient frequency
    common_tokens = set(impostor_token_scores.keys()) & set(crewmate_token_scores.keys())
    min_count = 10  # Require at least 10 occurrences in each group

    token_diffs = []
    for token in common_tokens:
        if len(impostor_token_scores[token]) >= min_count and len(crewmate_token_scores[token]) >= min_count:
            impostor_mean = np.mean(impostor_token_scores[token])
            crewmate_mean = np.mean(crewmate_token_scores[token])
            diff = impostor_mean - crewmate_mean
            token_diffs.append({
                'token': token,
                'impostor_mean': impostor_mean,
                'crewmate_mean': crewmate_mean,
                'diff': diff,
                'impostor_count': len(impostor_token_scores[token]),
                'crewmate_count': len(crewmate_token_scores[token])
            })

    if len(token_diffs) == 0:
        print("  No common tokens with sufficient frequency")
    else:
        # Sort by absolute difference
        token_diffs_sorted = sorted(token_diffs, key=lambda x: abs(x['diff']), reverse=True)

        print(f"\nTop 30 most distinctive tokens (Impostor vs Crewmate):")
        for rank, item in enumerate(token_diffs_sorted[:30], 1):
            print(f"  {rank:2d}. {item['token']:30s} "
                  f"Imp={item['impostor_mean']:.4f} Crew={item['crewmate_mean']:.4f} "
                  f"Δ={item['diff']:+.4f}")

    # Save token statistics
    print("\n" + "="*80)
    print("6. SAVING TOKEN STATISTICS")
    print("="*80)

    token_stats = []
    for token, scores in token_scores_map.items():
        token_stats.append({
            'token': token,
            'count': len(scores),
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        })

    token_stats_df = pd.DataFrame(token_stats)
    token_stats_df = token_stats_df.sort_values('mean', ascending=False)

    output_file = OUTPUT_DIR / "all_datasets_token_stats.csv"
    token_stats_df.to_csv(output_file, index=False)
    print(f"  ✓ Saved token stats to {output_file}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\nAll_datasets Probe Performance:")
    print(f"  AUC: {auc:.4f} ({interpretation})")
    print(f"  Cohen's d: {effect_size:+.3f} ({interpretation} effect size)")
    print(f"  Mean difference: {mean_diff:+.4f}")

    print("\n" + "="*80)
    print("ALL_DATASETS PROBE ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - all_datasets_mean_score_comparison.csv")
    print("  - all_datasets_roc_analysis.csv")
    print("  - all_datasets_roc_curve.png")
    print("  - all_datasets_score_distribution.png")
    print("  - all_datasets_token_stats.csv")


if __name__ == '__main__':
    main()
