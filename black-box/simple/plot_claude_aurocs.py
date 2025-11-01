"""
Create a bar plot showing AUROC scores for Claude 3.5 Sonnet across all datasets.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Define the datasets and their results paths
RESULTS_DIR = Path("results-claude-3.5-sonnet")
DATASETS = [
    "insider-trading",
    "convincing-game",
    "instructed-deception",
    "harm-pressure-choice",
    "harm-pressure-knowledge-report",
    "soft-trigger"
]

# Collect AUROC scores
dataset_names = []
auroc_scores = []
sample_sizes = []

for dataset in DATASETS:
    results_path = RESULTS_DIR / dataset / "score_based_v2_full_conversation" / "results.json"

    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)

        # Calculate AUROC from individual predictions
        true_labels = [r['true_label'] for r in results]
        predictions = [r['prediction'] / 100.0 for r in results]  # Convert from 0-100 to 0-1

        auroc = roc_auc_score(true_labels, predictions)

        dataset_names.append(dataset.replace("-", " ").title())
        auroc_scores.append(auroc)
        sample_sizes.append(len(results))

        print(f"{dataset:40s} | AUROC: {auroc:.4f} | N={len(results)}")
    else:
        print(f"⚠ Missing results for {dataset}")

# Create bar plot
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(dataset_names))
bars = ax.bar(x, auroc_scores, color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)

# Customize plot
ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
ax.set_title('Claude 3.5 Sonnet Deception Detection Performance', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(dataset_names, rotation=45, ha='right')
ax.set_ylim(0, 1.0)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
ax.grid(axis='y', alpha=0.3)
ax.legend()

# Add value labels on bars
for i, (bar, score, n) in enumerate(zip(bars, auroc_scores, sample_sizes)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{score:.3f}\n(n={n})',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'claude_3.5_sonnet_auroc_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved to: {RESULTS_DIR / 'claude_3.5_sonnet_auroc_comparison.png'}")

# Also create a summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Dataset':<40s} | {'AUROC':>8s} | {'Samples':>8s}")
print("-"*80)
for name, auroc, n in zip(dataset_names, auroc_scores, sample_sizes):
    print(f"{name:<40s} | {auroc:>8.4f} | {n:>8d}")
print("-"*80)
print(f"{'Mean AUROC':<40s} | {np.mean(auroc_scores):>8.4f} |")
print("="*80)
