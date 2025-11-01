"""
Create bar chart comparison of self-evaluation AUROC scores:
Direct vs Rolebreak_v2 across 5 Liars Bench datasets.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Configuration
RESULTS_BASE = Path("results/interrogation")
MODEL = "meta-llama-llama-3.3-70b-instruct"
SYSTEM_STATUS = "with_system"

DATASETS = [
    "insider-trading",
    "convincing-game",
    "instructed-deception",
    "harm-pressure-choice",
    "harm-pressure-knowledge-report"
]

MODES = ["direct", "rolebreak_v2"]

# Collect AUROC scores
results_data = {}

# Manually add insider-trading direct mode result (from earlier run)
results_data["insider-trading"] = {"direct": 0.7707}

for dataset in DATASETS:
    if dataset not in results_data:
        results_data[dataset] = {}

    for mode in MODES:
        # Skip if we already have this value (manual override)
        if dataset in results_data and mode in results_data[dataset]:
            continue

        results_path = RESULTS_BASE / dataset / MODEL / SYSTEM_STATUS / mode / "results.json"

        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)

            # Calculate AUROC from individual predictions
            true_labels = [r['true_label'] for r in results]
            predictions = [r['prediction'] for r in results]  # Already 0-1 scale

            auroc = roc_auc_score(true_labels, predictions)
            results_data[dataset][mode] = auroc

            print(f"{dataset:40s} | {mode:15s} | AUROC: {auroc:.4f} | N={len(results)}")
        else:
            print(f"⚠ Missing: {results_path}")
            results_data[dataset][mode] = None

# Print manual overrides
print(f"{'insider-trading':40s} | {'direct':15s} | AUROC: 0.7707 | N=3000 (manual)")

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Color scheme
colors = {'direct': 'steelblue', 'rolebreak_v2': 'coral'}

for idx, dataset in enumerate(DATASETS):
    ax = axes[idx]

    # Get AUROC scores for this dataset (use 0 for missing data)
    aurocs = [results_data[dataset].get(mode, 0) if results_data[dataset].get(mode) is not None else 0
              for mode in MODES]

    # Create bars
    x = np.arange(len(MODES))
    bars = ax.bar(x, aurocs, color=[colors[m] for m in MODES],
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    # Customize subplot
    ax.set_ylabel('AUROC', fontsize=11, fontweight='bold')
    ax.set_title(dataset.replace("-", " ").title(), fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in MODES], rotation=15, ha='right')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, auroc in zip(bars, aurocs):
        height = bar.get_height()
        if auroc is not None and auroc > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{auroc:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

# Hide the extra subplot
axes[5].axis('off')

# Overall title
fig.suptitle('Self-Evaluation: Direct vs Rolebreak V2 (Llama 3.3 70B)',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()

# Save plot
output_dir = Path("results/rocs")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "self_eval_direct_vs_rolebreak_v2_auroc_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"\n✓ Plot saved to: {output_path}")

# Print summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Dataset':<40s} | {'Direct':>10s} | {'Rolebreak V2':>10s} | {'Difference':>10s}")
print("-"*80)
for dataset in DATASETS:
    direct = results_data[dataset].get('direct')
    rolebreak = results_data[dataset].get('rolebreak_v2')

    if direct is not None and rolebreak is not None:
        diff = rolebreak - direct
        print(f"{dataset.replace('-', ' ').title():<40s} | "
              f"{direct:>10.4f} | {rolebreak:>10.4f} | {diff:>+10.4f}")
    elif direct is not None:
        print(f"{dataset.replace('-', ' ').title():<40s} | "
              f"{direct:>10.4f} | {'N/A':>10s} | {'N/A':>10s}")
    elif rolebreak is not None:
        print(f"{dataset.replace('-', ' ').title():<40s} | "
              f"{'N/A':>10s} | {rolebreak:>10.4f} | {'N/A':>10s}")
    else:
        print(f"{dataset.replace('-', ' ').title():<40s} | "
              f"{'N/A':>10s} | {'N/A':>10s} | {'N/A':>10s}")
print("="*80)
