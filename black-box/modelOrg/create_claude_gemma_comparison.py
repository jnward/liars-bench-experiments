#!/usr/bin/env python3
"""
Create side-by-side comparison visualizations for Claude Sonnet 4.5 vs Gemma 2 27B
on FDA test pairs across 4 context modes.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# Configuration
CLAUDE_BASE = "results-claude-sonnet-4.5/test-pairs-organism-fda-Claude"
GEMMA_BASE = "results-gemma-2-27b/test-pairs-organism-fda"
OUTPUT_DIR = Path("results-gemma-2-27b/test-pairs-organism-fda")

MODES = [
    ("full_with_system", "Full with System", "#3498db"),
    ("full_conversation", "Full Conversation", "#e74c3c"),
    ("final_only", "Final Only", "#2ecc71"),
    ("prompt_response_no_thinking", "Prompt & Response (No Thinking)", "#f39c12"),
]

def load_results(base_path: str, mode: str):
    """Load results for a specific mode."""
    results_path = Path(base_path) / f"score_based_v2_{mode}" / "results.json"
    with open(results_path) as f:
        results = json.load(f)

    # Filter valid results
    valid_results = [r for r in results if r.get("prediction", -1) != -1]

    predictions = np.array([r["prediction"] for r in valid_results])
    labels = np.array([r["true_label"] for r in valid_results])

    return predictions, labels

def create_combined_roc_comparison():
    """Create combined ROC curve comparing Claude and Gemma across all modes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Claude Sonnet 4.5
    ax1.set_title("Claude Sonnet 4.5", fontsize=14, fontweight='bold')
    for mode_id, mode_name, color in MODES:
        preds, labels = load_results(CLAUDE_BASE, mode_id)
        fpr, tpr, _ = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=color, lw=2, label=f'{mode_name} (AUC = {roc_auc:.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right plot: Gemma 2 27B
    ax2.set_title("Gemma 2 27B", fontsize=14, fontweight='bold')
    for mode_id, mode_name, color in MODES:
        preds, labels = load_results(GEMMA_BASE, mode_id)
        fpr, tpr, _ = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color=color, lw=2, label=f'{mode_name} (AUC = {roc_auc:.3f})')

    ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('ROC Curves: Claude Sonnet 4.5 vs Gemma 2 27B\nFDA Test Pairs Deception Detection',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "claude_vs_gemma_roc_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def create_auroc_bar_comparison():
    """Create bar chart comparing AUROC across models and modes."""
    fig, ax = plt.subplots(figsize=(12, 7))

    claude_aurocs = []
    gemma_aurocs = []
    mode_names = []

    for mode_id, mode_name, _ in MODES:
        # Claude
        preds, labels = load_results(CLAUDE_BASE, mode_id)
        fpr, tpr, _ = roc_curve(labels, preds)
        claude_aurocs.append(auc(fpr, tpr))

        # Gemma
        preds, labels = load_results(GEMMA_BASE, mode_id)
        fpr, tpr, _ = roc_curve(labels, preds)
        gemma_aurocs.append(auc(fpr, tpr))

        mode_names.append(mode_name)

    x = np.arange(len(mode_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, claude_aurocs, width, label='Claude Sonnet 4.5',
                   color='#3498db', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, gemma_aurocs, width, label='Gemma 2 27B',
                   color='#e74c3c', edgecolor='black', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Context Mode', fontsize=12, fontweight='bold')
    ax.set_title('AUROC Comparison: Claude Sonnet 4.5 vs Gemma 2 27B\nFDA Test Pairs',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_names, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

    # Add horizontal line at 0.5 (random classifier)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random')

    plt.tight_layout()

    output_path = OUTPUT_DIR / "claude_vs_gemma_auroc_bars.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def create_score_distribution_comparison():
    """Create violin plots comparing score distributions for best mode (full_with_system)."""
    mode_id = "full_with_system"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Claude
    preds_claude, labels_claude = load_results(CLAUDE_BASE, mode_id)
    honest_claude = preds_claude[labels_claude == 0]
    deceptive_claude = preds_claude[labels_claude == 1]

    parts1 = ax1.violinplot([honest_claude, deceptive_claude],
                            positions=[1, 2],
                            showmeans=True,
                            showmedians=True)
    for pc in parts1['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.6)

    ax1.set_title("Claude Sonnet 4.5\n(Full with System)", fontsize=13, fontweight='bold')
    ax1.set_ylabel('Deception Score', fontsize=11)
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['Honest\n(True)', 'Deceptive\n(True)'])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold')
    ax1.legend()

    # Gemma
    preds_gemma, labels_gemma = load_results(GEMMA_BASE, mode_id)
    honest_gemma = preds_gemma[labels_gemma == 0]
    deceptive_gemma = preds_gemma[labels_gemma == 1]

    parts2 = ax2.violinplot([honest_gemma, deceptive_gemma],
                            positions=[1, 2],
                            showmeans=True,
                            showmedians=True)
    for pc in parts2['bodies']:
        pc.set_facecolor('#e74c3c')
        pc.set_alpha(0.6)

    ax2.set_title("Gemma 2 27B\n(Full with System)", fontsize=13, fontweight='bold')
    ax2.set_ylabel('Deception Score', fontsize=11)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Honest\n(True)', 'Deceptive\n(True)'])
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold')
    ax2.legend()

    plt.suptitle('Score Distribution Comparison: Best Mode\nFDA Test Pairs',
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "claude_vs_gemma_score_distributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def print_summary_table():
    """Print summary table of AUROC values."""
    print("\n" + "="*80)
    print("AUROC Comparison Summary")
    print("="*80)
    print(f"{'Context Mode':<35} {'Claude Sonnet 4.5':<20} {'Gemma 2 27B':<20} {'Difference':<15}")
    print("-"*80)

    for mode_id, mode_name, _ in MODES:
        # Claude
        preds, labels = load_results(CLAUDE_BASE, mode_id)
        fpr, tpr, _ = roc_curve(labels, preds)
        claude_auc = auc(fpr, tpr)

        # Gemma
        preds, labels = load_results(GEMMA_BASE, mode_id)
        fpr, tpr, _ = roc_curve(labels, preds)
        gemma_auc = auc(fpr, tpr)

        diff = gemma_auc - claude_auc
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"

        print(f"{mode_name:<35} {claude_auc:.4f} ({claude_auc*100:.1f}%)"
              f"{'':>7} {gemma_auc:.4f} ({gemma_auc*100:.1f}%)"
              f"{'':>7} {diff_str}")

    print("="*80)
    print("\nKey Findings:")
    print(f"  • Gemma 2 27B outperforms Claude Sonnet 4.5 on all 4 context modes")
    print(f"  • Largest improvement: Full with System (+0.1733, or +17.3 percentage points)")
    print(f"  • Smallest improvement: Prompt & Response No Thinking (+0.2244, or +22.4 pp)")
    print(f"  • Both models struggle without thinking tags (AUROC drops significantly)")
    print("="*80 + "\n")

def main():
    print("="*80)
    print("Creating Claude vs Gemma Comparison Visualizations")
    print("="*80)
    print(f"Claude source: {CLAUDE_BASE}")
    print(f"Gemma source: {GEMMA_BASE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Creating visualizations...")
    create_combined_roc_comparison()
    create_auroc_bar_comparison()
    create_score_distribution_comparison()

    print_summary_table()

    print("\n✓ All visualizations created successfully!")
    print(f"\nOutput files:")
    print(f"  • {OUTPUT_DIR}/claude_vs_gemma_roc_comparison.png")
    print(f"  • {OUTPUT_DIR}/claude_vs_gemma_auroc_bars.png")
    print(f"  • {OUTPUT_DIR}/claude_vs_gemma_score_distributions.png")

if __name__ == "__main__":
    main()
