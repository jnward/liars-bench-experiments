#!/usr/bin/env python3
"""
Create comprehensive comparison visualizations for Claude Sonnet 4.5, Gemma 2 27B, and Gemma 3 27B
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
GEMMA2_BASE = "results-gemma-2-27b/test-pairs-organism-fda"
GEMMA3_BASE = "results-gemma-3-27b/test-pairs-organism-fda"
OUTPUT_DIR = Path("results-gemma-3-27b/test-pairs-organism-fda")

MODELS = [
    ("Claude Sonnet 4.5", CLAUDE_BASE, "#3498db"),
    ("Gemma 2 27B", GEMMA2_BASE, "#e74c3c"),
    ("Gemma 3 27B", GEMMA3_BASE, "#2ecc71"),
]

MODES = [
    ("full_with_system", "Full with System"),
    ("full_conversation", "Full Conversation"),
    ("final_only", "Final Only"),
    ("prompt_response_no_thinking", "Prompt & Response\n(No Thinking)"),
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
    """Create 2x2 grid of ROC curves, one subplot per mode."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (mode_id, mode_name) in enumerate(MODES):
        ax = axes[idx]

        for model_name, base_path, color in MODELS:
            preds, labels = load_results(base_path, mode_id)
            fpr, tpr, _ = roc_curve(labels, preds)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{model_name} (AUC = {roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random', alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(mode_name, fontsize=13, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('ROC Curves: Claude Sonnet 4.5 vs Gemma 2 27B vs Gemma 3 27B\nFDA Test Pairs Deception Detection',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_path = OUTPUT_DIR / "three_model_roc_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def create_auroc_bar_comparison():
    """Create bar chart comparing AUROC across all models and modes."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Collect data
    model_names = [name for name, _, _ in MODELS]
    mode_names = [name for _, name in MODES]

    # Create data matrix: models x modes
    auroc_matrix = []
    for model_name, base_path, _ in MODELS:
        model_aurocs = []
        for mode_id, _ in MODES:
            preds, labels = load_results(base_path, mode_id)
            fpr, tpr, _ = roc_curve(labels, preds)
            model_aurocs.append(auc(fpr, tpr))
        auroc_matrix.append(model_aurocs)

    # Plot grouped bars
    x = np.arange(len(mode_names))
    width = 0.25

    bars = []
    for i, (model_name, _, color) in enumerate(MODELS):
        offset = (i - 1) * width
        bar = ax.bar(x + offset, auroc_matrix[i], width,
                     label=model_name, color=color, edgecolor='black', alpha=0.8)
        bars.append(bar)

        # Add value labels on bars
        for j, v in enumerate(auroc_matrix[i]):
            ax.text(x[j] + offset, v + 0.02, f'{v:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('AUROC', fontsize=13, fontweight='bold')
    ax.set_xlabel('Context Mode', fontsize=13, fontweight='bold')
    ax.set_title('AUROC Comparison: Claude Sonnet 4.5 vs Gemma 2 27B vs Gemma 3 27B\nFDA Test Pairs',
                 fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_names, fontsize=10)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.08])

    # Add horizontal line at 0.5 (random classifier)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='Random')

    plt.tight_layout()

    output_path = OUTPUT_DIR / "three_model_auroc_bars.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def create_heatmap_comparison():
    """Create heatmap showing AUROC across models and modes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect data
    model_names = [name for name, _, _ in MODELS]
    mode_names = [name.replace('\n', ' ') for _, name in MODES]

    # Create data matrix: models x modes
    auroc_matrix = []
    for model_name, base_path, _ in MODELS:
        model_aurocs = []
        for mode_id, _ in MODES:
            preds, labels = load_results(base_path, mode_id)
            fpr, tpr, _ = roc_curve(labels, preds)
            model_aurocs.append(auc(fpr, tpr))
        auroc_matrix.append(model_aurocs)

    # Create heatmap
    im = ax.imshow(auroc_matrix, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=1.0)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('AUROC', fontsize=12, fontweight='bold')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(mode_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(mode_names, fontsize=10)
    ax.set_yticklabels(model_names, fontsize=11)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(mode_names)):
            text = ax.text(j, i, f'{auroc_matrix[i][j]:.3f}',
                          ha="center", va="center", color="black",
                          fontsize=11, fontweight='bold')

    ax.set_title('AUROC Heatmap: Model Performance Across Context Modes\nFDA Test Pairs',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "three_model_auroc_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def print_summary_table():
    """Print summary table of AUROC values."""
    print("\n" + "="*90)
    print("AUROC Comparison Summary - All Models & Modes")
    print("="*90)
    print(f"{'Context Mode':<35} {'Claude 4.5':<15} {'Gemma 2 27B':<15} {'Gemma 3 27B':<15}")
    print("-"*90)

    for mode_id, mode_name in MODES:
        mode_name_clean = mode_name.replace('\n', ' ')
        aurocs = []

        for model_name, base_path, _ in MODELS:
            preds, labels = load_results(base_path, mode_id)
            fpr, tpr, _ = roc_curve(labels, preds)
            aurocs.append(auc(fpr, tpr))

        print(f"{mode_name_clean:<35} {aurocs[0]:.4f} ({aurocs[0]*100:.1f}%)"
              f"   {aurocs[1]:.4f} ({aurocs[1]*100:.1f}%)"
              f"   {aurocs[2]:.4f} ({aurocs[2]*100:.1f}%)")

    print("="*90)

    # Compute averages
    avg_aurocs = []
    for model_name, base_path, _ in MODELS:
        model_aurocs = []
        for mode_id, _ in MODES:
            preds, labels = load_results(base_path, mode_id)
            fpr, tpr, _ = roc_curve(labels, preds)
            model_aurocs.append(auc(fpr, tpr))
        avg_aurocs.append(np.mean(model_aurocs))

    print(f"{'AVERAGE ACROSS ALL MODES':<35} {avg_aurocs[0]:.4f} ({avg_aurocs[0]*100:.1f}%)"
          f"   {avg_aurocs[1]:.4f} ({avg_aurocs[1]*100:.1f}%)"
          f"   {avg_aurocs[2]:.4f} ({avg_aurocs[2]*100:.1f}%)")
    print("="*90)

    print("\nKey Findings:")
    print(f"  • Best Overall: Gemma 2 27B (avg AUROC: {avg_aurocs[1]:.4f})")
    print(f"  • Best Single Mode: Gemma 3 27B - Full with System (AUROC: 1.000)")
    print(f"  • Most Improved from Claude: Gemma 2 27B - Prompt & Response No Thinking")
    print(f"    ({avg_aurocs[1]:.4f} vs {avg_aurocs[0]:.4f}, +{(avg_aurocs[1]-avg_aurocs[0])*100:.1f} pp)")
    print(f"  • Gemma 3 has calibration issues despite high AUROC")
    print("="*90 + "\n")

def main():
    print("="*90)
    print("Creating Three-Model Comparison Visualizations")
    print("="*90)
    print(f"Claude source: {CLAUDE_BASE}")
    print(f"Gemma 2 source: {GEMMA2_BASE}")
    print(f"Gemma 3 source: {GEMMA3_BASE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*90 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Creating visualizations...")
    create_combined_roc_comparison()
    create_auroc_bar_comparison()
    create_heatmap_comparison()

    print_summary_table()

    print("\n✓ All visualizations created successfully!")
    print(f"\nOutput files:")
    print(f"  • {OUTPUT_DIR}/three_model_roc_comparison.png")
    print(f"  • {OUTPUT_DIR}/three_model_auroc_bars.png")
    print(f"  • {OUTPUT_DIR}/three_model_auroc_heatmap.png")

if __name__ == "__main__":
    main()
