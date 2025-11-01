#!/usr/bin/env python3
"""
Create a single ROC curve comparing all 4 Claude judge modes on test pairs:
- final_only
- full_conversation
- full_with_system
- prompt_response_no_thinking
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# Base directory
BASE_DIR = Path("results-claude-sonnet-4.5/test-pairs-organism")

# Modes to compare (ordered by AUROC from highest to lowest)
MODES = {
    "score_based_v2_full_with_system": {
        "label": "Full with System Prompt (sanity check)",
        "color": "#9b59b6",  # Purple
        "linestyle": "--"
    },
    "score_based_v2_full_conversation": {
        "label": "Full Conversation (with thinking)",
        "color": "#e74c3c",  # Red
        "linestyle": "-"
    },
    "score_based_v2_final_only": {
        "label": "Final Only",
        "color": "#3498db",  # Blue
        "linestyle": "-"
    },
    "score_based_v2_prompt_response_no_thinking": {
        "label": "Prompt + Response (no thinking)",
        "color": "#2ecc71",  # Green
        "linestyle": "-"
    },
}


def load_results(mode):
    """Load results for a specific mode."""
    results_path = BASE_DIR / mode / "results.json"

    if not results_path.exists():
        print(f"⚠ Warning: {results_path} not found")
        return None, None

    with open(results_path, 'r') as f:
        results = json.load(f)

    # Filter valid results
    valid_results = [r for r in results
                    if r.get("prediction") is not None and r.get("prediction") != -1]

    if len(valid_results) == 0:
        print(f"⚠ Warning: No valid results for {mode}")
        return None, None

    predictions = np.array([r["prediction"] for r in valid_results])
    labels = np.array([r["true_label"] for r in valid_results])

    return labels, predictions


def create_combined_roc(output_path):
    """Create combined ROC curve for all modes."""
    plt.figure(figsize=(10, 8))

    # Plot ROC curve for each mode
    for mode, config in MODES.items():
        labels, predictions = load_results(mode)

        if labels is None or predictions is None:
            continue

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)

        # Plot
        linestyle = config.get("linestyle", "-")
        plt.plot(fpr, tpr, color=config["color"], lw=2.5, linestyle=linestyle,
                label=f'{config["label"]} (AUROC = {roc_auc:.3f})')

    # Plot random classifier line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random (AUROC = 0.500)')

    # Styling
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('Claude Judge: ROC Curves Across Context Modes\n(Test Pairs - Model Organism)',
              fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    print("="*80)
    print("Creating Combined ROC Curve for Claude Judge (Test Pairs)")
    print("="*80)

    # Check that base directory exists
    if not BASE_DIR.exists():
        print(f"⚠ Error: {BASE_DIR} not found")
        return

    # Print AUROC summary
    print("\nAUROC Summary:")
    print("-" * 60)
    for mode, config in MODES.items():
        labels, predictions = load_results(mode)
        if labels is not None and predictions is not None:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(labels, predictions)
            print(f"  {config['label']:40s}: {auroc:.4f}")

    # Create combined ROC curve
    print("\nCreating combined ROC curve...")
    output_file = BASE_DIR / "combined_roc_curve.png"
    create_combined_roc(output_file)

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()
