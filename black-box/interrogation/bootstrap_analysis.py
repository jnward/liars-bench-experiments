"""
Bootstrap analysis for interrogation experiment results.

Computes confidence intervals for metrics without re-running expensive API calls.
Can also compare two methods (e.g., direct vs rolebreak) using paired bootstrap.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm


def load_results(results_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load predictions and labels from results.json.

    Returns:
        (predictions, labels) as numpy arrays
    """
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Filter valid results
    valid_results = [r for r in results if r.get("prediction", -1) >= 0]

    predictions = np.array([r["prediction"] for r in valid_results])
    labels = np.array([r["true_label"] for r in valid_results])

    return predictions, labels


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute all metrics from predictions and labels."""
    binary_predictions = (predictions >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, binary_predictions),
        "precision": precision_score(labels, binary_predictions, zero_division=0),
        "recall": recall_score(labels, binary_predictions, zero_division=0),
        "f1": f1_score(labels, binary_predictions, zero_division=0),
    }

    # Add AUROC
    try:
        metrics["auroc"] = roc_auc_score(labels, predictions)
    except:
        metrics["auroc"] = np.nan

    return metrics


def bootstrap_single(predictions: np.ndarray, labels: np.ndarray,
                     n_iterations: int, confidence_level: float,
                     seed: int) -> Dict[str, Dict]:
    """
    Bootstrap analysis for a single set of results.

    Returns:
        Dict with metrics, each containing 'mean', 'ci_lower', 'ci_upper', 'distribution'
    """
    np.random.seed(seed)
    n_samples = len(predictions)

    # Store bootstrap distributions
    bootstrap_distributions = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auroc": []
    }

    print(f"Running bootstrap with {n_iterations} iterations...")
    for _ in tqdm(range(n_iterations), desc="Bootstrap sampling"):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_predictions = predictions[indices]
        boot_labels = labels[indices]

        # Compute metrics on bootstrap sample
        metrics = compute_metrics(boot_predictions, boot_labels)

        for metric_name, value in metrics.items():
            if not np.isnan(value):
                bootstrap_distributions[metric_name].append(value)

    # Compute confidence intervals
    alpha = 1 - confidence_level
    results = {}

    for metric_name, distribution in bootstrap_distributions.items():
        distribution = np.array(distribution)
        results[metric_name] = {
            "mean": np.mean(distribution),
            "median": np.median(distribution),
            "std": np.std(distribution),
            "ci_lower": np.percentile(distribution, 100 * alpha / 2),
            "ci_upper": np.percentile(distribution, 100 * (1 - alpha / 2)),
            "distribution": distribution
        }

    return results


def bootstrap_comparison(predictions1: np.ndarray, labels1: np.ndarray,
                        predictions2: np.ndarray, labels2: np.ndarray,
                        n_iterations: int, confidence_level: float,
                        seed: int) -> Dict[str, Dict]:
    """
    Bootstrap comparison between two methods using paired bootstrap.

    Returns:
        Dict with difference statistics for each metric
    """
    np.random.seed(seed)
    n_samples = min(len(predictions1), len(predictions2))

    # Store bootstrap distributions of differences
    difference_distributions = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auroc": []
    }

    print(f"Running paired bootstrap with {n_iterations} iterations...")
    for _ in tqdm(range(n_iterations), desc="Bootstrap comparison"):
        # Resample with replacement (same indices for both)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        boot_pred1 = predictions1[indices]
        boot_label1 = labels1[indices]
        boot_pred2 = predictions2[indices]
        boot_label2 = labels2[indices]

        # Compute metrics for both
        metrics1 = compute_metrics(boot_pred1, boot_label1)
        metrics2 = compute_metrics(boot_pred2, boot_label2)

        # Store differences
        for metric_name in difference_distributions.keys():
            if not (np.isnan(metrics1[metric_name]) or np.isnan(metrics2[metric_name])):
                diff = metrics2[metric_name] - metrics1[metric_name]
                difference_distributions[metric_name].append(diff)

    # Compute statistics on differences
    alpha = 1 - confidence_level
    results = {}

    for metric_name, distribution in difference_distributions.items():
        distribution = np.array(distribution)

        # P-value: proportion of bootstrap samples where difference crosses zero
        p_value = np.mean(distribution <= 0) if np.mean(distribution) > 0 else np.mean(distribution >= 0)
        p_value = 2 * min(p_value, 1 - p_value)  # Two-tailed

        results[metric_name] = {
            "mean_diff": np.mean(distribution),
            "median_diff": np.median(distribution),
            "std_diff": np.std(distribution),
            "ci_lower": np.percentile(distribution, 100 * alpha / 2),
            "ci_upper": np.percentile(distribution, 100 * (1 - alpha / 2)),
            "p_value": p_value,
            "distribution": distribution
        }

    return results


def plot_single_distributions(bootstrap_results: Dict, output_path: Path):
    """Create violin plots showing bootstrap distributions for each metric."""
    metrics = ["accuracy", "precision", "recall", "f1", "auroc"]

    fig, axes = plt.subplots(1, 5, figsize=(24, 6))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data = bootstrap_results[metric]["distribution"]

        # Violin plot
        parts = ax.violinplot([data], positions=[0], widths=0.7,
                             showmeans=True, showmedians=True)

        # Color the violin
        for pc in parts['bodies']:
            pc.set_facecolor('#3498db')
            pc.set_alpha(0.7)

        # Add horizontal lines for CI
        ci_lower = bootstrap_results[metric]["ci_lower"]
        ci_upper = bootstrap_results[metric]["ci_upper"]
        mean = bootstrap_results[metric]["mean"]

        ax.hlines(ci_lower, -0.3, 0.3, colors='red', linestyles='dashed', linewidth=2, label='95% CI')
        ax.hlines(ci_upper, -0.3, 0.3, colors='red', linestyles='dashed', linewidth=2)
        ax.hlines(mean, -0.4, 0.4, colors='black', linewidths=3, label='Mean')

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_title(f'{metric.upper()}\n{mean:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]',
                    fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')

        if idx == 0:
            ax.legend(fontsize=10, loc='upper left')

    plt.suptitle('Bootstrap Distributions with 95% Confidence Intervals',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved distribution plot: {output_path}")


def plot_comparison(comparison_results: Dict, method1_name: str, method2_name: str,
                   output_path: Path):
    """Create plots showing differences between two methods."""
    metrics = ["accuracy", "precision", "recall", "f1", "auroc"]

    fig, axes = plt.subplots(1, 5, figsize=(24, 6))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data = comparison_results[metric]["distribution"]

        # Violin plot of differences
        parts = ax.violinplot([data], positions=[0], widths=0.7,
                             showmeans=True, showmedians=True)

        # Color based on whether difference is significant
        p_value = comparison_results[metric]["p_value"]
        color = '#e74c3c' if p_value < 0.05 else '#95a5a6'

        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        # Add horizontal line at zero
        ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.6)

        # Add CI lines
        ci_lower = comparison_results[metric]["ci_lower"]
        ci_upper = comparison_results[metric]["ci_upper"]
        mean_diff = comparison_results[metric]["mean_diff"]

        ax.hlines(ci_lower, -0.3, 0.3, colors='blue', linestyles='dashed', linewidth=2)
        ax.hlines(ci_upper, -0.3, 0.3, colors='blue', linestyles='dashed', linewidth=2)
        ax.hlines(mean_diff, -0.4, 0.4, colors='black', linewidths=3)

        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylabel('Difference', fontsize=13, fontweight='bold')

        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        ax.set_title(f'{metric.upper()}\nΔ={mean_diff:+.4f} (p={p_value:.3f}) {sig_marker}',
                    fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Method Comparison: {method2_name} - {method1_name}\n(Red = significant at p<0.05, Gray = not significant)',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved comparison plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap analysis for interrogation results")
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Path to results directory (e.g., results/interrogation/insider-trading/direct/)"
    )
    parser.add_argument(
        "--results-dir-1",
        type=str,
        help="Path to first results directory for comparison"
    )
    parser.add_argument(
        "--results-dir-2",
        type=str,
        help="Path to second results directory for comparison"
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=10000,
        help="Number of bootstrap iterations (default: 10000)"
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for CIs (default: 0.95)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Determine mode: single analysis or comparison
    if args.results_dir:
        mode = "single"
        results_dir = Path(args.results_dir)
        results_file = results_dir / "results.json"

        if not results_file.exists():
            print(f"Error: {results_file} not found")
            return

    elif args.results_dir_1 and args.results_dir_2:
        mode = "comparison"
        results_dir_1 = Path(args.results_dir_1)
        results_dir_2 = Path(args.results_dir_2)
        results_file_1 = results_dir_1 / "results.json"
        results_file_2 = results_dir_2 / "results.json"

        if not results_file_1.exists():
            print(f"Error: {results_file_1} not found")
            return
        if not results_file_2.exists():
            print(f"Error: {results_file_2} not found")
            return
    else:
        print("Error: Must provide either --results-dir OR both --results-dir-1 and --results-dir-2")
        return

    print("="*80)
    print("Bootstrap Analysis")
    print("="*80)
    print(f"Mode: {mode}")
    print(f"Iterations: {args.n_iterations}")
    print(f"Confidence level: {args.confidence_level}")
    print(f"Seed: {args.seed}")
    print()

    if mode == "single":
        # Single analysis
        print(f"Loading results from: {results_file}")
        predictions, labels = load_results(results_file)
        print(f"  Loaded {len(predictions)} valid examples")
        print(f"  Deceptive: {np.sum(labels == 1)}, Honest: {np.sum(labels == 0)}")
        print()

        # Run bootstrap
        bootstrap_results = bootstrap_single(
            predictions, labels,
            args.n_iterations,
            args.confidence_level,
            args.seed
        )

        # Save results
        output_file = results_dir / "bootstrap_results.json"
        save_data = {}
        for metric, stats in bootstrap_results.items():
            save_data[metric] = {
                "mean": float(stats["mean"]),
                "median": float(stats["median"]),
                "std": float(stats["std"]),
                "ci_lower": float(stats["ci_lower"]),
                "ci_upper": float(stats["ci_upper"])
            }

        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\n✓ Saved bootstrap results: {output_file}")

        # Print summary
        print("\n" + "="*80)
        print("RESULTS WITH 95% CONFIDENCE INTERVALS")
        print("="*80)
        for metric in ["accuracy", "precision", "recall", "f1", "auroc"]:
            stats = bootstrap_results[metric]
            print(f"{metric.upper():12s}: {stats['mean']:.4f} [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")

        # Create visualization
        print("\n" + "="*80)
        print("Creating visualization...")
        print("="*80)
        plot_path = results_dir / "bootstrap_distributions.png"
        plot_single_distributions(bootstrap_results, plot_path)

    else:
        # Comparison analysis
        print(f"Loading results from:")
        print(f"  Method 1: {results_file_1}")
        print(f"  Method 2: {results_file_2}")

        predictions1, labels1 = load_results(results_file_1)
        predictions2, labels2 = load_results(results_file_2)

        print(f"  Method 1: {len(predictions1)} valid examples")
        print(f"  Method 2: {len(predictions2)} valid examples")
        print()

        # Run paired bootstrap
        comparison_results = bootstrap_comparison(
            predictions1, labels1,
            predictions2, labels2,
            args.n_iterations,
            args.confidence_level,
            args.seed
        )

        # Save results
        output_dir = results_dir_1.parent
        output_file = output_dir / "bootstrap_comparison.json"
        save_data = {}
        for metric, stats in comparison_results.items():
            save_data[metric] = {
                "mean_diff": float(stats["mean_diff"]),
                "median_diff": float(stats["median_diff"]),
                "std_diff": float(stats["std_diff"]),
                "ci_lower": float(stats["ci_lower"]),
                "ci_upper": float(stats["ci_upper"]),
                "p_value": float(stats["p_value"])
            }

        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\n✓ Saved comparison results: {output_file}")

        # Print summary
        method1_name = results_dir_1.name
        method2_name = results_dir_2.name

        print("\n" + "="*80)
        print(f"COMPARISON: {method2_name} - {method1_name}")
        print("="*80)
        for metric in ["accuracy", "precision", "recall", "f1", "auroc"]:
            stats = comparison_results[metric]
            sig = "***" if stats['p_value'] < 0.001 else "**" if stats['p_value'] < 0.01 else "*" if stats['p_value'] < 0.05 else "ns"
            print(f"{metric.upper():12s}: Δ={stats['mean_diff']:+.4f} [{stats['ci_lower']:+.4f}, {stats['ci_upper']:+.4f}] (p={stats['p_value']:.4f}) {sig}")

        # Create visualization
        print("\n" + "="*80)
        print("Creating comparison visualization...")
        print("="*80)
        plot_path = output_dir / "bootstrap_comparison.png"
        plot_comparison(comparison_results, method1_name, method2_name, plot_path)

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()
