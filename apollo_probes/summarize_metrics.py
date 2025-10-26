from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from .eval_datasets import EVAL_DATASETS, slugify_config


def load_metrics(results_dir: Path) -> dict[str, dict[str, float]]:
    layer_data: dict[str, dict[str, float]] = {}
    for layer_dir in sorted(results_dir.glob("layer*/")):
        metrics_path = layer_dir / "eval_metrics.json"
        if not metrics_path.exists():
            continue
        data = json.loads(metrics_path.read_text())
        entries = data.get("eval_datasets", {})
        per_dataset: dict[str, float] = {}
        for dataset, info in entries.items():
            auroc = info.get("auroc")
            if isinstance(auroc, (int, float)) and auroc == auroc:  # skip NaN
                per_dataset[dataset] = float(auroc)
        if per_dataset:
            layer_data[layer_dir.name] = per_dataset
    return layer_data


def plot_metrics(layer_data: dict[str, dict[str, float]], output_dir: Path) -> None:
    if not layer_data:
        print("No metrics to plot.")
        return

    layers = sorted(layer_data.keys())
    dataset_names = sorted({ds for values in layer_data.values() for ds in values.keys()})

    sums = [sum(layer_data[layer].get(ds, 0.0) for ds in dataset_names) for layer in layers]

    plt.figure(figsize=(10, 5))
    plt.plot(layers, sums, marker="o", label="Sum AUROC")
    plt.xticks(rotation=45)
    plt.ylabel("Sum AUROC")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    sum_path = output_dir / "auroc_sum.png"
    plt.savefig(sum_path, dpi=200)
    plt.close()
    print(f"Saved sum plot -> {sum_path}")

    plt.figure(figsize=(10, 6))
    for dataset in dataset_names:
        values = [layer_data[layer].get(dataset, float("nan")) for layer in layers]
        plt.plot(layers, values, marker="o", label=dataset)
    plt.xticks(rotation=45)
    plt.ylabel("AUROC")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    dataset_path = output_dir / "auroc_by_dataset.png"
    plt.savefig(dataset_path, dpi=200)
    plt.close()
    print(f"Saved dataset plot -> {dataset_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AUROC metrics across layers.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("apollo_probes/results/repe_honesty__plain"),
        help="Root directory containing per-layer eval_metrics.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("apollo_probes/results/repe_honesty__plain/plots"),
        help="Directory to save plots.",
    )
    args = parser.parse_args()

    layer_data = load_metrics(args.results_root)
    if not layer_data:
        print("No metrics found.")
        return

    layer_data = {layer: layer_data[layer] for layer in sorted(layer_data)}
    print("Layer\tSum_AUROC")
    totals: dict[str, float] = {}
    for layer, per_dataset in layer_data.items():
        total = sum(per_dataset.values())
        totals[layer] = total
        print(f"{layer}\t{total:.4f}")

    best_layer = max(totals, key=totals.get)
    print(f"\nBest layer by AUROC sum: {best_layer} ({totals[best_layer]:.4f})")

    plot_metrics(layer_data, args.output_dir)


if __name__ == "__main__":
    main()
