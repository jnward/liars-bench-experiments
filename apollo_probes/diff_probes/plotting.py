from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

DATASET_ORDER: Sequence[str] = (
    "convincing-game",
    "harm-pressure-knowledge-report",
    "instructed-deception",
    "insider-trading/report",
    "insider-trading/confirmation",
)

DATASET_LABELS: Dict[str, str] = {
    "convincing-game": "Convincing Game",
    "harm-pressure-knowledge-report": "Harm Pressure",
    "instructed-deception": "Instructed Deception",
    "insider-trading/report": "Insider Trading (report)",
    "insider-trading/confirmation": "Insider Trading (confirmation)",
}


def order_dataset_items(datasets: Dict[str, dict]) -> List[Tuple[str, dict]]:
    ordered: List[Tuple[str, dict]] = []
    seen = set()
    for key in DATASET_ORDER:
        if key in datasets:
            ordered.append((key, datasets[key]))
            seen.add(key)
    for key in sorted(datasets.keys()):
        if key not in seen:
            ordered.append((key, datasets[key]))
    return ordered


def display_name(dataset_key: str) -> str:
    return DATASET_LABELS.get(dataset_key, dataset_key.replace("-", " ").replace("_", " ").title())


def plot_combined_roc(variant: str, datasets: Dict[str, dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for dataset_key, metrics in order_dataset_items(datasets):
        scores = metrics.get("scores")
        labels = metrics.get("labels")
        curve = metrics.get("roc_curve")
        if scores is None or labels is None:
            continue
        scores_arr = np.array(scores)
        labels_arr = np.array(labels)
        if curve:
            fpr = np.array(curve.get("fpr", []))
            tpr = np.array(curve.get("tpr", []))
        else:
            if np.unique(labels_arr).size < 2:
                continue
            fpr, tpr, _ = roc_curve(labels_arr, scores_arr)
        if fpr.size == 0 or tpr.size == 0:
            continue
        auroc = metrics.get("auroc", float("nan"))
        ax.plot(fpr, tpr, label=f"{display_name(dataset_key)} (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{variant} probe ROC curves")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_combined_violin(variant: str, datasets: Dict[str, dict], output_path: Path) -> None:
    violin_data: List[List[float]] = []
    xtick_labels: List[str] = []
    dataset_boundaries: List[float] = []

    for dataset_key, metrics in order_dataset_items(datasets):
        scores = metrics.get("scores")
        labels = metrics.get("labels")
        if scores is None or labels is None:
            continue
        scores_arr = np.array(scores)
        labels_arr = np.array(labels)
        neg_scores = scores_arr[labels_arr == 0]
        pos_scores = scores_arr[labels_arr == 1]

        dataset_positions: List[int] = []
        if neg_scores.size:
            violin_data.append(neg_scores)
            xtick_labels.append(f"{display_name(dataset_key)}\nneg")
            dataset_positions.append(len(violin_data))
        if pos_scores.size:
            violin_data.append(pos_scores)
            xtick_labels.append(f"{display_name(dataset_key)}\npos")
            dataset_positions.append(len(violin_data))
        if dataset_positions:
            dataset_boundaries.append(dataset_positions[-1] + 0.5)

    if not violin_data:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(violin_data) * 0.9), 5))
    positions = np.arange(1, len(violin_data) + 1)
    parts = ax.violinplot(violin_data, positions=positions, showmeans=True, widths=0.8)
    for pc in parts["bodies"]:
        pc.set_facecolor("#87CEEB")
        pc.set_edgecolor("#1E90FF")
        pc.set_alpha(0.7)
    if "cmeans" in parts:
        parts["cmeans"].set_color("#FF8C00")
    ax.set_xticks(positions)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Probe logit score")
    ax.set_title(f"{variant} probe score distribution")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    for boundary in dataset_boundaries[:-1]:
        ax.axvline(boundary, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_combined_plots(results_dir: Path, summary: dict, variants: Iterable[str]) -> List[Path]:
    generated: List[Path] = []
    variant_info = summary.get("variants", {})
    for variant in variants:
        info = variant_info.get(variant)
        if not info:
            continue
        datasets = info.get("datasets", {})
        if not datasets:
            continue
        roc_out = results_dir / f"{variant}_combined_roc.png"
        violin_out = results_dir / f"{variant}_combined_violin.png"
        plot_combined_roc(variant, datasets, roc_out)
        plot_combined_violin(variant, datasets, violin_out)
        generated.extend([roc_out, violin_out])
    return generated
