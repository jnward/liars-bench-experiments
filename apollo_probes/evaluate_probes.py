from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve

from .config import DEFAULT_DATASET
from .eval_datasets import EVAL_DATASETS, slugify_config
from .paths import (
    dataset_probe_dir,
    eval_layer_cache_path,
    probe_layer_dir,
    results_layer_dir,
)
from .pooling import pool_dialogue_activations


def list_trained_layers(dataset: str) -> list[int]:
    probe_root = dataset_probe_dir(dataset)
    layers: list[int] = []
    for path in probe_root.glob("layer*/probe.pkl"):
        try:
            layer = int(path.parent.name.replace("layer", ""))
            layers.append(layer)
        except ValueError:
            continue
    return sorted(set(layers))


def load_probe_from_path(path: Path, layer: int | None = None) -> dict:
    try:
        probe = torch.load(path, map_location="cpu", weights_only=False)
    except (RuntimeError, pickle.UnpicklingError, AttributeError):
        with path.open("rb") as f:
            probe = pickle.load(f)
    if layer is not None and "layers" in probe:
        layers = list(probe["layers"])
        try:
            idx = layers.index(layer)
        except ValueError as exc:
            raise ValueError(f"Layer {layer} not found in probe {path}") from exc
        direction = torch.tensor(probe["directions"][idx], dtype=torch.float32)
        out = {
            "directions": direction.unsqueeze(0),
            "normalize": probe.get("normalize", False),
            "scaler_mean": probe.get("scaler_mean", [None])[idx],
            "scaler_scale": probe.get("scaler_scale", [None])[idx],
            "logistic_intercept": probe.get("logistic_intercept", [0.0])[idx] if probe.get("logistic_intercept") is not None else 0.0,
        }
        return out
    return probe


def load_probe(dataset: str, layer: int, external_path: Path | None = None, external_layer: int | None = None) -> dict:
    if external_path is not None:
        target_layer = external_layer if external_layer is not None else layer
        return load_probe_from_path(external_path, target_layer)
    probe_path = probe_layer_dir(dataset, layer) / "probe.pkl"
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe not found: {probe_path}")
    return torch.load(probe_path, map_location="cpu")


def load_eval_cache(dataset_alias: str, layer: int) -> dict:
    cache_path = eval_layer_cache_path(dataset_alias, layer)
    if not cache_path.exists():
        raise FileNotFoundError(f"Eval cache missing: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu")
    return payload


def tensorize(value) -> torch.Tensor:
    if value is None:
        return torch.tensor(0.0)
    if isinstance(value, torch.Tensor):
        return value.to(torch.float32)
    return torch.tensor(value, dtype=torch.float32)


def compute_scores(acts: torch.Tensor, probe: dict) -> np.ndarray:
    direction = tensorize(probe["directions"]).view(-1)
    intercept = tensorize(probe.get("logistic_intercept", 0.0)).view(-1)[0].item()
    if probe.get("normalize", False):
        mean = tensorize(probe.get("scaler_mean"))
        scale = tensorize(probe.get("scaler_scale"))
        acts = (acts - mean) / scale
    scores = acts @ direction
    scores = scores + float(intercept)
    return scores.numpy()


def optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return 0.0
    f1_scores = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-8, None)
    best_idx = int(np.nanargmax(f1_scores))
    return float(thresholds[best_idx])


def plot_roc(layer: int, train_dataset: str, metrics: dict, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for name, data in metrics.items():
        fpr = data["roc"]["fpr"]
        tpr = data["roc"]["tpr"]
        auroc = data["auroc"]
        plt.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {train_dataset} Layer {layer:02d}")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_violin(layer: int, train_dataset: str, metrics: dict, output_path: Path) -> None:
    positions = []
    data = []
    labels = []
    offset = 1
    spacing = 1.5
    width = 0.35
    for name, info in metrics.items():
        scores = info["scores"]
        labels_arr = info["labels"]
        neg_scores = scores[labels_arr == 0]
        pos_scores = scores[labels_arr == 1]
        if neg_scores.size == 0 or pos_scores.size == 0:
            continue
        positions.extend([offset - width / 2, offset + width / 2])
        data.extend([neg_scores, pos_scores])
        labels.extend([f"{name}\nneg", f"{name}\npos"])
        offset += spacing

    if not data:
        print(f"[eval] Skipping violin plot for layer {layer}: insufficient class balance.")
        return

    plt.figure(figsize=(max(10, len(data) * 0.8), 6))
    parts = plt.violinplot(data, positions=positions, widths=width, showmeans=False, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.7)
    for name, collection in parts.items():
        if name == "bodies":
            continue
        collection.set_alpha(0.5)

    plt.axhline(0.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.xticks(ticks=positions, labels=labels, rotation=45, ha="right")
    plt.ylabel("Probe scores")
    plt.title(f"Score Distribution - {train_dataset} Layer {layer:02d}")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate_layer(
    train_dataset: str,
    layer: int,
    eval_configs: list[str],
    external_probe: Path | None = None,
    external_layer: int | None = None,
    probe_name: str | None = None,
) -> None:
    probe = load_probe(train_dataset, layer, external_path=external_probe, external_layer=external_layer)
    plot_inputs: dict[str, dict] = {}
    metrics_summary: dict[str, dict] = {}
    combined_scores = []
    combined_labels = []

    for config in eval_configs:
        alias = slugify_config(config)
        payload = load_eval_cache(alias, layer)
        counts = payload.get("counts")
        dialogue_labels = payload.get("dialogue_labels")
        acts = payload["activations"]
        if counts is None or dialogue_labels is None:
            print(f"[eval] Warning: {alias} cache missing counts/dialogue labels; regenerate.")
            continue
        pooled, pooled_labels, skipped = pool_dialogue_activations(acts, counts, dialogue_labels)
        if pooled.numel() == 0 or len(np.unique(pooled_labels.numpy())) < 2:
            print(f"[eval] Warning: {alias} layer {layer} lacks sufficient class variety; skipping.")
            continue
        scores = compute_scores(pooled.to(torch.float32), probe)
        labels = pooled_labels.numpy()

        roc = roc_curve(labels, scores)
        auroc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")
        plot_inputs[alias] = {
            "scores": scores,
            "labels": labels,
            "roc": {"fpr": roc[0], "tpr": roc[1]},
            "auroc": float(auroc),
            "skipped_dialogues": skipped,
        }
        metrics_summary[alias] = {"auroc": float(auroc)}
        combined_scores.append(scores)
        combined_labels.append(labels)

    if not plot_inputs:
        print(f"[eval] No eval data available for layer {layer}; skipping.")
        return

    all_scores = np.concatenate(combined_scores)
    all_labels = np.concatenate(combined_labels)
    threshold = optimal_threshold(all_scores, all_labels)

    aggregate_f1s = []
    for alias, info in plot_inputs.items():
        labels_arr = info["labels"]
        scores_arr = info["scores"]
        preds = (scores_arr >= threshold).astype(int)
        accuracy = float(accuracy_score(labels_arr, preds))
        f1 = float(f1_score(labels_arr, preds, zero_division=0)) if len(np.unique(labels_arr)) > 1 else float("nan")
        metrics_summary[alias]["accuracy"] = accuracy
        metrics_summary[alias]["f1"] = f1
        aggregate_f1s.append(f1)

    avg_f1 = float(np.nanmean(aggregate_f1s)) if aggregate_f1s else float("nan")

    if external_probe is not None:
        tag = probe_name or external_probe.stem
        results_dir = results_layer_dir(train_dataset, layer) / tag
    else:
        results_dir = results_layer_dir(train_dataset, layer)
    results_dir.mkdir(parents=True, exist_ok=True)
    roc_path = results_dir / "eval_roc.png"
    violin_path = results_dir / "eval_violin.png"
    plot_roc(layer, train_dataset, plot_inputs, roc_path)
    plot_violin(layer, train_dataset, plot_inputs, violin_path)

    metrics_path = results_dir / "eval_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "train_dataset": train_dataset,
                "layer_index": layer,
                "threshold": threshold,
                "average_f1": avg_f1,
                "eval_datasets": metrics_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[eval] Saved plots and metrics for layer {layer}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Apollo probes on Cadenza datasets.")
    parser.add_argument("--train-dataset", default=DEFAULT_DATASET, help="Training dataset used for probes.")
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="Specific layers to evaluate.")
    parser.add_argument("--eval-datasets", choices=EVAL_DATASETS, nargs="+", default=EVAL_DATASETS)
    parser.add_argument("--probe-path", type=Path, default=None, help="External probe path (optional).")
    parser.add_argument("--probe-layer", type=int, default=None, help="Layer index inside external probe (defaults to eval layer).")
    parser.add_argument("--probe-name", type=str, default=None, help="Custom name for external probe outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = args.layers or list_trained_layers(args.train_dataset)
    if not layers:
        raise SystemExit(f"No trained probes found for dataset {args.train_dataset}.")
    for layer in layers:
        evaluate_layer(
            args.train_dataset,
            layer,
            args.eval_datasets,
            external_probe=args.probe_path,
            external_layer=args.probe_layer,
            probe_name=args.probe_name,
        )


if __name__ == "__main__":
    main()
