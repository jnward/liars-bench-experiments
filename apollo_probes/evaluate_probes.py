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

from .config import DEFAULT_DATASET, DEFAULT_APOLLO_EVALS
from .eval_datasets import EVAL_DATASETS, slugify_config
from .paths import (
    dataset_probe_dir,
    eval_layer_cache_path,
    probe_layer_dir,
    results_layer_dir,
)
from .pooling import pool_dialogue_activations
from .data import prepare_datasets
from .beaver_eval import (
    BEAVER_DEFAULT_SPLIT,
    BEAVER_POSITIVE_CATEGORIES,
    BEAVER_NEGATIVE_TAG,
    beaver_slug,
)


def _default_violin_groups(scores: np.ndarray, labels: np.ndarray) -> tuple[dict[str, list[float]], list[str]]:
    neg_scores = scores[labels == 0]
    pos_scores = scores[labels == 1]
    groups = {
        "negative": neg_scores.tolist(),
        "positive": pos_scores.tolist(),
    }
    order = ["negative", "positive"]
    return groups, order


def _beaver_violin_groups(
    scores: np.ndarray,
    labels: np.ndarray,
    categories: Sequence[Sequence[str]],
) -> tuple[dict[str, list[float]], list[str]]:
    groups: dict[str, list[float]] = {BEAVER_NEGATIVE_TAG: []}
    order: list[str] = [BEAVER_NEGATIVE_TAG]
    for cat in BEAVER_POSITIVE_CATEGORIES:
        groups[cat] = []
        order.append(cat)

    for score, label, cats in zip(scores, labels, categories, strict=False):
        cats = list(cats or [])
        if label == 0:
            groups[BEAVER_NEGATIVE_TAG].append(float(score))
            continue
        matched = False
        for cat in BEAVER_POSITIVE_CATEGORIES:
            if cat in cats:
                groups[cat].append(float(score))
                matched = True
        if not matched:
            groups.setdefault("other_positive", []).append(float(score))
            if "other_positive" not in order:
                order.append("other_positive")

    # Remove empty groups while preserving order
    filtered_groups = {name: vals for name, vals in groups.items() if vals}
    filtered_order = [name for name in order if name in filtered_groups]
    return filtered_groups, filtered_order
from .beaver_eval import (
    BEAVER_DEFAULT_SPLIT,
    BEAVER_POSITIVE_CATEGORIES,
    BEAVER_NEGATIVE_TAG,
    beaver_slug,
)


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
    positions: list[float] = []
    data: list[Sequence[float]] = []
    tick_labels: list[str] = []
    offset = 1.0
    spacing = 1.6
    width = 0.35

    for name, info in metrics.items():
        groups = info.get("violin_groups") or {}
        order = info.get("violin_order") or list(groups.keys())
        valid_order = [label for label in order if groups.get(label)]
        if not valid_order:
            continue
        total = len(valid_order)
        start = offset - width * (total - 1) / 2
        for idx, label in enumerate(valid_order):
            values = groups[label]
            if not values:
                continue
            pos = start + idx * width
            positions.append(pos)
            data.append(values)
            tick_labels.append(f"{name}\n{label}")
        offset += spacing

    if not data:
        print(f"[eval] Skipping violin plot for layer {layer}: insufficient data.")
        return

    flat_scores = [score for bucket in data for score in bucket]
    global_min = min(flat_scores)
    global_max = max(flat_scores)
    span = max(global_max - global_min, 1e-6)
    margin = max(0.05 * span, 1e-3)

    plt.figure(figsize=(max(10, len(data) * 0.8), 6))
    parts = plt.violinplot(data, positions=positions, widths=width * 0.9, showmeans=False, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.7)
    for pname, collection in parts.items():
        if pname == "bodies":
            continue
        collection.set_alpha(0.5)

    if global_min <= 0.0 <= global_max:
        plt.axhline(0.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.ylim(global_min - margin, global_max + margin)
    plt.xticks(ticks=positions, labels=tick_labels, rotation=45, ha="right")
    plt.ylabel("Probe scores")
    plt.title(f"Score Distribution - {train_dataset} Layer {layer:02d}")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate_layer(
    dataset_slug: str,
    layer: int,
    eval_targets: list[tuple[str, str]],
    external_probe: Path | None = None,
    external_layer: int | None = None,
    probe_name: str | None = None,
) -> None:
    probe = load_probe(dataset_slug, layer, external_path=external_probe, external_layer=external_layer)
    plot_inputs: dict[str, dict] = {}
    metrics_summary: dict[str, dict] = {}
    combined_scores = []
    combined_labels = []

    seen_keys: set[str] = set()
    for display_name, cache_key in eval_targets:
        if cache_key in seen_keys:
            continue
        seen_keys.add(cache_key)
        try:
            payload = load_eval_cache(cache_key, layer)
        except FileNotFoundError:
            print(f"[eval] Warning: cache missing for {cache_key} layer {layer}; skipping.")
            continue
        counts = payload.get("counts")
        dialogue_labels = payload.get("dialogue_labels")
        acts = payload["activations"]
        if counts is None or dialogue_labels is None:
            print(f"[eval] Warning: {cache_key} cache missing counts/dialogue labels; regenerate.")
            continue
        pooled, pooled_labels, skipped = pool_dialogue_activations(acts, counts, dialogue_labels)
        if pooled.numel() == 0 or len(np.unique(pooled_labels.numpy())) < 2:
            print(f"[eval] Warning: {cache_key} layer {layer} lacks sufficient class variety; skipping.")
            continue
        scores = compute_scores(pooled.to(torch.float32), probe)
        labels = pooled_labels.numpy()

        category_labels = payload.get("category_labels")
        filtered_categories: list[Sequence[str]] | None = None
        if category_labels is not None:
            filtered_categories = []
            idx = 0
            for count in counts:
                cats = category_labels[idx] if idx < len(category_labels) else []
                idx += 1
                if count is None or count <= 0:
                    continue
                filtered_categories.append(cats)
            if len(filtered_categories) != len(labels):
                filtered_categories = filtered_categories[: len(labels)]

        roc = roc_curve(labels, scores)
        auroc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")
        violin_groups, violin_order = _default_violin_groups(scores, labels)
        if payload.get("meta", {}).get("source") == "beaver_tails" and filtered_categories is not None:
            beaver_groups, beaver_order = _beaver_violin_groups(scores, labels, filtered_categories)
            if beaver_groups:
                violin_groups, violin_order = beaver_groups, beaver_order

        plot_inputs[display_name] = {
            "scores": scores,
            "labels": labels,
            "roc": {"fpr": roc[0], "tpr": roc[1]},
            "auroc": float(auroc),
            "skipped_dialogues": skipped,
            "violin_groups": violin_groups,
            "violin_order": violin_order,
        }
        metrics_summary[display_name] = {"auroc": float(auroc)}
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
        results_dir = results_layer_dir(dataset_slug, layer) / tag
    else:
        results_dir = results_layer_dir(dataset_slug, layer)
    results_dir.mkdir(parents=True, exist_ok=True)
    roc_path = results_dir / "eval_roc.png"
    violin_path = results_dir / "eval_violin.png"
    plot_roc(layer, dataset_slug, plot_inputs, roc_path)
    plot_violin(layer, dataset_slug, plot_inputs, violin_path)

    metrics_path = results_dir / "eval_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "train_dataset": dataset_slug,
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
    parser.add_argument("--dataset", nargs="+", default=[DEFAULT_DATASET], help="Training dataset name(s).")
    parser.add_argument("--train-dataset", default=None, help="Explicit dataset slug (legacy).")
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="Specific layers to evaluate.")
    parser.add_argument("--eval-datasets", choices=EVAL_DATASETS, nargs="+", default=EVAL_DATASETS)
    parser.add_argument(
        "--apollo-eval",
        nargs="+",
        default=DEFAULT_APOLLO_EVALS,
        help="Apollo training-style dataset slugs to evaluate against (e.g., 'got_cities__plain').",
    )
    parser.add_argument(
        "--beaver-eval",
        dest="include_beaver",
        action="store_true",
        help="Include BeaverTails evaluation data (default).",
    )
    parser.add_argument(
        "--no-beaver-eval",
        dest="include_beaver",
        action="store_false",
        help="Disable BeaverTails evaluation.",
    )
    parser.add_argument(
        "--beaver-split",
        default=BEAVER_DEFAULT_SPLIT,
        help="BeaverTails split to evaluate (default: 30k_test).",
    )
    parser.add_argument(
        "--beaver-sample-size",
        type=int,
        default=None,
        help="Optional number of BeaverTails rows to sample.",
    )
    parser.add_argument(
        "--beaver-seed",
        type=int,
        default=42,
        help="Seed used when caching BeaverTails eval data.",
    )
    parser.set_defaults(include_beaver=True)
    parser.add_argument("--probe-path", type=Path, default=None, help="External probe path (optional).")
    parser.add_argument("--probe-layer", type=int, default=None, help="Layer index inside external probe (defaults to eval layer).")
    parser.add_argument("--probe-name", type=str, default=None, help="Custom name for external probe outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_dataset:
        dataset_slug = args.train_dataset
        dataset_names = args.train_dataset.split("+")
    else:
        dataset_names, dataset_slug = prepare_datasets(args.dataset)
    print(f"[eval] Training dataset slug: {dataset_slug} ({', '.join(dataset_names)})")

    apollo_eval_slugs: list[str] = []
    seen_apollo: set[str] = set()
    for entry in args.apollo_eval or []:
        _, slug = prepare_datasets([entry])
        if slug in seen_apollo:
            continue
        seen_apollo.add(slug)
        apollo_eval_slugs.append(slug)

    eval_targets: list[tuple[str, str]] = []
    for config in args.eval_datasets:
        alias = slugify_config(config)
        eval_targets.append((config, alias))
    for slug in apollo_eval_slugs:
        eval_targets.append((slug, slug))

    if args.include_beaver:
        beaver_dataset_slug = beaver_slug(args.beaver_split, args.beaver_sample_size, args.beaver_seed)
        beaver_name = f"BeaverTails ({args.beaver_split}"
        if args.beaver_sample_size:
            beaver_name += f", n={args.beaver_sample_size}"
        beaver_name += ")"
        eval_targets.append((beaver_name, beaver_dataset_slug))

    layers = args.layers or list_trained_layers(dataset_slug)
    if not layers:
        raise SystemExit(f"No trained probes found for dataset {dataset_slug}.")
    for layer in layers:
        evaluate_layer(
            dataset_slug,
            layer,
            eval_targets,
            external_probe=args.probe_path,
            external_layer=args.probe_layer,
            probe_name=args.probe_name,
        )


if __name__ == "__main__":
    main()
