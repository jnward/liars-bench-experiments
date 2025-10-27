from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from .config import DEFAULT_DATASET, REG_COEFF
from .data import prepare_datasets
from .paths import layer_cache_path, probe_layer_dir, results_layer_dir


def load_single_cache(dataset_name: str, layer: int) -> dict:
    path = layer_cache_path(dataset_name, layer)
    if not path.exists():
        raise FileNotFoundError(f"Missing cached activations for dataset={dataset_name}, layer={layer}: {path}")
    return torch.load(path, map_location="cpu")


def merge_split_sections(sections: Sequence[dict]) -> dict:
    activations_list = []
    labels_list = []
    counts: list[int] = []
    dialogue_labels_list = []
    hidden_dim = None

    for section in sections:
        acts = section["activations"]
        if hidden_dim is None and acts.numel() > 0:
            hidden_dim = acts.size(1)
        if acts.numel() > 0:
            activations_list.append(acts.to(torch.float16))
        labels = section["labels"]
        if labels.numel() > 0:
            labels_list.append(labels.to(torch.long))
        section_counts = section.get("counts", [])
        counts.extend(int(c) for c in section_counts)
        dialogue_labels = section.get("dialogue_labels")
        if dialogue_labels is not None and dialogue_labels.numel() > 0:
            dialogue_labels_list.append(dialogue_labels.to(torch.long))

    if hidden_dim is None:
        hidden_dim = next(
            (section["activations"].size(1) for section in sections if section["activations"].dim() == 2),
            0,
        )

    if activations_list:
        activations = torch.cat(activations_list, dim=0)
    else:
        activations = torch.empty((0, hidden_dim), dtype=torch.float16)

    if labels_list:
        labels = torch.cat(labels_list, dim=0)
    else:
        labels = torch.empty(0, dtype=torch.long)

    if dialogue_labels_list:
        dialogue_labels = torch.cat(dialogue_labels_list, dim=0)
    else:
        if counts:
            raise ValueError("Missing dialogue_labels data while counts are present; regenerate caches.")
        dialogue_labels = torch.empty(0, dtype=torch.long)

    return {
        "activations": activations,
        "labels": labels,
        "counts": counts,
        "dialogue_labels": dialogue_labels,
    }


def merge_caches(dataset_names: Sequence[str], caches: Sequence[dict], dataset_slug: str, layer: int) -> dict:
    train_sections = [cache["train"] for cache in caches]
    val_sections = [cache["val"] for cache in caches]
    merged_train = merge_split_sections(train_sections)
    merged_val = merge_split_sections(val_sections)
    meta = caches[0].get("meta", {})

    combined_meta = {
        "model_name": meta.get("model_name"),
        "val_fraction": meta.get("val_fraction"),
        "train_dialogues": len(merged_train["counts"]),
        "val_dialogues": len(merged_val["counts"]),
        "train_tokens": int(merged_train["labels"].numel()),
        "val_tokens": int(merged_val["labels"].numel()),
        "seed": meta.get("seed"),
        "source_datasets": list(dataset_names),
    }

    return {
        "dataset": dataset_slug,
        "layer_index": layer,
        "train": merged_train,
        "val": merged_val,
        "meta": combined_meta,
    }


def load_cache(dataset_names: Sequence[str], dataset_slug: str, layer: int) -> dict:
    combo_path = layer_cache_path(dataset_slug, layer)
    if combo_path.exists():
        return torch.load(combo_path, map_location="cpu")
    if len(dataset_names) == 1:
        return load_single_cache(dataset_names[0], layer)

    caches = [load_single_cache(name, layer) for name in dataset_names]
    print(f"[train] Merging cached activations for datasets: {', '.join(dataset_names)}")
    return merge_caches(dataset_names, caches, dataset_slug, layer)


def normalize_features(acts: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    if acts.numel() == 0:
        return np.empty((0, mean.numel()), dtype=np.float32)
    return ((acts.to(torch.float32) - mean) / std).numpy()


def train_probe(
    dataset_names: Sequence[str],
    dataset_slug: str,
    layer: int,
    reg_coeff: float = REG_COEFF,
) -> tuple[Path, Path]:
    cache = load_cache(dataset_names, dataset_slug, layer)
    train = cache["train"]
    val = cache["val"]
    stats = cache.get("train_stats") or {}
    mean = stats.get("mean")
    std = stats.get("std")

    if mean is None or std is None:
        train_float = train["activations"].to(torch.float32)
        mean = train_float.mean(dim=0)
        std = train_float.std(dim=0).clamp_min(1e-8)
    else:
        mean = mean.to(torch.float32)
        std = std.to(torch.float32)

    if train["activations"].numel() == 0:
        raise ValueError("Training activations are empty; cannot fit probe.")

    train_X = normalize_features(train["activations"], mean, std)
    train_y = train["labels"].numpy()
    if np.unique(train_y).size < 2:
        raise ValueError("Training labels contain a single class; cannot fit logistic regression.")

    clf = LogisticRegression(
        C=1.0 / reg_coeff,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(train_X, train_y)

    train_probs = clf.predict_proba(train_X)[:, 1]
    train_metrics = {
        "accuracy": float(accuracy_score(train_y, (train_probs > 0.5).astype(int))),
        "auroc": float(roc_auc_score(train_y, train_probs)) if len(np.unique(train_y)) > 1 else float("nan"),
        "tokens": int(len(train_y)),
    }

    val_metrics = None
    if val["activations"].numel() > 0:
        val_X = normalize_features(val["activations"], mean, std)
        val_y = val["labels"].numpy()
        val_probs = clf.predict_proba(val_X)[:, 1]
        val_metrics = {
            "accuracy": float(accuracy_score(val_y, (val_probs > 0.5).astype(int))),
            "auroc": float(roc_auc_score(val_y, val_probs)) if len(np.unique(val_y)) > 1 else float("nan"),
            "tokens": int(len(val_y)),
        }

    probe_dir = probe_layer_dir(dataset_slug, layer)
    probe_path = probe_dir / "probe.pkl"

    direction = torch.tensor(clf.coef_, dtype=torch.float32)
    intercept = torch.tensor(clf.intercept_, dtype=torch.float32)

    torch.save(
        {
            "layers": [layer],
            "directions": direction,
            "normalize": True,
            "scaler_mean": mean,
            "scaler_scale": std,
            "logistic_intercept": intercept,
            "reg_coeff": reg_coeff,
            "dataset": dataset_slug,
        },
        probe_path,
    )

    results_dir = results_layer_dir(dataset_slug, layer)
    metrics_path = results_dir / "metrics.json"
    metrics = {
        "dataset": dataset_slug,
        "layer_index": layer,
        "train": train_metrics,
        "val": val_metrics,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved probe -> {probe_path}")
    print(f"Saved metrics -> {metrics_path}")
    return probe_path, metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train logistic probes on cached activations.")
    parser.add_argument("--dataset", nargs="+", default=[DEFAULT_DATASET], help="Training dataset name(s).")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to train.")
    parser.add_argument("--reg-coeff", type=float, default=REG_COEFF, help="Regularization coefficient.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing probe artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names, dataset_slug = prepare_datasets(args.dataset)
    print(f"[train] Dataset slug: {dataset_slug} ({', '.join(dataset_names)}) | Layer {args.layer}")
    probe_dir = probe_layer_dir(dataset_slug, args.layer)
    probe_path = probe_dir / "probe.pkl"
    if probe_path.exists() and not args.force:
        print(f"Probe already exists for dataset={dataset_slug}, layer={args.layer}; skipping.")
        return
    train_probe(dataset_names, dataset_slug, args.layer, reg_coeff=args.reg_coeff)


if __name__ == "__main__":
    main()
