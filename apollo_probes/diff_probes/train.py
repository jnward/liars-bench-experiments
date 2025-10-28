from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from .config import DEFAULT_LAYER_INDEX
from .paths import diff_cache_path, diff_probe_dir, diff_results_dir


FeatureVariant = str


@dataclass
class TrainSplit:
    features: np.ndarray
    labels: np.ndarray


def _load_cache(dataset_slug: str, layer: int) -> dict:
    path = diff_cache_path(dataset_slug, layer)
    if not path.exists():
        raise FileNotFoundError(f"Missing diff cache for dataset={dataset_slug}, layer={layer}: {path}")
    return torch.load(path, map_location="cpu")


def _select_feature(split: dict, variant: FeatureVariant) -> np.ndarray:
    key_map: Dict[str, str] = {
        "user": "user_diff",
        "assistant": "assistant_diff",
        "combined": "combined_diff",
    }
    if variant not in key_map:
        raise ValueError(f"Unknown feature variant '{variant}'. Expected one of {sorted(key_map)}.")
    tensor = split[key_map[variant]]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor for key '{key_map[variant]}', found {type(tensor)}")
    return tensor.to(torch.float32).numpy()


def _prepare_split(split: dict, variant: FeatureVariant) -> TrainSplit:
    features = _select_feature(split, variant)
    labels_tensor = split["labels"]
    if not isinstance(labels_tensor, torch.Tensor):
        raise TypeError("Split labels must be a tensor.")
    labels = labels_tensor.numpy().astype(np.int64)
    return TrainSplit(features=features, labels=labels)


def _fit_scaler(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64)
    std = features.std(axis=0, dtype=np.float64)
    std = np.clip(std, a_min=1e-8, a_max=None)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_scaler(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / std


def _compute_metrics(model: LogisticRegression, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    probs = model.predict_proba(features)[:, 1]
    preds = (probs > 0.5).astype(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
    }
    if len(np.unique(labels)) > 1:
        metrics["auroc"] = float(roc_auc_score(labels, probs))
    else:
        metrics["auroc"] = float("nan")
    metrics["samples"] = int(labels.shape[0])
    return metrics


def train_diff_probe(
    dataset_slug: str,
    variant: FeatureVariant = "user",
    layer: int = DEFAULT_LAYER_INDEX,
    reg_strength: float = 1.0,
    max_iter: int = 1000,
    seed: int = 42,
) -> dict:
    cache = _load_cache(dataset_slug, layer)

    train_split = _prepare_split(cache["train"], variant)
    val_split = _prepare_split(cache["val"], variant)

    if train_split.features.size == 0:
        raise ValueError("Training features are empty; regenerate diff cache.")
    if np.unique(train_split.labels).size < 2:
        raise ValueError("Training labels contain fewer than two classes; cannot fit logistic regression.")

    mean, std = _fit_scaler(train_split.features)
    train_X = _apply_scaler(train_split.features, mean, std)
    val_X = _apply_scaler(val_split.features, mean, std)

    clf = LogisticRegression(
        C=reg_strength,
        penalty="l2",
        solver="lbfgs",
        max_iter=max_iter,
        random_state=seed,
    )
    clf.fit(train_X, train_split.labels)

    train_metrics = _compute_metrics(clf, train_X, train_split.labels)
    val_metrics = _compute_metrics(clf, val_X, val_split.labels)

    probe_dir = diff_probe_dir(dataset_slug, layer)
    probe_path = probe_dir / f"{variant}_probe.pkl"
    torch.save(
        {
            "dataset": dataset_slug,
            "feature_variant": variant,
            "layer_index": layer,
            "mean": torch.from_numpy(mean),
            "std": torch.from_numpy(std),
            "coef": torch.from_numpy(clf.coef_.astype(np.float32)),
            "intercept": torch.from_numpy(clf.intercept_.astype(np.float32)),
            "reg_strength": reg_strength,
            "max_iter": max_iter,
            "seed": seed,
        },
        probe_path,
    )

    metrics = {
        "dataset": dataset_slug,
        "layer_index": layer,
        "feature_variant": variant,
        "train": train_metrics,
        "val": val_metrics,
        "probe_path": str(probe_path),
    }

    results_dir = diff_results_dir(dataset_slug, layer)
    results_path = results_dir / f"{variant}_train_metrics.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    metrics["results_path"] = str(results_path)
    return metrics
