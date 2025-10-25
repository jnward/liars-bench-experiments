from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from .config import DEFAULT_DATASET, REG_COEFF
from .paths import layer_cache_path, probe_layer_dir, results_layer_dir


def load_cache(dataset: str, layer: int) -> dict:
    path = layer_cache_path(dataset, layer)
    if not path.exists():
        raise FileNotFoundError(f"Missing cached activations for dataset={dataset}, layer={layer}: {path}")
    return torch.load(path, map_location="cpu")


def normalize_features(acts: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    if acts.numel() == 0:
        return np.empty((0, mean.numel()), dtype=np.float32)
    return ((acts.to(torch.float32) - mean) / std).numpy()


def train_probe(dataset: str, layer: int, reg_coeff: float = REG_COEFF) -> tuple[Path, Path]:
    cache = load_cache(dataset, layer)
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

    probe_dir = probe_layer_dir(dataset, layer)
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
            "dataset": dataset,
        },
        probe_path,
    )

    results_dir = results_layer_dir(dataset, layer)
    metrics_path = results_dir / "metrics.json"
    metrics = {
        "dataset": dataset,
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
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Training dataset name.")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to train.")
    parser.add_argument("--reg-coeff", type=float, default=REG_COEFF, help="Regularization coefficient.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing probe artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe_dir = probe_layer_dir(args.dataset, args.layer)
    probe_path = probe_dir / "probe.pkl"
    if probe_path.exists() and not args.force:
        print(f"Probe already exists for dataset={args.dataset}, layer={args.layer}; skipping.")
        return
    train_probe(args.dataset, args.layer, reg_coeff=args.reg_coeff)


if __name__ == "__main__":
    main()
