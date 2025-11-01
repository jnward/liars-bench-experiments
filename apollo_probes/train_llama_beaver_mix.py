from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from .config import REG_COEFF
from .data import canonicalize_dataset_names
from .llama_beaver_mix import BeaverIngestionConfig, DEFAULT_CORE_DATASETS, beaver_mix_slug
from .paths import layer_cache_path, probe_layer_dir, results_layer_dir


def _build_beaver_config(args: argparse.Namespace) -> BeaverIngestionConfig:
    categories = tuple(dict.fromkeys(args.beaver_category)) if args.beaver_category else None
    return BeaverIngestionConfig(
        split=args.beaver_split,
        categories=categories or BeaverIngestionConfig().categories,
        sample_size=args.beaver_sample_size,
    )


def _load_cache(slug: str, layer: int) -> dict:
    cache_path = layer_cache_path(slug, layer)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing cached activations for slug={slug}, layer={layer}. "
            "Run cache_llama_beaver_mix.py first."
        )
    return torch.load(cache_path, map_location="cpu")


def _normalize_features(acts: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    if acts.numel() == 0:
        return np.empty((0, mean.numel()), dtype=np.float32)
    return ((acts.to(torch.float32) - mean) / std).numpy()


def _compute_stats(cache: dict) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    train_section = cache["train"]
    stats = cache.get("train_stats") or {}
    mean = stats.get("mean")
    std = stats.get("std")
    if mean is None or std is None:
        activations: torch.Tensor = train_section["activations"].to(torch.float32)
        mean = activations.mean(dim=0)
        std = activations.std(dim=0).clamp_min(1e-8)
    else:
        mean = mean.to(torch.float32)
        std = std.to(torch.float32)
    train_y = train_section["labels"].numpy()
    train_X = _normalize_features(train_section["activations"], mean, std)
    return train_X, train_y, mean, std


def train_probe(
    slug: str,
    layer: int,
    reg_coeff: float,
) -> tuple[Path, Path]:
    cache = _load_cache(slug, layer)
    train_section = cache["train"]
    val_section = cache["val"]

    if train_section["activations"].numel() == 0:
        raise ValueError("Training activations are empty; cannot fit probe.")

    train_X, train_y, scaler_mean, scaler_std = _compute_stats(cache)
    if np.unique(train_y).size < 2:
        raise ValueError("Training labels contain a single class; cannot train logistic probe.")

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
    if val_section["activations"].numel() > 0:
        val_X = _normalize_features(val_section["activations"], scaler_mean, scaler_std)
        val_y = val_section["labels"].numpy()
        val_probs = clf.predict_proba(val_X)[:, 1]
        val_metrics = {
            "accuracy": float(accuracy_score(val_y, (val_probs > 0.5).astype(int))),
            "auroc": float(roc_auc_score(val_y, val_probs)) if len(np.unique(val_y)) > 1 else float("nan"),
            "tokens": int(len(val_y)),
        }

    probe_dir = probe_layer_dir(slug, layer)
    probe_path = probe_dir / "probe.pkl"
    direction = torch.tensor(clf.coef_, dtype=torch.float32)
    intercept = torch.tensor(clf.intercept_, dtype=torch.float32)

    torch.save(
        {
            "layers": [layer],
            "directions": direction,
            "normalize": True,
            "scaler_mean": scaler_mean,
            "scaler_scale": scaler_std,
            "logistic_intercept": intercept,
            "reg_coeff": reg_coeff,
            "dataset": slug,
        },
        probe_path,
    )

    results_dir = results_layer_dir(slug, layer)
    metrics_path = results_dir / "metrics.json"
    metrics = {
        "dataset": slug,
        "layer_index": layer,
        "train": train_metrics,
        "val": val_metrics,
        "meta": cache.get("meta", {}),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[llama-beaver-train] Saved probe -> {probe_path}")
    print(f"[llama-beaver-train] Saved metrics -> {metrics_path}")
    return probe_path, metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train probes on the Llama + Beaver negative dataset mix.")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to train.")
    parser.add_argument("--reg-coeff", type=float, default=REG_COEFF, help="Regularization coefficient.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing probe artifacts.")
    parser.add_argument("--core-datasets", nargs="+", default=None, help="Override core Apollo datasets.")
    parser.add_argument(
        "--beaver-split",
        default=BeaverIngestionConfig().split,
        help="BeaverTails split used during caching.",
    )
    parser.add_argument(
        "--beaver-category",
        action="append",
        help="BeaverTails categories used during caching (repeatable).",
    )
    parser.add_argument(
        "--beaver-sample-size",
        type=int,
        default=None,
        help="Sample size used during caching (must match cache generation).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    beaver_config = _build_beaver_config(args)
    core_datasets: Sequence[str] = (
        canonicalize_dataset_names(args.core_datasets)
        if args.core_datasets
        else list(DEFAULT_CORE_DATASETS)
    )
    slug = beaver_mix_slug(beaver_config, core_datasets)

    probe_dir = probe_layer_dir(slug, args.layer)
    probe_path = probe_dir / "probe.pkl"
    if probe_path.exists() and not args.force:
        print(f"[llama-beaver-train] Probe already exists for slug={slug}, layer={args.layer}; skipping.")
        return

    train_probe(slug, args.layer, reg_coeff=args.reg_coeff)


if __name__ == "__main__":
    main()
