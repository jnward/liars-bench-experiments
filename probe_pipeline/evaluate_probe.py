# %% [markdown]
# Evaluate trained logistic regression probe on cached activations.

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List

import torch
from sklearn.metrics import accuracy_score, roc_auc_score


def slugify(name: str) -> str:
    return name.replace("/", "_")


DEFAULT_LAYER_INDEX = 18
LAYER_INDEX = int(os.environ.get("LAYER_INDEX", DEFAULT_LAYER_INDEX))
PROBE_NAME = os.environ.get("PROBE_NAME", "all_datasets")

PROBE_PATH = Path(f"probe_pipeline/probes-qwen-baseline/layer{LAYER_INDEX}/{PROBE_NAME}.pkl")
CACHE_DIR = Path("probe_pipeline/cache-qwen-baseline")
DATASET_KEYS: List[str] = [
    "harm-pressure-choice",
    "harm-pressure-knowledge-report",
    "instructed-deception",
]
CACHE_FILES = [f"{slugify(key)}_layer{LAYER_INDEX}.pt" for key in DATASET_KEYS]
SPLIT = os.environ.get("EVAL_SPLIT", "eval")  # or "train"


def load_probe(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    return data["model"], data


def evaluate() -> None:
    clf, info = load_probe(PROBE_PATH)

    for dataset_key, fname in zip(DATASET_KEYS, CACHE_FILES):
        cache_path = CACHE_DIR / fname
        if not cache_path.exists():
            print(f"Skipping {dataset_key}: cache file not found ({cache_path})")
            continue
        cache = torch.load(cache_path, map_location="cpu")
        activations = cache["activations"]
        y = cache["labels"]
        indices = cache["meta"][f"{SPLIT}_indices"]
        tensor_idx = torch.tensor(indices)
        X = activations[tensor_idx].numpy()
        y_true = y[tensor_idx].numpy()

        probs = clf.predict_proba(X)[:, 1]
        preds = (probs > 0.5).astype(int)

        acc = accuracy_score(y_true, preds)
        auroc = roc_auc_score(y_true, probs)

        print(f"\nProbe: {PROBE_NAME} | Dataset: {dataset_key}")
        print(f"  Split: {SPLIT}")
        print(f"  Samples: {len(y_true)}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  AUROC: {auroc:.4f}")


if __name__ == "__main__":
    evaluate()
