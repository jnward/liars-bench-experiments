# %% [markdown]
# Train logistic regression probes from cached activations.

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def slugify(name: str) -> str:
    return name.replace("/", "_")


# Configuration
CACHE_DIR = Path("probe_pipeline/cache-qwen-org")
DATASET_KEYS: List[str] = [
    "harm-pressure-choice",
    "harm-pressure-knowledge-report",
    "instructed-deception",
]
DEFAULT_LAYER_INDEX = 18
LAYER_INDEX = int(os.environ.get("LAYER_INDEX", DEFAULT_LAYER_INDEX))
CACHE_FILES = [f"{slugify(key)}_layer{LAYER_INDEX}.pt" for key in DATASET_KEYS]
OUTPUT_DIR = Path(f"probe_pipeline/probes-qwen-org/layer{LAYER_INDEX}")
RESULTS_DIR = Path(f"probe_pipeline/results-qwen-org/layer{LAYER_INDEX}")
RANDOM_SEED = 42
LOGREG_C = 1.0
MAX_ITER = 1000


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_cache(file_name: str) -> dict:
    path = CACHE_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing cache file: {path}")
    data = torch.load(path, map_location="cpu")
    return data


def train_logreg() -> None:
    act_list = []
    label_list = []

    eval_acts = []
    eval_labels = []

    metadata = []

    for fname in CACHE_FILES:
        cache = load_cache(fname)
        activations = cache["activations"]
        labels = cache["labels"]
        meta = cache["meta"]
        train_idx = torch.tensor(meta["train_indices"])
        eval_idx = torch.tensor(meta["eval_indices"])

        act_list.append(activations[train_idx])
        label_list.append(labels[train_idx])

        eval_acts.append(activations[eval_idx])
        eval_labels.append(labels[eval_idx])

        metadata.append(meta)

    X_train = torch.cat(act_list, dim=0).numpy()
    y_train = torch.cat(label_list, dim=0).numpy()
    X_eval = torch.cat(eval_acts, dim=0).numpy()
    y_eval = torch.cat(eval_labels, dim=0).numpy()

    clf = LogisticRegression(
        C=LOGREG_C,
        max_iter=MAX_ITER,
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)

    # Compute train metrics
    train_probs = clf.predict_proba(X_train)[:, 1]
    train_preds = (train_probs > 0.5).astype(int)
    train_acc = accuracy_score(y_train, train_preds)
    train_auroc = roc_auc_score(y_train, train_probs)

    # Compute eval metrics
    eval_probs = clf.predict_proba(X_eval)[:, 1]
    eval_preds = (eval_probs > 0.5).astype(int)
    eval_acc = accuracy_score(y_eval, eval_preds)
    eval_auroc = roc_auc_score(y_eval, eval_probs)

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Train AUROC: {train_auroc:.4f}")
    print(f"Eval accuracy: {eval_acc:.4f}")
    print(f"Eval AUROC: {eval_auroc:.4f}")

    out_path = OUTPUT_DIR / "combined_logreg.pkl"
    with out_path.open("wb") as f:
        pickle.dump(
            {
                "model": clf,
                "cache_files": CACHE_FILES,
                "datasets": DATASET_KEYS,
                "metadata": metadata,
                "config": {
                    "C": LOGREG_C,
                    "max_iter": MAX_ITER,
                    "seed": RANDOM_SEED,
                },
            },
            f,
        )
    print(f"Saved probe to {out_path}")

    metrics_path = RESULTS_DIR / "combined_logreg_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "train_accuracy": float(train_acc),
                "train_auroc": float(train_auroc),
                "datasets": DATASET_KEYS,
                "layer_index": LAYER_INDEX,
                "eval_accuracy": float(eval_acc),
                "eval_auroc": float(eval_auroc),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    train_logreg()
