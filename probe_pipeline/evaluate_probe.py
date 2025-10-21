# %% [markdown]
# Evaluate trained logistic regression probe on cached activations.

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, roc_auc_score


PROBE_PATH = Path("probe_pipeline/probes/combined_logreg.pkl")
CACHE_DIR = Path("probe_pipeline/cache")
CACHE_FILES = [
    "convincing-game.pt",
    "harm-pressure-choice.pt",
    "harm-pressure-knowledge-report.pt",
    "insider-trading_report.pt",
    "insider-trading_confirmation.pt",
    "instructed-deception.pt",
]
SPLIT = "eval"  # or "train"


def load_probe(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    return data["model"], data


def evaluate() -> None:
    clf, info = load_probe(PROBE_PATH)

    for fname in CACHE_FILES:
        cache = torch.load(CACHE_DIR / fname, map_location="cpu")
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

        print(f"\nDataset: {fname}")
        print(f"  Split: {SPLIT}")
        print(f"  Samples: {len(y_true)}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  AUROC: {auroc:.4f}")


if __name__ == "__main__":
    evaluate()
