# %% [markdown]
# Train logistic regression probes from cached activations.

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


# Configuration
CACHE_DIR = Path("probe_pipeline/cache")
CACHE_FILES = [
    "convincing-game.pt",
    "harm-pressure-choice.pt",
    "harm-pressure-knowledge-report.pt",
    # "insider-trading_report.pt",
    # "insider-trading_confirmation.pt",
    # "instructed-deception.pt",
]
OUTPUT_DIR = Path("probe_pipeline/probes")
RANDOM_SEED = 42
LOGREG_C = 1.0
MAX_ITER = 1000


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cache(file_name: str) -> dict:
    data = torch.load(CACHE_DIR / file_name, map_location="cpu")
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

    eval_probs = clf.predict_proba(X_eval)[:, 1]
    eval_preds = (eval_probs > 0.5).astype(int)

    acc = accuracy_score(y_eval, eval_preds)
    auroc = roc_auc_score(y_eval, eval_probs)

    print(f"Eval accuracy: {acc:.4f}")
    print(f"Eval AUROC: {auroc:.4f}")

    out_path = OUTPUT_DIR / "combined_logreg.pkl"
    with out_path.open("wb") as f:
        pickle.dump(
            {
                "model": clf,
                "cache_files": CACHE_FILES,
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


if __name__ == "__main__":
    train_logreg()

