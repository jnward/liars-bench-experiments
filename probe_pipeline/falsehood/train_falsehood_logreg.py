# %% [markdown]
# Train logistic regression probes for falsehood detection across dataset combinations.

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from tqdm import tqdm


def slugify(name: str) -> str:
    return name.replace("/", "_")


# Configuration
CACHE_DIR = Path("probe_pipeline/falsehood/cache")
DATASET_KEYS: List[str] = [
    "convincing-game",
    "harm-pressure-choice",
    "harm-pressure-knowledge-report",
    "instructed-deception",
    "insider-trading/report",
    "insider-trading/confirmation",
]
DEFAULT_LAYER_INDEX = 22
LAYER_INDEX = int(os.environ.get("LAYER_INDEX", DEFAULT_LAYER_INDEX))
OUTPUT_DIR = Path(f"probe_pipeline/falsehood/probes/layer{LAYER_INDEX}")
RESULTS_DIR = Path(f"probe_pipeline/falsehood/results/layer{LAYER_INDEX}")
PLOT_DIR = Path(f"probe_pipeline/falsehood/plots/layer{LAYER_INDEX}")
RANDOM_SEED = 42
LOGREG_C = 1.0
MAX_ITER = 1000


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def cache_path_for(dataset_key: str) -> Path:
    return CACHE_DIR / f"{slugify(dataset_key)}_layer{LAYER_INDEX}.pt"


def load_cache(dataset_key: str) -> dict:
    path = cache_path_for(dataset_key)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found for {dataset_key}: {path}")
    return torch.load(path, map_location="cpu")


def get_split(cache: dict, split: str) -> Tuple[np.ndarray, np.ndarray]:
    indices = cache["meta"].get(f"{split}_indices")
    if indices is None:
        raise KeyError(f"Split '{split}' unavailable in cache meta.")
    idx_tensor = torch.tensor(indices, dtype=torch.long)
    activations = cache["activations"].index_select(0, idx_tensor).detach().cpu().numpy()
    labels = cache["labels"].index_select(0, idx_tensor).detach().cpu().numpy()
    return activations, labels


def train_probe(train_datasets: Sequence[str]) -> LogisticRegression:
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    for dataset in train_datasets:
        cache = load_cache(dataset)
        X_split, y_split = get_split(cache, "train")
        X_parts.append(X_split)
        y_parts.append(y_split)

    if not X_parts:
        raise RuntimeError("No training data collected; check cache availability.")

    X_train = np.concatenate(X_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)

    if np.unique(y_train).size < 2:
        raise ValueError("Training data contains fewer than two classes; cannot fit probe.")

    clf = LogisticRegression(
        C=LOGREG_C,
        max_iter=MAX_ITER,
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_probe(
    clf: LogisticRegression,
    probe_name: str,
    train_datasets: Sequence[str],
    all_datasets: Sequence[str],
) -> dict:
    metrics: Dict[str, dict] = {
        "probe": probe_name,
        "trained_on": list(train_datasets),
        "datasets": {},
        "layer_index": LAYER_INDEX,
        "logreg_C": LOGREG_C,
        "max_iter": MAX_ITER,
        "seed": RANDOM_SEED,
    }

    plt.figure(figsize=(7, 6))
    any_curve = False

    if len(train_datasets) == len(all_datasets):
        highlight_mode = "all"
    elif len(train_datasets) == 1:
        highlight_mode = "single"
    else:
        highlight_mode = "leaveout"

    for dataset in all_datasets:
        try:
            cache = load_cache(dataset)
        except FileNotFoundError:
            print(f"Skipping ROC for {dataset}: cache missing.")
            continue

        try:
            X_eval, y_eval = get_split(cache, "eval")
        except KeyError:
            print(f"Skipping ROC for {dataset}: eval split missing.")
            continue

        probs = clf.predict_proba(X_eval)[:, 1]
        preds = (probs > 0.5).astype(int)

        acc = accuracy_score(y_eval, preds)
        try:
            auroc = roc_auc_score(y_eval, probs)
        except ValueError:
            auroc = float("nan")

        dataset_entry: Dict[str, object] = {
            "eval_samples": int(len(y_eval)),
            "accuracy": float(acc),
            "auroc": float(auroc),
        }

        if np.unique(y_eval).size >= 2:
            fpr, tpr, _ = roc_curve(y_eval, probs)
            dataset_entry["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
            }

            in_training = dataset in train_datasets
            if highlight_mode == "single":
                highlight = dataset == train_datasets[0]
                label_tag = "train" if highlight else "eval"
                line_width = 2.5 if highlight else 1.0
                alpha = 1.0 if highlight else 0.5
            elif highlight_mode == "leaveout":
                highlight = not in_training
                label_tag = "held-out" if highlight else "train"
                line_width = 2.5 if highlight else 1.2
                alpha = 1.0 if highlight else 0.7
            else:
                label_tag = "train"
                line_width = 1.5
                alpha = 0.9

            plt.plot(
                fpr,
                tpr,
                label=f"{dataset} ({label_tag}, AUROC={auroc:.3f})",
                linewidth=line_width,
                alpha=alpha,
            )
            any_curve = True
        else:
            dataset_entry["roc_curve"] = None
            print(f"Skipping ROC curve plot for {dataset}: only one class in eval labels.")

        metrics["datasets"][dataset] = dataset_entry

    if any_curve:
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Falsehood Probe ROC Curves: {probe_name}")
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.4)
    else:
        plt.text(0.5, 0.5, "Insufficient data for ROC curves", ha="center", va="center")
        plt.axis("off")

    plot_path = PLOT_DIR / f"{probe_name}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    metrics["plot_path"] = str(plot_path)

    result_path = RESULTS_DIR / f"{probe_name}.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    metrics["result_path"] = str(result_path)
    return metrics


def build_training_sets(dataset_keys: Sequence[str]) -> List[tuple[str, List[str]]]:
    combos: List[tuple[str, List[str]]] = []

    # Single-dataset probes
    for key in dataset_keys:
        combos.append((f"single_{slugify(key)}", [key]))

    # Leave-one-out combinations
    for key in dataset_keys:
        remaining = [k for k in dataset_keys if k != key]
        combos.append((f"leaveout_{slugify(key)}", remaining))

    # Skyline (all datasets)
    combos.append(("all_datasets", list(dataset_keys)))

    return combos


def compute_cosine_similarity(probe_vectors: List[tuple[str, np.ndarray]]) -> None:
    if not probe_vectors:
        return

    names = [name for name, _ in probe_vectors]
    matrix = np.zeros((len(probe_vectors), len(probe_vectors)), dtype=np.float32)
    for i, (_, vec_i) in enumerate(probe_vectors):
        for j, (_, vec_j) in enumerate(probe_vectors):
            matrix[i, j] = float(np.dot(vec_i, vec_j))

    plt.figure(figsize=(8, 7))
    im = plt.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Cosine similarity")
    plt.xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.title("Falsehood Probe Direction Cosine Similarity")
    plt.tight_layout()
    heatmap_path = PLOT_DIR / "probe_cosine_similarity.png"
    plt.savefig(heatmap_path)
    plt.close()

    with (RESULTS_DIR / "probe_cosine_similarity.json").open("w", encoding="utf-8") as f:
        json.dump({"probes": names, "cosine_matrix": matrix.tolist()}, f, indent=2)


def main() -> None:
    dataset_keys = list(DATASET_KEYS)
    combos = build_training_sets(dataset_keys)

    probe_vectors: List[tuple[str, np.ndarray]] = []
    summary: Dict[str, dict] = {}

    for probe_name, train_list in tqdm(combos, desc="Training probes"):
        try:
            clf = train_probe(train_list)
        except FileNotFoundError as exc:
            print(f"Skipping {probe_name}: {exc}")
            summary[probe_name] = {"status": "missing-cache", "reason": str(exc)}
            continue
        except ValueError as exc:
            print(f"Skipping {probe_name}: {exc}")
            summary[probe_name] = {"status": "skipped", "reason": str(exc)}
            continue

        metrics = evaluate_probe(clf, probe_name, train_list, dataset_keys)

        out_path = OUTPUT_DIR / f"{probe_name}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(
                {
                    "model": clf,
                    "trained_on": list(train_list),
                    "datasets": dataset_keys,
                    "config": {
                        "C": LOGREG_C,
                        "max_iter": MAX_ITER,
                        "seed": RANDOM_SEED,
                    },
                },
                f,
            )
        print(f"Saved probe '{probe_name}' -> {out_path}")

        vec = clf.coef_.astype(np.float32).squeeze(0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        probe_vectors.append((probe_name, vec))

        metrics.update({"status": "ok", "probe_path": str(out_path)})
        summary[probe_name] = metrics

    compute_cosine_similarity(probe_vectors)

    summary["layer_index"] = LAYER_INDEX
    summary_path = RESULTS_DIR / "falsehood_probe_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved training summary ->", summary_path)


if __name__ == "__main__":
    main()
