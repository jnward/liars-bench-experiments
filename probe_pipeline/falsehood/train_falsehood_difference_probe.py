# %% [markdown]
# Train logistic regression probes on difference vectors between
# "Yes"/"No" user judgments for falsehood detection datasets.

from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
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
PROBE_DIR = Path(f"probe_pipeline/falsehood/diff_probes/layer{LAYER_INDEX}")
RESULTS_DIR = Path(f"probe_pipeline/falsehood/diff_results/layer{LAYER_INDEX}")
PLOT_DIR = Path(f"probe_pipeline/falsehood/diff_plots/layer{LAYER_INDEX}")
RANDOM_SEED = 42
LOGREG_C = 1.0
MAX_ITER = 1000


PROBE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

YES_NO_GUESS_A = "yes_no_guess_a"
YES_NO_GUESS_B = "yes_no_guess_b"


class DatasetDiffSplit:
    """Container for difference vectors and labels per dataset."""

    def __init__(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        eval_X: np.ndarray,
        eval_y: np.ndarray,
        metadata: dict,
    ) -> None:
        self.train_X = train_X
        self.train_y = train_y
        self.eval_X = eval_X
        self.eval_y = eval_y
        self.metadata = metadata

    @property
    def train_size(self) -> int:
        return int(self.train_X.shape[0]) if self.train_X.size else 0

    @property
    def eval_size(self) -> int:
        return int(self.eval_X.shape[0]) if self.eval_X.size else 0


def cache_path_for(dataset_key: str) -> Path:
    return CACHE_DIR / f"{slugify(dataset_key)}_layer{LAYER_INDEX}.pt"


def load_cache(dataset_key: str) -> dict:
    path = cache_path_for(dataset_key)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found for {dataset_key}: {path}")
    return torch.load(path, map_location="cpu")


def build_difference_split(cache: dict) -> DatasetDiffSplit:
    activations: torch.Tensor = cache["activations"]
    labels: torch.Tensor = cache["labels"]
    meta: dict = cache["meta"]

    tags: List[str] = meta["variant_tags"]
    group_ids: List[int] = meta["group_ids"]
    train_indices = set(meta["train_indices"])
    eval_indices = set(meta["eval_indices"])

    group_to_indices: Dict[int, Dict[str, int]] = defaultdict(dict)
    for idx, gid in enumerate(group_ids):
        tag = tags[idx]
        if tag == YES_NO_GUESS_A:
            group_to_indices[gid]["A"] = idx
        elif tag == YES_NO_GUESS_B:
            group_to_indices[gid]["B"] = idx

    train_vectors: List[np.ndarray] = []
    train_labels: List[int] = []
    eval_vectors: List[np.ndarray] = []
    eval_labels: List[int] = []

    for gid, pair in group_to_indices.items():
        idx_a = pair.get("A")
        idx_b = pair.get("B")
        if idx_a is None or idx_b is None:
            continue

        vec_a = activations[idx_a]
        vec_b = activations[idx_b]
        diff_vec = (vec_a - vec_b).detach().cpu().numpy()

        deceptive_label = int(labels[idx_a].item())

        if idx_a in train_indices:
            train_vectors.append(diff_vec)
            train_labels.append(deceptive_label)
        elif idx_a in eval_indices:
            eval_vectors.append(diff_vec)
            eval_labels.append(deceptive_label)
        else:
            # Fallback: if indices were not assigned (shouldn't happen), skip.
            continue

    def stack_or_empty(vectors: List[np.ndarray]) -> np.ndarray:
        if not vectors:
            return np.zeros((0, activations.size(-1)), dtype=np.float32)
        return np.stack(vectors, axis=0).astype(np.float32)

    train_X = stack_or_empty(train_vectors)
    eval_X = stack_or_empty(eval_vectors)
    train_y_arr = np.array(train_labels, dtype=np.int64)
    eval_y_arr = np.array(eval_labels, dtype=np.int64)

    return DatasetDiffSplit(train_X, train_y_arr, eval_X, eval_y_arr, meta)


def assemble_split(
    datasets: Sequence[str],
    dataset_cache: Dict[str, DatasetDiffSplit],
    split: str,
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for dataset in datasets:
        data = dataset_cache.get(dataset)
        if data is None:
            continue
        if split == "train" and data.train_size:
            xs.append(data.train_X)
            ys.append(data.train_y)
        elif split == "eval" and data.eval_size:
            xs.append(data.eval_X)
            ys.append(data.eval_y)
    if not xs:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


def train_probe(train_datasets: Sequence[str], dataset_cache: Dict[str, DatasetDiffSplit]) -> LogisticRegression:
    X_train, y_train = assemble_split(train_datasets, dataset_cache, "train")
    if X_train.size == 0:
        raise ValueError(f"No training data available for datasets: {train_datasets}")
    if np.unique(y_train).size < 2:
        raise ValueError("Training labels contain fewer than two classes; cannot fit probe.")

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
    dataset_cache: Dict[str, DatasetDiffSplit],
    ordered_datasets: Sequence[str],
) -> dict:
    all_datasets = list(ordered_datasets)

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
        data = dataset_cache.get(dataset)
        if data is None or data.eval_size == 0:
            print(f"Skipping evaluation for {dataset}: no eval data.")
            continue

        X_eval = data.eval_X
        y_eval = data.eval_y

        probs = clf.predict_proba(X_eval)[:, 1]
        preds = (probs > 0.5).astype(int)

        acc = accuracy_score(y_eval, preds)
        try:
            auroc = roc_auc_score(y_eval, probs)
        except ValueError:
            auroc = float("nan")

        entry: Dict[str, object] = {
            "eval_samples": int(len(y_eval)),
            "accuracy": float(acc),
            "auroc": float(auroc),
        }

        if np.unique(y_eval).size >= 2:
            fpr, tpr, _ = roc_curve(y_eval, probs)
            entry["roc_curve"] = {
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
            entry["roc_curve"] = None
            print(f"Skipping ROC curve plot for {dataset}: only one class in eval labels.")

        metrics["datasets"][dataset] = entry

    if any_curve:
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Falsehood Difference Probe ROC Curves: {probe_name}")
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

    pred_summary = RESULTS_DIR / f"{probe_name}.json"
    with pred_summary.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    metrics["result_path"] = str(pred_summary)

    print(f"[{probe_name}] Saved probe metrics -> {pred_summary}")
    print(f"[{probe_name}] Saved ROC plot -> {plot_path}")

    return metrics


def build_training_sets(dataset_keys: Sequence[str]) -> List[tuple[str, List[str]]]:
    combos: List[tuple[str, List[str]]] = []
    for key in dataset_keys:
        combos.append((f"single_{slugify(key)}", [key]))
    for key in dataset_keys:
        remaining = [k for k in dataset_keys if k != key]
        combos.append((f"leaveout_{slugify(key)}", remaining))
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
    plt.title("Falsehood Difference Probe Cosine Similarity")
    plt.tight_layout()
    heatmap_path = PLOT_DIR / "difference_probe_cosine_similarity.png"
    plt.savefig(heatmap_path)
    plt.close()

    with (RESULTS_DIR / "difference_probe_cosine_similarity.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "probes": names,
                "cosine_matrix": matrix.tolist(),
            },
            f,
            indent=2,
        )


def main() -> None:
    dataset_cache: Dict[str, DatasetDiffSplit] = {}
    for dataset_key in tqdm(DATASET_KEYS, desc="Loading difference vectors"):
        try:
            cache = load_cache(dataset_key)
            dataset_cache[dataset_key] = build_difference_split(cache)
        except FileNotFoundError:
            print(f"Skipping {dataset_key}: cache file missing.")
        except Exception as exc:
            print(f"Skipping {dataset_key}: {exc}")

    if not dataset_cache:
        raise RuntimeError("No datasets available; ensure caches exist and include Yes/No variants.")

    ordered_datasets = [key for key in DATASET_KEYS if key in dataset_cache]
    combos = build_training_sets(ordered_datasets)
    probe_vectors: List[tuple[str, np.ndarray]] = []
    summary: Dict[str, dict] = {}

    for probe_name, train_list in tqdm(combos, desc="Training difference probes"):
        try:
            clf = train_probe(train_list, dataset_cache)
        except ValueError as exc:
            print(f"Skipping {probe_name}: {exc}")
            summary[probe_name] = {"status": "skipped", "reason": str(exc)}
            continue

        metrics = evaluate_probe(clf, probe_name, train_list, dataset_cache, ordered_datasets)

        out_path = PROBE_DIR / f"{probe_name}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(
                {
                    "model": clf,
                    "trained_on": list(train_list),
                    "datasets": list(dataset_cache.keys()),
                    "config": {
                        "C": LOGREG_C,
                        "max_iter": MAX_ITER,
                        "seed": RANDOM_SEED,
                    },
                },
                f,
            )
        print(f"[{probe_name}] Saved probe -> {out_path}")

        vec = clf.coef_.astype(np.float32).squeeze(0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        probe_vectors.append((probe_name, vec))

        metrics.update({"status": "ok", "probe_path": str(out_path)})
        summary[probe_name] = metrics

    compute_cosine_similarity(probe_vectors)

    summary["layer_index"] = LAYER_INDEX
    summary_path = RESULTS_DIR / "falsehood_difference_probe_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved difference probe summary ->", summary_path)


if __name__ == "__main__":
    main()
