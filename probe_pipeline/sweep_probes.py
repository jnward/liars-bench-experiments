# %% [markdown]
# Train and evaluate multiple logistic regression probes across dataset combinations.

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


# Configuration
CACHE_DIR = Path("probe_pipeline/cache")
DEFAULT_LAYER_INDEX = 22
LAYER_INDEX = int(os.environ.get("LAYER_INDEX", DEFAULT_LAYER_INDEX))
PROBE_DIR = Path(f"probe_pipeline/probes/layer{LAYER_INDEX}")
PLOT_DIR = Path(f"probe_pipeline/plots/layer{LAYER_INDEX}")
RESULTS_DIR = Path(f"probe_pipeline/results/layer{LAYER_INDEX}")
APOLLO_PROBE_PATH = Path("/workspace/jake/deception-detection/example_results/instructed_pairs/detector.pt")

DATASET_KEYS: List[str] = [
    "convincing-game",
    "harm-pressure-choice",
    "harm-pressure-knowledge-report",
    "instructed-deception",
    "insider-trading/report",
    "insider-trading/confirmation",
]

LOGREG_C = 1.0
MAX_ITER = 1000
RANDOM_SEED = 42


PROBE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def slugify(name: str) -> str:
    return name.replace("/", "_")


def load_cache(dataset_key: str) -> dict:
    path = CACHE_DIR / f"{slugify(dataset_key)}_layer{LAYER_INDEX}.pt"
    # path = CACHE_DIR / f"{slugify(dataset_key)}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found for {dataset_key}: {path}")
    return torch.load(path, map_location="cpu")


def get_split(cache: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    activations = cache["activations"]
    labels = cache["labels"]
    indices = cache["meta"][f"{split}_indices"]
    tensor_idx = torch.tensor(indices)
    X = activations[tensor_idx].numpy()
    y = labels[tensor_idx].numpy()
    return X, y


def train_probe(train_datasets: List[str]) -> LogisticRegression:
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    for dataset in train_datasets:
        cache = load_cache(dataset)
        X_split, y_split = get_split(cache, "train")
        X_parts.append(X_split)
        y_parts.append(y_split)

    X_train = np.concatenate(X_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)

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
    train_datasets: List[str],
    all_datasets: List[str],
) -> None:
    metrics = {
        "probe": probe_name,
        "trained_on": train_datasets,
        "datasets": {},
        "layer_index": LAYER_INDEX,
    }

    plt.figure(figsize=(7, 6))

    if len(train_datasets) == len(all_datasets):
        highlight_mode = "all"
    elif len(train_datasets) == 1:
        highlight_mode = "single"
    else:
        highlight_mode = "leaveout"

    for dataset in all_datasets:
        cache = load_cache(dataset)
        X_eval, y_eval = get_split(cache, "eval")
        if len(np.unique(y_eval)) < 2:
            print(f"Skipping ROC for {dataset}: only one class present.")
            continue

        probs = clf.predict_proba(X_eval)[:, 1]
        preds = (probs > 0.5).astype(int)

        acc = accuracy_score(y_eval, preds)
        auroc = roc_auc_score(y_eval, probs)
        fpr, tpr, _ = roc_curve(y_eval, probs)

        metrics["datasets"][dataset] = {
            "eval_samples": len(y_eval),
            "accuracy": float(acc),
            "auroc": float(auroc),
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
        else:  # all datasets
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

    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Probe ROC Curves: {probe_name}")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)

    plot_path = PLOT_DIR / f"{probe_name}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    with (RESULTS_DIR / f"{probe_name}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    probe_path = PROBE_DIR / f"{probe_name}.pkl"
    with probe_path.open("wb") as f:
        pickle.dump(
            {
                "model": clf,
                "trained_on": train_datasets,
                "config": {
                    "C": LOGREG_C,
                    "max_iter": MAX_ITER,
                    "seed": RANDOM_SEED,
                },
            },
            f,
        )

    print(f"Saved probe '{probe_name}' -> {probe_path}")
    print(f"Saved plot -> {plot_path}")


def evaluate_apollo_probe(dataset_keys: List[str]) -> None:
    with APOLLO_PROBE_PATH.open("rb") as f:
        apollo = pickle.load(f)

    layers = apollo["layers"]
    try:
        idx = list(layers).index(22)
    except ValueError:
        print("Apollo probe missing layer 22 entry; skipping.")
        return

    direction = apollo["directions"][idx]
    if isinstance(direction, torch.Tensor):
        direction = direction.detach().cpu().float()
    else:
        direction = torch.tensor(direction, dtype=torch.float32)
    direction = direction.view(-1)

    normalize = bool(apollo.get("normalize", False))
    scaler_mean = apollo.get("scaler_mean")
    scaler_scale = apollo.get("scaler_scale")

    if normalize:
        if isinstance(scaler_mean, torch.Tensor):
            scaler_mean = scaler_mean.detach().cpu().float()
        else:
            scaler_mean = torch.tensor(scaler_mean, dtype=torch.float32)
        if isinstance(scaler_scale, torch.Tensor):
            scaler_scale = scaler_scale.detach().cpu().float()
        else:
            scaler_scale = torch.tensor(scaler_scale, dtype=torch.float32)

    metrics = {}

    plt.figure(figsize=(7, 6))

    for dataset in dataset_keys:
        cache = load_cache(dataset)
        X_eval, y_eval = get_split(cache, "eval")
        if len(np.unique(y_eval)) < 2:
            print(f"Skipping Apollo ROC for {dataset}: single class.")
            continue

        acts = torch.from_numpy(X_eval).float()
        vec = direction
        if normalize:
            acts = (acts - scaler_mean) / scaler_scale
        scores = acts @ vec
        scores_np = scores.numpy()

        auroc = roc_auc_score(y_eval, scores_np)
        fpr, tpr, _ = roc_curve(y_eval, scores_np)

        metrics[dataset] = {
            "eval_samples": len(y_eval),
            "auroc": float(auroc),
        }

        plt.plot(fpr, tpr, label=f"{dataset} (AUROC={auroc:.3f})", linewidth=1.6)

    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Apollo Probe ROC Curves")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)

    plot_path = PLOT_DIR / "apollo_probe.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    with (RESULTS_DIR / "apollo_probe.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved Apollo probe plot -> {plot_path}")


def build_training_sets(dataset_keys: List[str]) -> List[tuple[str, List[str]]]:
    combos: List[tuple[str, List[str]]] = []

    # Individual datasets
    for key in dataset_keys:
        combos.append((f"single_{slugify(key)}", [key]))

    # Leave-one-out combinations
    for key in dataset_keys:
        remaining = [k for k in dataset_keys if k != key]
        combos.append((f"leaveout_{slugify(key)}", remaining))

    # All datasets combined
    combos.append(("all_datasets", list(dataset_keys)))

    return combos


def main() -> None:
    dataset_keys = list(DATASET_KEYS)
    combos = build_training_sets(dataset_keys)
    probe_vectors: List[tuple[str, np.ndarray]] = []

    for probe_name, train_list in combos:
        print(f"\n=== Training probe: {probe_name} ===")
        clf = train_probe(train_list)
        evaluate_probe(clf, probe_name, train_list, dataset_keys)
        vec = clf.coef_.astype(np.float32).squeeze(0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        probe_vectors.append((probe_name, vec))

    names = [name for name, _ in probe_vectors]
    # Add original Apollo probe for layer 22
    if LAYER_INDEX == 22 and APOLLO_PROBE_PATH.exists():
        with APOLLO_PROBE_PATH.open("rb") as f:
            apollo = pickle.load(f)
        directions = apollo["directions"]
        layers = apollo["layers"]
        idx = list(layers).index(22)
        vec = directions[idx].detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        names.append("apollo_probe")
        probe_vectors.append(("apollo_probe", vec))

    matrix = np.zeros((len(probe_vectors), len(probe_vectors)), dtype=np.float32)
    for i, (_, vec_i) in enumerate(probe_vectors):
        for j, (_, vec_j) in enumerate(probe_vectors):
            matrix[i, j] = float(np.dot(vec_i, vec_j))

    plt.figure(figsize=(8, 7))
    im = plt.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Cosine similarity")
    plt.xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.title("Probe Direction Cosine Similarity")
    plt.tight_layout()
    heatmap_path = PLOT_DIR / "probe_cosine_similarity.png"
    plt.savefig(heatmap_path)
    plt.close()

    with (RESULTS_DIR / "probe_cosine_similarity.json").open("w", encoding="utf-8") as f:
        json.dump({"probes": names, "cosine_matrix": matrix.tolist()}, f, indent=2)

    print(f"Saved cosine similarity heatmap -> {heatmap_path}")

    if LAYER_INDEX == 22 and APOLLO_PROBE_PATH.exists():
        evaluate_apollo_probe(dataset_keys)


if __name__ == "__main__":
    main()
