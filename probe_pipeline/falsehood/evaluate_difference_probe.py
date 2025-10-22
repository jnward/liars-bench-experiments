"""
Evaluate a pre-trained falsehood difference probe on cached Liars Bench activations.

Designed for probes that expect difference vectors between the model's
Yes/No follow-up answers (e.g., the cities_transplanted geometry probe).
Requires the standard caches produced by cache_falsehood_activations.py.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from tqdm.auto import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from falsehood.train_falsehood_difference_probe import (  # noqa: E402
    CACHE_DIR,
    DATASET_KEYS as DEFAULT_DATASET_KEYS,
    LAYER_INDEX,
    PLOT_DIR,
    RESULTS_DIR,
    YES_NO_GUESS_A,
    YES_NO_GUESS_B,
)


def slugify(name: str) -> str:
    return name.replace("/", "_")


def cache_path_for(dataset_key: str) -> Path:
    return CACHE_DIR / f"{slugify(dataset_key)}_layer{LAYER_INDEX}.pt"


def torch_load(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu")


class DatasetDiffSplit:
    def __init__(self, eval_X: np.ndarray, eval_y: np.ndarray, metadata: dict) -> None:
        self.eval_X = eval_X
        self.eval_y = eval_y
        self.metadata = metadata


def build_difference_split(cache: dict) -> DatasetDiffSplit:
    import torch

    activations: torch.Tensor = cache["activations"]
    labels: torch.Tensor = cache["labels"]
    meta: dict = cache["meta"]

    tags: List[str] = meta["variant_tags"]
    group_ids: List[int] = meta["group_ids"]
    eval_indices = set(meta["eval_indices"])

    group_to_indices: Dict[int, Dict[str, int]] = {}
    for idx, gid in enumerate(group_ids):
        tag = tags[idx]
        if tag == YES_NO_GUESS_A:
            group_to_indices.setdefault(gid, {})["A"] = idx
        elif tag == YES_NO_GUESS_B:
            group_to_indices.setdefault(gid, {})["B"] = idx

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

        if idx_a in eval_indices or idx_b in eval_indices:
            eval_vectors.append(diff_vec)
            eval_labels.append(deceptive_label)

    if not eval_vectors:
        return DatasetDiffSplit(
            eval_X=np.zeros((0, activations.size(-1)), dtype=np.float32),
            eval_y=np.zeros((0,), dtype=np.int64),
            metadata=meta,
        )

    eval_X = np.stack(eval_vectors, axis=0).astype(np.float32)
    eval_y = np.array(eval_labels, dtype=np.int64)
    return DatasetDiffSplit(eval_X=eval_X, eval_y=eval_y, metadata=meta)


def locate_probe(probe_name: str | None, probe_path_env: str | None) -> Path:
    if probe_path_env:
        path = Path(probe_path_env).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Probe not found at {path}")
        return path

    if not probe_name:
        raise ValueError("Set PROBE_NAME or PROBE_PATH to choose a probe to evaluate.")

    candidate_dirs = [
        Path(f"probe_pipeline/falsehood/probes/layer{LAYER_INDEX}"),
        Path(f"probe_pipeline/falsehood/diff_probes/layer{LAYER_INDEX}"),
    ]

    for directory in candidate_dirs:
        candidate = directory / (probe_name if probe_name.endswith(".pkl") else f"{probe_name}.pkl")
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Probe '{probe_name}' not found. Looked in: "
        + ", ".join(str(d) for d in candidate_dirs)
    )


def load_probe(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    return data["model"], data


def determine_datasets(env_value: str | None) -> Sequence[str]:
    if not env_value:
        return DEFAULT_DATASET_KEYS
    parts = [item.strip() for item in env_value.split(",")]
    datasets = [item for item in parts if item]
    if not datasets:
        raise ValueError("DATASETS environment variable was provided but no valid entries were found.")
    return datasets


def evaluate_probe_on_dataset(clf, X: np.ndarray, y: np.ndarray, dataset: str) -> dict:
    if X.size == 0 or y.size == 0:
        return {
            "samples": int(y.size),
            "accuracy": float("nan"),
            "auroc": float("nan"),
            "roc_curve": None,
        }

    probs = clf.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y, preds)
    try:
        auroc = roc_auc_score(y, probs)
        fpr, tpr, _ = roc_curve(y, probs)
        roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    except ValueError:
        auroc = float("nan")
        roc_data = None

    return {
        "samples": int(y.size),
        "accuracy": float(acc),
        "auroc": float(auroc),
        "roc_curve": roc_data,
    }


def main() -> None:
    probe_name = os.environ.get("PROBE_NAME")
    probe_path_env = os.environ.get("PROBE_PATH")
    dataset_env = os.environ.get("DATASETS")
    split_name = os.environ.get("EVAL_SPLIT", "eval")

    probe_path = locate_probe(probe_name, probe_path_env)
    clf, probe_payload = load_probe(probe_path)

    dataset_keys = determine_datasets(dataset_env)

    dataset_summaries = {}
    plt.figure(figsize=(7, 6))
    any_curve = False

    for dataset in tqdm(dataset_keys, desc="Evaluating datasets"):
        try:
            cache = torch_load(cache_path_for(dataset))
        except FileNotFoundError:
            print(f"[{probe_path.stem}] Skipping {dataset}: cache missing.")
            continue

        split = build_difference_split(cache)
        metrics = evaluate_probe_on_dataset(clf, split.eval_X, split.eval_y, dataset)
        dataset_summaries[dataset] = metrics

        roc_curve_data = metrics.get("roc_curve")
        if roc_curve_data:
            plt.plot(
                roc_curve_data["fpr"],
                roc_curve_data["tpr"],
                label=f"{dataset} (AUROC={metrics['auroc']:.3f})",
                linewidth=2.0,
            )
            any_curve = True
        else:
            print(f"[{probe_path.stem}] No ROC curve for {dataset}: insufficient class variety.")

    if any_curve:
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Difference Probe ROC: {probe_path.stem} ({split_name})")
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.4)
    else:
        plt.text(0.5, 0.5, "No ROC curves available", ha="center", va="center")
        plt.axis("off")

    plot_path = PLOT_DIR / f"{probe_path.stem}_transfer.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    summary = {
        "probe": probe_path.stem,
        "probe_path": str(probe_path),
        "split": split_name,
        "layer_index": LAYER_INDEX,
        "datasets": dataset_summaries,
        "metadata": probe_payload.get("metadata"),
    }

    out_path = RESULTS_DIR / f"{probe_path.stem}_transfer.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{probe_path.stem}] Wrote metrics -> {out_path}")
    print(f"[{probe_path.stem}] Wrote ROC plot -> {plot_path}")


if __name__ == "__main__":
    main()
