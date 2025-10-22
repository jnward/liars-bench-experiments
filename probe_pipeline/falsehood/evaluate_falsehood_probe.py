# %% [markdown]
# Evaluate falsehood probes on cached activations, emitting per-dataset ROC plots.

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def slugify(name: str) -> str:
    return name.replace("/", "_")


DEFAULT_LAYER_INDEX = 22
LAYER_INDEX = int(os.environ.get("LAYER_INDEX", DEFAULT_LAYER_INDEX))
PROBE_NAME = os.environ.get("PROBE_NAME")  # optional; stem without .pkl is ok
SPLIT = os.environ.get("EVAL_SPLIT", "eval")

PROBE_DIR = Path(f"probe_pipeline/falsehood/probes/layer{LAYER_INDEX}")
CACHE_DIR = Path("probe_pipeline/falsehood/cache")
PLOT_DIR = Path(f"probe_pipeline/falsehood/plots/layer{LAYER_INDEX}")
RESULTS_DIR = Path(f"probe_pipeline/falsehood/results/layer{LAYER_INDEX}")

DATASET_KEYS: List[str] = [
    "convincing-game",
    "harm-pressure-choice",
    "harm-pressure-knowledge-report",
    "instructed-deception",
    "insider-trading/report",
    "insider-trading/confirmation",
]


PLOT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def probe_paths() -> Iterable[Path]:
    if PROBE_NAME:
        stem = PROBE_NAME[:-4] if PROBE_NAME.endswith(".pkl") else PROBE_NAME
        path = PROBE_DIR / f"{stem}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Requested probe '{stem}' not found at {path}")
        return [path]
    return sorted(PROBE_DIR.glob("*.pkl"))


def load_probe(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    return data["model"], data


def evaluate_probe(path: Path) -> Dict[str, dict]:
    clf, info = load_probe(path)
    probe_name = path.stem

    trained_on: Sequence[str] = info.get("trained_on") or []
    all_datasets: Sequence[str] = info.get("datasets") or list(DATASET_KEYS)

    result: Dict[str, dict] = {
        "probe": probe_name,
        "trained_on": list(trained_on),
        "split": SPLIT,
        "datasets": {},
        "layer_index": LAYER_INDEX,
    }

    plt.figure(figsize=(7, 6))
    any_curve = False

    if len(trained_on) == len(all_datasets):
        highlight_mode = "all"
    elif len(trained_on) == 1:
        highlight_mode = "single"
    else:
        highlight_mode = "leaveout"

    for dataset in all_datasets:
        try:
            cache = load_cache(dataset)
        except FileNotFoundError:
            print(f"[{probe_name}] Skipping {dataset}: cache missing.")
            continue

        try:
            X_eval, y_eval = get_split(cache, SPLIT)
        except KeyError:
            print(f"[{probe_name}] Skipping {dataset}: split '{SPLIT}' missing.")
            continue

        probs = clf.predict_proba(X_eval)[:, 1]
        preds = (probs > 0.5).astype(int)

        acc = accuracy_score(y_eval, preds)
        try:
            auroc = roc_auc_score(y_eval, probs)
        except ValueError:
            auroc = float("nan")

        entry: Dict[str, object] = {
            "samples": int(len(y_eval)),
            "accuracy": float(acc),
            "auroc": float(auroc),
        }

        if np.unique(y_eval).size >= 2:
            fpr, tpr, _ = roc_curve(y_eval, probs)
            entry["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

            in_training = dataset in trained_on
            if highlight_mode == "single":
                highlight = dataset == trained_on[0]
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
            print(f"[{probe_name}] No ROC curve for {dataset}: single-class labels in split.")

        result["datasets"][dataset] = entry

    if any_curve:
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Falsehood Probe ROC Curves: {probe_name} ({SPLIT} split)")
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.4)
    else:
        plt.text(0.5, 0.5, "Insufficient data for ROC curves", ha="center", va="center")
        plt.axis("off")

    plot_path = PLOT_DIR / f"{probe_name}_{SPLIT}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    result["plot_path"] = str(plot_path)

    out_path = RESULTS_DIR / f"{probe_name}_{SPLIT}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["result_path"] = str(out_path)

    print(f"\n[{probe_name}] Saved ROC plot -> {plot_path}")
    print(f"[{probe_name}] Saved metrics -> {out_path}")

    for dataset, stats in result["datasets"].items():
        print(
            f"  {dataset:35s} | samples: {stats['samples']:4d} | "
            f"acc: {stats['accuracy']:6.3f} | auroc: {stats['auroc']:6.3f}"
        )

    return result


def main() -> None:
    paths = list(probe_paths())
    if not paths:
        raise RuntimeError(f"No probe files found in {PROBE_DIR}")

    summary: Dict[str, dict] = {}
    for path in paths:
        summary[path.stem] = evaluate_probe(path)

    summary_path = RESULTS_DIR / f"evaluation_summary_{SPLIT}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved evaluation summary -> {summary_path}")


if __name__ == "__main__":
    main()
