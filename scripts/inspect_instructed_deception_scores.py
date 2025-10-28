from __future__ import annotations

import ast
import numpy as np
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probe_pipeline.utils import load_filtered_dataset, init_model
from apollo_probes.diff_probes.evaluate import (
    _prepare_eval_entries,
    _compute_difference_group,
    _features_from_group,
    _score_features,
    _logit_to_prob,
)
from apollo_probes.diff_probes.paths import diff_probe_dir
from apollo_probes.diff_probes.config import DEFAULT_LAYER_INDEX

DATASET_SLUG = "repe_honesty__plain+roleplaying__plain"
VARIANT = "user"
LAYER = DEFAULT_LAYER_INDEX
DATASET_ID = "Cadenza-Labs/liars-bench"
CONFIG = "instructed-deception"
SPLIT = "test"
FILTER_MODEL = "llama-v3.3-70b-instruct"
BATCH_SIZE = 2
CHUNK_SIZE = 128
SAMPLE_SIZE = None  # set to int for subsampling


def load_probe() -> dict:
    probe_path = diff_probe_dir(DATASET_SLUG, LAYER) / f"{VARIANT}_probe.pkl"
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe not found at {probe_path}")
    return torch.load(probe_path, map_location="cpu")


import torch
def main() -> None:
    probe = load_probe()
    tokenizer, model, device, _dtype = init_model("meta-llama/Llama-3.3-70B-Instruct", seed=42)
    model.eval()

    dataset = load_filtered_dataset(
        DATASET_ID,
        CONFIG,
        SPLIT,
        FILTER_MODEL,
        sample_size=SAMPLE_SIZE,
        seed=42,
    )
    total = len(dataset)
    print(f"Total dataset rows considered: {total}")

    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    for start in range(0, total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total)
        subset = dataset.select(range(start, end))
        entries = _prepare_eval_entries(subset, CONFIG)
        if not entries:
            continue
        group = _compute_difference_group(
            entries,
            layer=LAYER,
            batch_size=BATCH_SIZE,
            tokenizer=tokenizer,
            model=model,
            store_variant_hidden=False,
        )
        features = _features_from_group(group, VARIANT)
        if features.size == 0:
            continue
        logits = _score_features(features, probe)
        labels = group.labels.numpy()
        logits_list.append(logits)
        labels_list.append(labels)

    if not logits_list:
        print("No logits produced; check configuration.")
        return

    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    probs = _logit_to_prob(logits)

    print(f"Collected logits: {logits.shape[0]} examples")
    print(f"Unique logits: {np.unique(np.round(logits, 6)).shape[0]}")
    print("Logit summary:")
    for q in [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]:
        val = np.quantile(logits, q)
        print(f"  q{int(q*100):02d}: {val:.4f}")

    neg_logits = logits[labels == 0]
    pos_logits = logits[labels == 1]
    print(f"Neg count: {neg_logits.shape[0]}, Pos count: {pos_logits.shape[0]}")
    print(f"Neg quantiles 5/50/95: {np.quantile(neg_logits, [0.05,0.5,0.95])}")
    print(f"Pos quantiles 5/50/95: {np.quantile(pos_logits, [0.05,0.5,0.95])}")

    # ROC curve points to inspect for elbow behaviour
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, probs)
    print("ROC points (first 10):")
    for fp, tp, thr in list(zip(fpr, tpr, thresholds))[:10]:
        print(f"  thr={thr:.4f}, FPR={fp:.4f}, TPR={tp:.4f}")
    print("ROC points (last 10):")
    for fp, tp, thr in list(zip(fpr, tpr, thresholds))[-10:]:
        print(f"  thr={thr:.4f}, FPR={fp:.4f}, TPR={tp:.4f}")


if __name__ == "__main__":
    main()
