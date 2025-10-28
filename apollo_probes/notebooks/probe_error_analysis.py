# %% [markdown]
# # Apollo Probe Error Inspection
#
# Explore where a trained deception probe makes confident mistakes on specific Cadenza evaluation splits.
# The notebook mirrors the production evaluation pipeline: it pools detection token activations per dialogue,
# applies the saved probe direction (including normalization and intercept), and examines false positives/negatives.

# %%
from __future__ import annotations

import math
import ast
from pathlib import Path
from typing import Iterable, Sequence
import sys
from collections import defaultdict

import numpy as np
import torch
from datasets.utils.logging import disable_progress_bar, set_verbosity_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from probe_pipeline.utils import load_filtered_dataset

disable_progress_bar()
set_verbosity_error()

from apollo_probes.config import RANDOM_SEED
from apollo_probes.evaluate_probes import compute_scores, load_eval_cache, load_probe
from apollo_probes.eval_datasets import liars_bench_specs, slugify_config
from apollo_probes.pooling import pool_dialogue_activations


# %%
# ---- Configuration ----
TRAIN_DATASET_SLUG = "repe_honesty__plain+roleplaying__plain"
EVAL_CONFIG = "instructed-deception"
LAYER_INDEX = 34
TOP_K = 10
EVAL_SAMPLE_SIZE: int | None = None  # match cached evals (None = full split)
THRESHOLD = 0.0  # default probe decision boundary (logit >= 0 predicts deceptive)


# %%
def get_eval_spec(config_name: str):
    specs = {spec.config: spec for spec in liars_bench_specs()}
    if config_name not in specs:
        raise ValueError(f"Unknown eval config '{config_name}'. Available: {sorted(specs)}")
    return specs[config_name]


def load_eval_rows(config_name: str, sample_size: int | None, seed: int):
    spec = get_eval_spec(config_name)
    try:
        rows = load_filtered_dataset(
            spec.dataset_id,
            spec.config,
            split=spec.split,
            filter_model_name=spec.filter_model,
            sample_size=sample_size,
            seed=seed,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Failed to load the eval dataset. "
            "If this environment is offline, run the notebook after downloading "
            f"'{spec.dataset_id}/{spec.config}' locally or grant network access."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset '{spec.dataset_id}/{spec.config}'. "
            "If this environment lacks internet access, pre-download the dataset "
            "or place a local copy under the Hugging Face cache directory."
        ) from exc
    return rows


def format_dialogue(messages: Sequence[dict]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content", "").strip()
        parts.append(f"{role.upper()}: {content}")
    return "\n".join(parts)


def compute_confusion_details(
    scores: np.ndarray,
    labels: np.ndarray,
    kept_indices: Sequence[int],
    threshold: float,
):
    preds = (scores >= threshold).astype(int)
    false_positive_mask = (preds == 1) & (labels == 0)
    false_negative_mask = (preds == 0) & (labels == 1)

    fp_indices = np.where(false_positive_mask)[0]
    fn_indices = np.where(false_negative_mask)[0]

    # Higher positive logit => more confident deception prediction
    fp_order = fp_indices[np.argsort(scores[fp_indices])[::-1]]
    # More negative logit => more confident honesty prediction
    fn_order = fn_indices[np.argsort(scores[fn_indices])]

    def collect(indices: Iterable[int]) -> list[dict[str, object]]:
        examples: list[dict[str, object]] = []
        for local_idx in indices:
            dataset_idx = int(kept_indices[local_idx])
            score = float(scores[local_idx])
            prob = float(1 / (1 + math.exp(-score)))
            examples.append(
                {
                    "dataset_idx": dataset_idx,
                    "score": score,
                    "probability": prob,
                    "label": int(labels[local_idx]),
                    "prediction": int(preds[local_idx]),
                }
            )
        return examples

    return collect(fp_order), collect(fn_order)


# %%
# ---- Load cached activations & probe ----
eval_alias = slugify_config(EVAL_CONFIG)
probe = load_probe(TRAIN_DATASET_SLUG, LAYER_INDEX)
payload = load_eval_cache(eval_alias, LAYER_INDEX)

counts: Sequence[int | None] = payload["counts"]
dialogue_labels: torch.Tensor = payload["dialogue_labels"]
activations: torch.Tensor = payload["activations"]

pooled_acts, pooled_labels, skipped_dialogues = pool_dialogue_activations(
    activations, counts, dialogue_labels
)
scores = compute_scores(pooled_acts.to(torch.float32), probe)
labels = pooled_labels.numpy()

kept_dialogue_indices = [idx for idx, count in enumerate(counts) if count and count > 0]
assert len(kept_dialogue_indices) == len(labels), "Mismatch between pooled labels and kept dialogues."

print(f"Loaded probe '{TRAIN_DATASET_SLUG}' layer {LAYER_INDEX}")
print(f"Eval dataset '{EVAL_CONFIG}' ({eval_alias}) -> {len(labels)} dialogues (skipped {skipped_dialogues})")


# %%
# ---- Inspect aggregate metrics ----
predictions = (scores >= THRESHOLD).astype(int)
accuracy = float((predictions == labels).mean())
num_fp = int(((predictions == 1) & (labels == 0)).sum())
num_fn = int(((predictions == 0) & (labels == 1)).sum())

print(f"Decision threshold: {THRESHOLD:.2f}")
print(f"Accuracy: {accuracy:.3f}")
print(f"False positives: {num_fp} | False negatives: {num_fn}")


# %%
# ---- Gather high-confidence mistakes ----
false_positive_examples, false_negative_examples = compute_confusion_details(
    scores,
    labels,
    kept_dialogue_indices,
    threshold=THRESHOLD,
)

print(f"High-confidence false positives available: {len(false_positive_examples)}")
print(f"High-confidence false negatives available: {len(false_negative_examples)}")


# %%
# ---- Reload raw dialogues to display text ----
eval_rows = load_eval_rows(EVAL_CONFIG, sample_size=EVAL_SAMPLE_SIZE, seed=RANDOM_SEED)

assert len(eval_rows) == len(counts), (
    "Dialogue count mismatch between eval cache and dataset rows. "
    "Regenerate caches with matching parameters if this persists."
)


# %%
# ---- Display helper ----
def display_examples(tag: str, examples: Sequence[dict[str, object]], top_k: int):
    print(f"\n=== {tag}: top {top_k} ===")
    for example in examples[:top_k]:
        idx = example["dataset_idx"]
        row = eval_rows[idx]
        messages = row["messages"]
        print(f"\nExample #{idx}")
        print(f"  Score: {example['score']:.3f} | Probability(deceptive): {example['probability']:.3f}")
        print(f"  Label: {example['label']} | Prediction: {example['prediction']}")
        if "row_id" in row:
            print(f"  Row ID: {row['row_id']}")
        print("-" * 40)
        print(format_dialogue(messages))
        print("-" * 40)


# %%
# ---- Render top mistakes ----
display_examples("False Positives", false_positive_examples, TOP_K)
display_examples("False Negatives", false_negative_examples, TOP_K)


# %%
# ---- Meta analysis: instructed deceptiveness ----
def parse_meta(meta_value):
    if isinstance(meta_value, dict):
        return meta_value
    if isinstance(meta_value, str):
        try:
            parsed = ast.literal_eval(meta_value)
        except (ValueError, SyntaxError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


prediction_map = {int(idx): int(predictions[pos]) for pos, idx in enumerate(kept_dialogue_indices)}
label_map = {int(idx): int(labels[pos]) for pos, idx in enumerate(kept_dialogue_indices)}

fp_with_instruction = 0
for example in false_positive_examples:
    dataset_idx = int(example["dataset_idx"])
    meta = parse_meta(eval_rows[dataset_idx].get("meta"))
    if meta.get("instructed_deceptiveness", False):
        fp_with_instruction += 1

total_fp = len(false_positive_examples)
fp_share = (fp_with_instruction / total_fp) if total_fp else float("nan")

print("\n--- False positive instructed deceptiveness ---")
print(f"False positives with instructed_deceptiveness=True: {fp_with_instruction} / {total_fp} ({fp_share:.1%})")


target_total = 0
target_correct = 0
target_missing = 0
for dataset_idx, row in enumerate(eval_rows):
    meta = parse_meta(row.get("meta"))
    if not meta.get("instructed_deceptiveness", False):
        continue
    label = label_map.get(dataset_idx)
    if label is None:
        target_missing += 1
        continue
    if label != 0:
        continue
    target_total += 1
    pred = prediction_map.get(dataset_idx)
    if pred is None:
        target_missing += 1
        continue
    if pred == 0:
        target_correct += 1

coverage_total = target_total + target_missing
accuracy = (target_correct / target_total) if target_total else float("nan")

print("\n--- Dataset coverage for (label=False, instructed_deceptiveness=True) ---")
print(f"Examples evaluated: {target_total} (skipped due to missing activations: {target_missing})")
print(f"Correctly predicted HONEST: {target_correct} / {target_total} ({accuracy:.1%})")

# %%
# ---- Correct option analysis for instructed deceptiveness ----
subset_total = 0
subset_correct_option_zero = 0
subset_missing_meta = 0

overall_total = 0
overall_correct_option_zero = 0
overall_missing_meta = 0

for idx, row in enumerate(eval_rows):
    meta = parse_meta(row.get("meta"))
    if not meta.get("instructed_deceptiveness", False):
        continue

    correct_option = meta.get("correct_option")
    if correct_option is None:
        overall_missing_meta += 1
    else:
        overall_total += 1
        if int(correct_option) == 0:
            overall_correct_option_zero += 1

    label = row.get("deceptive")
    if label is None:
        label = label_map.get(idx)
    if label is None:
        subset_missing_meta += 1
        continue
    if int(label) != 0:
        continue

    if correct_option is None:
        subset_missing_meta += 1
        continue

    subset_total += 1
    if int(correct_option) == 0:
        subset_correct_option_zero += 1

subset_pct = (subset_correct_option_zero / subset_total) if subset_total else float("nan")
overall_pct = (overall_correct_option_zero / overall_total) if overall_total else float("nan")

print("\n--- Correct option stats (instructed_deceptiveness=True) ---")
print(
    f"Label=False subset: correct_option=0 for {subset_correct_option_zero} / {subset_total} "
    f"({subset_pct:.1%}) | missing: {subset_missing_meta}"
)
print(
    f"All instructed examples: correct_option=0 for {overall_correct_option_zero} / {overall_total} "
    f"({overall_pct:.1%}) | missing: {overall_missing_meta}"
)

# %%
# ---- Conditional truthfulness given correct option ----
truth_given_false_total = 0
truth_given_false_correct = 0

truth_given_true_total = 0
truth_given_true_correct = 0

truth_missing = 0

for idx, row in enumerate(eval_rows):
    meta = parse_meta(row.get("meta"))
    if not meta.get("instructed_deceptiveness", False):
        continue

    correct_option = meta.get("correct_option")
    if correct_option is None:
        truth_missing += 1
        continue

    label = label_map.get(idx)
    if label is None:
        truth_missing += 1
        continue

    if int(correct_option) == 0:
        truth_given_false_total += 1
        if label == 0:
            truth_given_false_correct += 1
    else:
        truth_given_true_total += 1
        if label == 0:
            truth_given_true_correct += 1

truth_given_false_pct = (
    truth_given_false_correct / truth_given_false_total if truth_given_false_total else float("nan")
)
truth_given_true_pct = (
    truth_given_true_correct / truth_given_true_total if truth_given_true_total else float("nan")
)

print("\n--- Truthfulness conditional on ground-truth statement ---")
print(
    f"P(truth | correct_option=0) = {truth_given_false_correct} / {truth_given_false_total} "
    f"({truth_given_false_pct:.1%})"
)
print(
    f"P(truth | correct_option=1) = {truth_given_true_correct} / {truth_given_true_total} "
    f"({truth_given_true_pct:.1%})"
)
print(f"Missing label/meta entries skipped: {truth_missing}")


# %%
# ---- Probe firing conditional on correct option for truthful responses ----
probe_fire_zero_total = 0
probe_fire_zero_triggered = 0

probe_fire_one_total = 0
probe_fire_one_triggered = 0

probe_fire_missing = 0

for idx, row in enumerate(eval_rows):
    meta = parse_meta(row.get("meta"))
    if not meta.get("instructed_deceptiveness", False):
        continue

    label = label_map.get(idx)
    pred = prediction_map.get(idx)
    if label is None or pred is None:
        probe_fire_missing += 1
        continue
    if label != 0:
        continue

    correct_option = meta.get("correct_option")
    if correct_option is None:
        probe_fire_missing += 1
        continue

    if int(correct_option) == 0:
        probe_fire_zero_total += 1
        if pred == 1:
            probe_fire_zero_triggered += 1
    else:
        probe_fire_one_total += 1
        if pred == 1:
            probe_fire_one_triggered += 1

prob_fire_zero = probe_fire_zero_triggered / probe_fire_zero_total if probe_fire_zero_total else float("nan")
prob_fire_one = probe_fire_one_triggered / probe_fire_one_total if probe_fire_one_total else float("nan")

print("\n--- Probe firing when model tells the truth under instruction ---")
print(
    f"P(probe=1 | label=0, correct_option=0) = {probe_fire_zero_triggered} / {probe_fire_zero_total} "
    f"({prob_fire_zero:.1%})"
)
print(
    f"P(probe=1 | label=0, correct_option=1) = {probe_fire_one_triggered} / {probe_fire_one_total} "
    f"({prob_fire_one:.1%})"
)
print(f"Missing entries skipped: {probe_fire_missing}")


# %%
# ---- Breakdown by source dataset ----
dataset_stats = defaultdict(lambda: {"total": 0, "correct": 0, "missing": 0})

for dataset_idx, row in enumerate(eval_rows):
    meta = parse_meta(row.get("meta"))
    if not meta.get("instructed_deceptiveness", False):
        continue
    dataset_name = row.get("dataset", "unknown")
    label = label_map.get(dataset_idx)
    pred = prediction_map.get(dataset_idx)
    if label is None or pred is None:
        dataset_stats[dataset_name]["missing"] += 1
        continue
    if label != 0:
        continue
    dataset_stats[dataset_name]["total"] += 1
    if pred == 0:
        dataset_stats[dataset_name]["correct"] += 1

print("\n--- Per-dataset accuracy for (label=False, instructed_deceptiveness=True) ---")
header = f"{'Dataset':40s} {'Total':>7s} {'Correct':>8s} {'Missing':>8s} {'Accuracy':>10s}"
print(header)
print("-" * len(header))
for name, stats in sorted(dataset_stats.items(), key=lambda item: item[1]["total"], reverse=True):
    total = stats["total"]
    correct = stats["correct"]
    missing = stats["missing"]
    acc = (correct / total) if total else float("nan")
    acc_str = f"{acc:.1%}" if total else "   n/a"
    print(f"{name:40s} {total:7d} {correct:8d} {missing:8d} {acc_str:>10s}")


# %%
# ---- Inspect correctly predicted honest examples with instruction ----
true_negative_examples: list[dict[str, object]] = []
for pos, dataset_idx in enumerate(kept_dialogue_indices):
    lbl = int(labels[pos])
    pred = int(predictions[pos])
    if lbl != 0 or pred != 0:
        continue
    meta = parse_meta(eval_rows[dataset_idx].get("meta"))
    if not meta.get("instructed_deceptiveness", False):
        continue
    score = float(scores[pos])
    prob = float(1 / (1 + math.exp(-score)))
    true_negative_examples.append(
        {
            "dataset_idx": dataset_idx,
            "score": score,
            "probability": prob,
            "label": lbl,
            "prediction": pred,
        }
    )

true_negative_examples.sort(key=lambda entry: entry["score"])

print("\n--- True negatives with instructed_deceptiveness=True ---")
print(f"Available examples: {len(true_negative_examples)}")

display_examples("True Negatives (instructed)", true_negative_examples, TOP_K)

# %%
