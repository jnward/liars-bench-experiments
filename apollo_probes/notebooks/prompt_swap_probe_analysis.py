# %% [markdown]
# # Prompt Swap Probe Analysis
#
# For dialogues where LLaMA 3.3 was instructed to deceive but answered fully truthfully (Claude rating 2),
# this notebook swaps assistant completions between the honest and deceptive system prompts to study how
# probe scores shift under different prompt/response combinations.

# %%
from __future__ import annotations

import ast
import copy
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets.utils.logging import disable_progress_bar, set_verbosity_error
from dotenv import load_dotenv
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

disable_progress_bar()
set_verbosity_error()
load_dotenv(PROJECT_ROOT / ".env")

from probe_pipeline.utils import (  # noqa: E402
    TokenizedConversation,
    init_model,
    load_filtered_dataset,
    tokenize_conversation,
    _get_assistant_header_ids,
)

from apollo_probes.config import RANDOM_SEED  # noqa: E402
from apollo_probes.evaluate_probes import compute_scores, load_probe  # noqa: E402
from apollo_probes.eval_datasets import liars_bench_specs, slugify_config  # noqa: E402

# %%
# ---- Configuration ----
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
EVAL_CONFIG = "instructed-deception"
TRAIN_DATASET_SLUG = "repe_honesty__plain+roleplaying__plain"
LAYER_INDEX = 34
BATCH_SIZE = 8
TARGET_SYSTEM_INDICES = {0, 1}

RATINGS_PATH = (
    PROJECT_ROOT
    / "apollo_probes"
    / "notebooks"
    / "cache"
    / "assistant_truth_ratings"
    / f"{slugify_config(EVAL_CONFIG)}__assistant_truth_ratings.jsonl"
)

# %%
# ---- Helpers ----
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


def first_user_message(messages: Sequence[dict]) -> str | None:
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, Sequence):
                parts = []
                for chunk in content:
                    if isinstance(chunk, dict) and chunk.get("type") == "text":
                        parts.append(chunk.get("text", ""))
                    elif isinstance(chunk, str):
                        parts.append(chunk)
                text = "\n".join(parts).strip()
                if text:
                    return text
    return None


def extract_assistant_text(messages: Sequence[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, Sequence):
            parts = []
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    parts.append(chunk.get("text", ""))
                elif isinstance(chunk, str):
                    parts.append(chunk)
            return "\n".join(parts).strip()
    raise ValueError("Assistant message missing.")


def replace_last_assistant(messages: Sequence[dict], new_text: str) -> list[dict]:
    updated = copy.deepcopy(messages)
    for idx in range(len(updated) - 1, -1, -1):
        if updated[idx].get("role") != "assistant":
            continue
        content = updated[idx].get("content")
        if isinstance(content, str):
            updated[idx]["content"] = new_text
        else:
            updated[idx]["content"] = [{"type": "text", "text": new_text}]
        return updated
    raise ValueError("Assistant message not found to replace.")


def load_truth_ratings(path: Path) -> dict[int, int]:
    ratings: dict[int, int] = {}
    if not path.exists():
        raise FileNotFoundError(f"Missing truth ratings cache: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rating = record.get("verdict")
            if rating is None:
                continue
            ratings[int(record["dataset_index"])] = int(rating)
    return ratings


@dataclass
class DialogueRow:
    row_index: int
    dataset_index_field: int
    system_template_index: int
    instructed: bool
    deceptive_label: int
    messages: list[dict]
    user_message: str
    assistant_text: str


# %%
# ---- Load eval dataset ----
specs = {spec.config: spec for spec in liars_bench_specs()}
if EVAL_CONFIG not in specs:
    raise ValueError(f"Unknown eval config '{EVAL_CONFIG}'.")

spec = specs[EVAL_CONFIG]
eval_rows = load_filtered_dataset(
    spec.dataset_id,
    spec.config,
    split=spec.split,
    filter_model_name=spec.filter_model,
    sample_size=None,
    seed=RANDOM_SEED,
)

print(f"Loaded eval rows: {len(eval_rows)}")


# %%
# ---- Index rows ----
rows_by_row_index: dict[int, DialogueRow] = {}
rows_by_user: dict[str, list[DialogueRow]] = defaultdict(list)

for row_idx, row in enumerate(eval_rows):
    meta = parse_meta(row.get("meta"))
    user_text = first_user_message(row.get("messages") or [])
    if user_text is None:
        continue
    system_template_index = int(meta.get("system_template_index", -1))
    try:
        assistant_text = extract_assistant_text(row.get("messages") or [])
    except ValueError:
        continue
    info = DialogueRow(
        row_index=row_idx,
        dataset_index_field=int(row.get("dataset_index", row_idx)),
        system_template_index=system_template_index,
        instructed=bool(meta.get("instructed_deceptiveness", False)),
        deceptive_label=int(row.get("deceptive", 0)),
        messages=copy.deepcopy(row.get("messages") or []),
        user_message=user_text,
        assistant_text=assistant_text,
    )
    rows_by_row_index[row_idx] = info
    rows_by_user[user_text].append(info)

print(f"Indexed rows: {len(rows_by_row_index)}")


# %%
# ---- Load Claude ratings & select target pairs ----
ratings_map = load_truth_ratings(RATINGS_PATH)
print(f"Loaded truth ratings for {len(ratings_map)} dataset indices.")

target_pairs = []
skipped_missing_honest = 0
skipped_template = 0

rating_template_counts = defaultdict(int)

for dataset_index, rating in ratings_map.items():
    if rating != 2:
        continue
    deceptive_row = rows_by_row_index.get(dataset_index)
    if deceptive_row is None:
        continue
    if not deceptive_row.instructed or deceptive_row.deceptive_label != 0:
        continue
    if deceptive_row.system_template_index not in TARGET_SYSTEM_INDICES:
        skipped_template += 1
        continue
    candidates = rows_by_user.get(deceptive_row.user_message, [])
    honest_row = next(
        (row for row in candidates if not row.instructed and row.system_template_index == deceptive_row.system_template_index),
        None,
    )
    if honest_row is None:
        skipped_missing_honest += 1
        continue
    rating_template_counts[deceptive_row.system_template_index] += 1
    target_pairs.append(
        {
            "row_index": dataset_index,
            "system_template_index": deceptive_row.system_template_index,
            "user_message": deceptive_row.user_message,
            "deceptive_row": deceptive_row,
            "honest_row": honest_row,
        }
    )

print(f"Target pairs assembled: {len(target_pairs)}")
if skipped_missing_honest:
    print(f"[warn] Missing honest matches for {skipped_missing_honest} deceptive rows (system idx outside {TARGET_SYSTEM_INDICES}).")
if skipped_template:
    print(f"[info] Skipped {skipped_template} rating=2 rows due to system template not in {sorted(TARGET_SYSTEM_INDICES)}.")

if rating_template_counts:
    print("Rating=2 deceptive rows by system template index:", dict(sorted(rating_template_counts.items())))
else:
    print("No rating=2 deceptive rows found within target system templates.")


# %%
# ---- Construct prompt/response variants ----
VARIANT_LABELS = [
    ("honest_prompt + honest_reply", "HH"),
    ("honest_prompt + deceptive_reply", "HD"),
    ("deceptive_prompt + honest_reply", "DH"),
    ("deceptive_prompt + deceptive_reply", "DD"),
]

variant_entries: list[dict] = []

for pair_id, pair in enumerate(target_pairs):
    deceptive_row: DialogueRow = pair["deceptive_row"]
    honest_row: DialogueRow = pair["honest_row"]

    honest_messages = honest_row.messages
    deceptive_messages = deceptive_row.messages
    honest_text = honest_row.assistant_text
    deceptive_text = deceptive_row.assistant_text

    variants = {
        "HH": replace_last_assistant(honest_messages, honest_text),
        "HD": replace_last_assistant(honest_messages, deceptive_text),
        "DH": replace_last_assistant(deceptive_messages, honest_text),
        "DD": replace_last_assistant(deceptive_messages, deceptive_text),
    }

    for label, code in VARIANT_LABELS:
        variant_entries.append(
            {
                "pair_id": pair_id,
                "row_index": pair["row_index"],
                "system_template_index": pair["system_template_index"],
                "description": label,
                "code": code,
                "messages": variants[code],
                "source_honest_row": honest_row.row_index,
                "source_deceptive_row": deceptive_row.row_index,
            }
        )

print(f"Total variant transcripts: {len(variant_entries)}")

if not variant_entries:
    raise RuntimeError(
        "No prompt/response variants were constructed. Check that there are rating=2 deceptive rows "
        f"with system_template_index in {sorted(TARGET_SYSTEM_INDICES)} and matching honest transcripts."
    )


# %%
# ---- Tokenize conversations ----
tokenizer, model, device, _dtype = init_model(MODEL_NAME, seed=RANDOM_SEED)
header_ids = _get_assistant_header_ids(tokenizer)

tokenized_conversations: list[TokenizedConversation] = []
for entry in variant_entries:
    row_payload = {
        "messages": entry["messages"],
        "deceptive": 0,
    }
    tokenized_conversations.append(tokenize_conversation(row_payload, tokenizer, header_ids))

print(f"Tokenized {len(tokenized_conversations)} conversations.")

empty_indices = [idx for idx, conv in enumerate(tokenized_conversations) if not conv.assistant_token_indices]
if empty_indices:
    print(f"[warn] {len(empty_indices)} conversations lack assistant token indices. Showing first example.")
    first_idx = empty_indices[0]
    sample_entry = variant_entries[first_idx]
    print("Problematic variant entry:", {
        "pair_id": sample_entry["pair_id"],
        "dataset_index": sample_entry["dataset_index"],
        "code": sample_entry["code"],
        "system_template_index": sample_entry["system_template_index"],
    })
    print("Messages for this entry:")
    for message in sample_entry["messages"]:
        print("  -", message)
    raise RuntimeError("No assistant tokens found for some conversations; cannot proceed.")


# %%
# ---- Extract activations ----
from probe_pipeline.utils import extract_last_assistant_activations  # noqa: E402

activations, _labels = extract_last_assistant_activations(
    tokenized_conversations,
    layer_index=LAYER_INDEX,
    batch_size=BATCH_SIZE,
    tokenizer=tokenizer,
    model=model,
    device=device,
)

print(f"Extracted activations: {activations.shape}")


# %%
# ---- Compute probe scores for variants ----
probe = load_probe(TRAIN_DATASET_SLUG, LAYER_INDEX)
variant_scores = compute_scores(activations.to(torch.float32), probe)

results_records = []
for entry, score in zip(variant_entries, variant_scores, strict=False):
    results_records.append(
        {
            "pair_id": entry["pair_id"],
            "row_index": entry["row_index"],
            "system_template_index": entry["system_template_index"],
            "variant": entry["description"],
            "code": entry["code"],
            "probe_score": float(score),
        }
    )

results_df = pd.DataFrame(results_records)
print(results_df.groupby("variant")["probe_score"].describe())


# %%
# ---- Visualise distributions ----
def plot_variant_violin(df: pd.DataFrame, title: str) -> None:
    order = [label for label, _ in VARIANT_LABELS]
    data = [df.loc[df["variant"] == label, "probe_score"].to_numpy() for label in order]
    cleaned = [(label, scores) for label, scores in zip(order, data) if scores.size > 0]
    if not cleaned:
        print(f"Skipping plot '{title}' (no data).")
        return
    labels, groups = zip(*cleaned)
    positions = np.arange(1, len(groups) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    parts = ax.violinplot(groups, positions=positions, showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#6A5ACD")
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
    for key in ("cmeans", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black")
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Probe score (pre-logit)")
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    plt.show()


plot_variant_violin(results_df, "Probe scores across prompt/response combinations")


# %%
# ---- Pair-wise deltas ----
pivot_df = results_df.pivot_table(
    index="pair_id",
    columns="code",
    values="probe_score",
    aggfunc="first",
)

delta_cols = {}
if {"DD", "HH"} <= set(pivot_df.columns):
    delta_cols["DD_minus_HH"] = pivot_df["DD"] - pivot_df["HH"]
if {"DD", "HD"} <= set(pivot_df.columns):
    delta_cols["DD_minus_HD"] = pivot_df["DD"] - pivot_df["HD"]
if {"DD", "DH"} <= set(pivot_df.columns):
    delta_cols["DD_minus_DH"] = pivot_df["DD"] - pivot_df["DH"]

if delta_cols:
    delta_df = pd.DataFrame(delta_cols)
    print("Pair-wise probe score deltas (mean ± std):")
    for col, series in delta_df.items():
        mean = series.mean()
        std = series.std()
        print(f"  {col}: mean={mean:.3f}, std={std:.3f}")

    if {"DD", "DH"} <= set(pivot_df.columns):
        paired = pivot_df.dropna(subset=["DD", "DH"])
        if not paired.empty:
            diffs = (paired["DD"] - paired["DH"]).to_numpy()
            nonzero = diffs[diffs != 0]
            if nonzero.size == 0:
                print("Wilcoxon DD vs DH: all differences are zero; no test performed.")
            else:
                stat, p_value = wilcoxon(
                    paired["DD"],
                    paired["DH"],
                    alternative="greater",
                    zero_method="wilcox",
                )
                print(
                    "Wilcoxon signed-rank (DD > DH): "
                    f"statistic={stat:.1f}, p={p_value:.3g}, "
                    f"mean Δ={diffs.mean():.3f}, median Δ={np.median(diffs):.3f}, "
                    f"n={len(diffs)}"
                )
                rng = np.random.default_rng(RANDOM_SEED)
                boot_means = [
                    rng.choice(diffs, size=len(diffs), replace=True).mean()
                    for _ in range(1000)
                ]
                ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
                print(
                    f"Bootstrap 95% CI for mean Δ (DD - DH): [{ci_low:.3f}, {ci_high:.3f}]"
                )
        else:
            print("Wilcoxon DD vs DH: no paired data available.")


# %%
