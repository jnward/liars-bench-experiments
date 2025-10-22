# %% [markdown]
# Cache activations for falsehood probes (user judgment correctness).

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List
import sys

import torch
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils import init_model, load_filtered_dataset
from falsehood.utils import (
    TokenizedFalsehoodExample,
    build_falsehood_variants,
    pad_batch,
    tokenize_falsehood_variant,
)


def dataset_slug(config: str) -> str:
    return config.replace("/", "_")


# Configuration
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
DATASET_ID = "Cadenza-Labs/liars-bench"
DATASET_CONFIGS: List[str] = [
    "convincing-game",
    "harm-pressure-choice",
    "harm-pressure-knowledge-report",
    "instructed-deception",
    "insider-trading/report",
    "insider-trading/confirmation",
]
FILTER_MODEL_NAME = "llama-v3.3-70b-instruct"
SPLIT = "test"
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", 1000))
DEFAULT_LAYER_INDEX = 22
LAYER_INDEX = int(os.environ.get("LAYER_INDEX", DEFAULT_LAYER_INDEX))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 2))
OUTPUT_DIR = Path("probe_pipeline/falsehood/cache")
RANDOM_SEED = 42


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_falsehood_activations(
    examples: List[TokenizedFalsehoodExample],
    layer_index: int,
    batch_size: int,
    tokenizer,
    model,
    device,
) -> torch.Tensor:
    activations: List[torch.Tensor] = []

    for start in tqdm(
        range(0, len(examples), batch_size),
        desc="Activations",
        leave=False,
    ):
        batch = examples[start : start + batch_size]
        input_ids = pad_batch([ex.input_ids for ex in batch], tokenizer.pad_token_id)
        attention_mask = pad_batch([ex.attention_mask for ex in batch], 0)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"), torch.no_grad():
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden = outputs.hidden_states[layer_index].to(torch.float32).cpu()

        for sample_idx, example in enumerate(batch):
            activations.append(hidden[sample_idx, example.target_index, :])

    if not activations:
        raise ValueError("No activations extracted; check dataset filtering.")

    return torch.stack(activations, dim=0)


def cache_dataset(config: str) -> None:
    tokenizer, model, device, _dtype = init_model(MODEL_NAME, seed=RANDOM_SEED)
    dataset = load_filtered_dataset(
        DATASET_ID,
        config,
        SPLIT,
        FILTER_MODEL_NAME,
        SAMPLE_SIZE,
        RANDOM_SEED,
    )

    tokenized_examples: List[TokenizedFalsehoodExample] = []
    labels: List[int] = []
    variant_tags: List[str] = []
    group_ids: List[int] = []

    for row_idx, row in enumerate(tqdm(dataset, desc=f"Rows[{config}]", leave=False)):
        messages = row["messages"]
        is_deceptive = bool(row.get("deceptive", False))
        variants = build_falsehood_variants(messages, is_deceptive)
        for variant in variants:
            tokenized = tokenize_falsehood_variant(variant, tokenizer)
            tokenized_examples.append(tokenized)
            labels.append(tokenized.label)
            variant_tags.append(tokenized.tag)
            group_ids.append(row_idx)

    if not tokenized_examples:
        print(f"No examples produced for {config}; skipping.")
        return

    activations = extract_falsehood_activations(
        tokenized_examples,
        layer_index=LAYER_INDEX,
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

    label_tensor = torch.tensor(labels, dtype=torch.long)
    unique_groups = list(dict.fromkeys(group_ids))
    if len(unique_groups) >= 2:
        shuffled_groups = unique_groups[:]
        random.Random(RANDOM_SEED).shuffle(shuffled_groups)
        split_point = max(1, len(shuffled_groups) // 2)
        train_groups = set(shuffled_groups[:split_point])
        train_indices = [idx for idx, gid in enumerate(group_ids) if gid in train_groups]
        eval_indices = [idx for idx, gid in enumerate(group_ids) if gid not in train_groups]
    else:
        shuffled_indices = list(range(len(tokenized_examples)))
        random.Random(RANDOM_SEED).shuffle(shuffled_indices)
        split_point = max(1, len(shuffled_indices) // 2)
        train_indices = shuffled_indices[:split_point]
        eval_indices = shuffled_indices[split_point:]

    if not eval_indices:
        eval_indices = train_indices[:1]

    payload = {
        "dataset_config": config,
        "filter_model": FILTER_MODEL_NAME,
        "layer_index": LAYER_INDEX,
        "seed": RANDOM_SEED,
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "group_ids": group_ids,
        "variant_tags": variant_tags,
    }

    torch.save(
        {
            "activations": activations,
            "labels": label_tensor,
            "meta": payload,
        },
        OUTPUT_DIR / f"{dataset_slug(config)}_layer{LAYER_INDEX}.pt",
    )

    (OUTPUT_DIR / f"{dataset_slug(config)}_layer{LAYER_INDEX}.meta.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(f"Cached {config}: {activations.shape[0]} samples ({len(unique_groups)} conversations)")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    for cfg in tqdm(DATASET_CONFIGS, desc="Configs"):
        cache_dataset(cfg)
