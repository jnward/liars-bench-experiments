# %% [markdown]
# Cache activations for Liars' Bench datasets.
# Config params at top of script.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from utils import (
    init_model,
    load_filtered_dataset,
    tokenize_dataset_rows,
    extract_last_assistant_activations,
)


def dataset_slug(config: str) -> str:
    return config.replace("/", "_")


# Configuration
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
DATASET_ID = "Cadenza-Labs/liars-bench"
DATASET_CONFIGS = [
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
OUTPUT_DIR = Path("probe_pipeline/cache")
RANDOM_SEED = 42


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

    conversations = tokenize_dataset_rows(dataset, tokenizer)
    activations, labels = extract_last_assistant_activations(
        conversations,
        layer_index=LAYER_INDEX,
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

    indices = torch.randperm(labels.size(0))
    train_cut = labels.size(0) // 2
    train_indices = indices[:train_cut]
    eval_indices = indices[train_cut:]

    payload = {
        "dataset_config": config,
        "filter_model": FILTER_MODEL_NAME,
        "layer_index": LAYER_INDEX,
        "seed": RANDOM_SEED,
        "train_indices": train_indices.tolist(),
        "eval_indices": eval_indices.tolist(),
    }

    torch.save(
        {
            "activations": activations,
            "labels": labels,
            "meta": payload,
        },
        OUTPUT_DIR / f"{dataset_slug(config)}_layer{LAYER_INDEX}.pt",
    )

    (OUTPUT_DIR / f"{dataset_slug(config)}_layer{LAYER_INDEX}.meta.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(f"Cached {config}: {activations.shape[0]} samples")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    for cfg in DATASET_CONFIGS:
        cache_dataset(cfg)
