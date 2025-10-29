# %% [markdown]
# Cache activations for Liars' Bench datasets.
# Config params at top of script.

from __future__ import annotations

import json
import os
import torch

from model_configs import get_model_config
from utils import (
    init_model,
    load_filtered_dataset,
    tokenize_dataset_rows,
    extract_last_assistant_activations,
)


def dataset_slug(config: str) -> str:
    return config.replace("/", "_")


# Configuration
CONFIG_KEY = os.environ.get("PROBE_MODEL", "llama")
CONFIG = get_model_config(CONFIG_KEY)

MODEL_NAME = CONFIG.model_name
DATASET_ID = "Cadenza-Labs/liars-bench"
DATASET_CONFIGS = CONFIG.dataset_configs
FILTER_MODEL_NAME = CONFIG.filter_model_name
SPLIT = "test"
if "SAMPLE_SIZE" in os.environ:
    SAMPLE_SIZE: int | None = int(os.environ["SAMPLE_SIZE"])
else:
    SAMPLE_SIZE = CONFIG.default_sample_size
DEFAULT_LAYER_INDEX = 30
LAYER_INDEX = int(os.environ.get("LAYER_INDEX", DEFAULT_LAYER_INDEX))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", CONFIG.default_batch_size))
OUTPUT_DIR = CONFIG.cache_dir
RANDOM_SEED = 42


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if CONFIG.assistant_header_override:
    os.environ["PROBE_ASSISTANT_HEADER_OVERRIDE"] = CONFIG.assistant_header_override
else:
    os.environ.pop("PROBE_ASSISTANT_HEADER_OVERRIDE", None)


def cache_dataset(config: str) -> None:
    cache_file = OUTPUT_DIR / f"{dataset_slug(config)}.pt"
    meta_file = OUTPUT_DIR / f"{dataset_slug(config)}.meta.json"

    def clear_existing_cache() -> None:
        for path in (cache_file, meta_file):
            if path.exists():
                path.unlink()

    tokenizer, model, device, _dtype = init_model(MODEL_NAME, seed=RANDOM_SEED)
    if CONFIG.tokenizer_padding_side:
        tokenizer.padding_side = CONFIG.tokenizer_padding_side
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    dataset = load_filtered_dataset(
        DATASET_ID,
        config,
        SPLIT,
        FILTER_MODEL_NAME,
        SAMPLE_SIZE,
        RANDOM_SEED,
    )

    total_rows = len(dataset)
    if total_rows == 0:
        clear_existing_cache()
        print(f"Skipping {config}: no rows matched filter '{FILTER_MODEL_NAME}'.")
        return

    conversations = tokenize_dataset_rows(dataset, tokenizer)
    try:
        activations, labels = extract_last_assistant_activations(
            conversations,
            layer_index=LAYER_INDEX,
            batch_size=BATCH_SIZE,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )
    except ValueError as exc:
        clear_existing_cache()
        print(f"Skipping {config}: {exc}")
        return

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

    torch.save({"activations": activations, "labels": labels, "meta": payload}, cache_file)

    meta_file.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(f"Cached {config}: {activations.shape[0]} samples")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    for cfg in DATASET_CONFIGS:
        cache_dataset(cfg)
