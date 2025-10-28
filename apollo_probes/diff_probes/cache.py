from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch

from ..config import DEFAULT_BATCH_SIZE, MODEL_NAME, RANDOM_SEED, VAL_FRACTION
from ..data import load_dataset_splits, prepare_datasets
from .config import DEFAULT_LAYER_INDEX
from .types import DialogueInfo, VariantResult
from .paths import diff_cache_path
from .processing import DifferenceGroup, assemble_difference_vectors, extract_variant_results, prepare_requests
from probe_pipeline.utils import init_model


def _assemble_split(results: Sequence[VariantResult]) -> DifferenceGroup:
    return assemble_difference_vectors(results)


def cache_diff_vectors(
    dataset_args: Iterable[str],
    layer: int = DEFAULT_LAYER_INDEX,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_fraction: float = VAL_FRACTION,
    seed: int = RANDOM_SEED,
    force: bool = False,
) -> str:
    dataset_names, dataset_slug = prepare_datasets(list(dataset_args))
    cache_path = diff_cache_path(dataset_slug, layer)
    if cache_path.exists() and not force:
        print(f"[diff-cache] {dataset_slug} layer{layer:02d} already exists -> {cache_path}")
        return str(cache_path)

    tokenizer, model, device, _dtype = init_model(MODEL_NAME, seed=seed)
    model.eval()

    train_split, val_split = load_dataset_splits(dataset_names, val_fraction=val_fraction, seed=seed)

    def process_split(dialogues, labels) -> DifferenceGroup:
        entries = [
            (
                DialogueInfo(
                    dataset=dataset_slug,
                    dialogue_index=idx,
                    label=int(labels[idx]),
                    meta={},
                ),
                dialogue,
            )
            for idx, dialogue in enumerate(dialogues)
        ]
        requests = prepare_requests(entries)
        results = extract_variant_results(requests, layer_index=layer, batch_size=batch_size, tokenizer=tokenizer, model=model)
        return _assemble_split(results)

    train_cached = process_split(train_split.dialogues, train_split.labels)
    val_cached = process_split(val_split.dialogues, val_split.labels)

    if train_cached.variant_hidden is None or val_cached.variant_hidden is None:
        raise ValueError("Variant hidden states missing; ensure cache generation stores them.")

    payload = {
        "dataset": dataset_slug,
        "datasets": dataset_names,
        "layer_index": layer,
        "seed": seed,
        "val_fraction": val_fraction,
        "train": {
            "labels": train_cached.labels,
            "dialogue_indices": train_cached.dialogue_indices,
            "user_diff": train_cached.user_diff,
            "assistant_diff": train_cached.assistant_diff,
            "combined_diff": train_cached.combined_diff,
            "variant_hidden": train_cached.variant_hidden,
        },
        "val": {
            "labels": val_cached.labels,
            "dialogue_indices": val_cached.dialogue_indices,
            "user_diff": val_cached.user_diff,
            "assistant_diff": val_cached.assistant_diff,
            "combined_diff": val_cached.combined_diff,
            "variant_hidden": val_cached.variant_hidden,
        },
    }

    torch.save(payload, cache_path)
    print(f"[diff-cache] Saved -> {cache_path}")
    return str(cache_path)
