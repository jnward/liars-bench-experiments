from __future__ import annotations

import argparse
from typing import Iterable, Sequence

import torch
from transformers import AutoConfig

from probe_pipeline.utils import init_model

from .activations import extract_layerwise_detection_activations
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET,
    DEFAULT_LAYER,
    DEFAULT_LAYER_SWEEP,
    LAYER_STEP,
    MODEL_NAME,
    RANDOM_SEED,
    VAL_FRACTION,
)
from .data import load_dataset_splits, prepare_datasets
from .paths import layer_cache_path


def cache_layers(
    dataset_names: Sequence[str],
    dataset_slug: str,
    layers: Iterable[int],
    batch_size: int,
    val_fraction: float,
    seed: int,
    force: bool = False,
) -> None:
    layers = sorted(set(int(layer) for layer in layers))
    pending = []
    for layer in layers:
        path = layer_cache_path(dataset_slug, layer)
        if path.exists() and not force:
            print(f"[cache] {dataset_slug} layer {layer} already cached; skipping.")
        else:
            pending.append(layer)

    if not pending:
        print("[cache] Nothing to do.")
        return

    train_split, val_split = load_dataset_splits(dataset_names, val_fraction=val_fraction, seed=seed)
    tokenizer, model, device, _dtype = init_model(MODEL_NAME, seed=seed)

    dataset_desc = ", ".join(dataset_names)
    print(f"[cache] Extracting train activations for layers {pending} on datasets [{dataset_desc}]")
    train_per_layer = extract_layerwise_detection_activations(
        train_split.dialogues,
        train_split.labels,
        model,
        tokenizer,
        pending,
        batch_size,
        desc="Train",
    )

    print(f"[cache] Extracting val activations for layers {pending}")
    val_per_layer = extract_layerwise_detection_activations(
        val_split.dialogues,
        val_split.labels,
        model,
        tokenizer,
        pending,
        batch_size,
        desc="Val",
    )

    for layer in pending:
        cache_path = layer_cache_path(dataset_slug, layer)
        train_info = train_per_layer[layer]
        val_info = val_per_layer[layer]
        train_acts = train_info["activations"]
        train_labels = train_info["labels"]
        train_counts = train_info["counts"]
        train_dialogue_labels = train_info["dialogue_labels"]
        val_acts = val_info["activations"]
        val_labels = val_info["labels"]
        val_counts = val_info["counts"]
        val_dialogue_labels = val_info["dialogue_labels"]

        payload = {
            "dataset": dataset_slug,
            "layer_index": layer,
            "train": {
                "activations": train_acts,
                "labels": train_labels,
                "counts": train_counts,
                "dialogue_labels": train_dialogue_labels,
            },
            "val": {
                "activations": val_acts,
                "labels": val_labels,
                "counts": val_counts,
                "dialogue_labels": val_dialogue_labels,
            },
            "meta": {
                "model_name": MODEL_NAME,
                "val_fraction": val_fraction,
                "train_dialogues": train_split.size,
                "val_dialogues": val_split.size,
                "train_tokens": int(train_labels.size(0)),
                "val_tokens": int(val_labels.size(0)),
                "seed": seed,
            },
        }

        torch.save(payload, cache_path)
        print(f"[cache] Saved -> {cache_path}")


def generate_layers_from_range(start: int, step: int) -> list[int]:
    config = AutoConfig.from_pretrained(MODEL_NAME)
    max_layer = config.num_hidden_layers
    return [layer for layer in range(start, max_layer + 1, step)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache detection token activations for Apollo probe training.")
    parser.add_argument("--dataset", nargs="+", default=[DEFAULT_DATASET], help="Training dataset name(s).")
    parser.add_argument("--layer", type=int, help="Single layer index to cache.")
    parser.add_argument("--layers", type=int, nargs="+", help="Multiple specific layer indices to cache.")
    parser.add_argument("--layer-start", type=int, help="Start layer for range-based caching.")
    parser.add_argument("--layer-step", type=int, default=LAYER_STEP, help="Step for range-based caching.")
    parser.add_argument(
        "--default-sweep",
        action="store_true",
        help="Cache the default layer sweep (2,6,10,...). Overrides manual layer arguments.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for extraction.")
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION, help="Validation split fraction.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for dataset ops.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing caches.")
    return parser.parse_args()


def resolve_layers(args) -> Sequence[int]:
    if args.default_sweep:
        return DEFAULT_LAYER_SWEEP
    if args.layers:
        return args.layers
    if args.layer_start is not None:
        step = args.layer_step or LAYER_STEP
        return generate_layers_from_range(args.layer_start, step)
    if args.layer is not None:
        return [args.layer]
    return [DEFAULT_LAYER]


def main() -> None:
    args = parse_args()
    dataset_names, dataset_slug = prepare_datasets(args.dataset)
    layers = resolve_layers(args)

    cache_layers(
        dataset_names=dataset_names,
        dataset_slug=dataset_slug,
        layers=layers,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
