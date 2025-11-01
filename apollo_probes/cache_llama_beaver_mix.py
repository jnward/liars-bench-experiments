from __future__ import annotations

import argparse
from typing import Iterable, Sequence

import torch
from transformers import AutoConfig

from probe_pipeline.utils import init_model

from .activations import extract_layerwise_detection_activations
from .config import (
    DEFAULT_BATCH_SIZE,
    LAYER_START,
    LAYER_STEP,
    MODEL_NAME,
    RANDOM_SEED,
    VAL_FRACTION,
)
from .data import canonicalize_dataset_names
from .llama_beaver_mix import (
    BeaverIngestionConfig,
    DEFAULT_CORE_DATASETS,
    beaver_mix_slug,
    split_llama_beaver_mix,
)
from .paths import layer_cache_path


def _resolve_layers(layer: int | None, layers: Sequence[int] | None, layer_start: int | None, layer_step: int) -> Sequence[int]:
    if layers:
        return sorted(set(layers))
    if layer is not None:
        return [layer]
    if layer_start is not None:
        if layer_step <= 0:
            raise ValueError("layer_step must be > 0 when using layer_start.")
        current = int(layer_start)
        resolved: list[int] = []
        config = AutoConfig.from_pretrained(MODEL_NAME)
        max_layer = getattr(config, "num_hidden_layers", None)
        if max_layer is None:
            raise ValueError("Model config missing num_hidden_layers; cannot infer sweep.")
        while current <= max_layer:
            resolved.append(current)
            current += layer_step
        return resolved
    return [LAYER_START]


def _build_beaver_config(args: argparse.Namespace) -> BeaverIngestionConfig:
    categories = tuple(dict.fromkeys(args.beaver_category)) if args.beaver_category else None
    return BeaverIngestionConfig(
        split=args.beaver_split,
        categories=categories or BeaverIngestionConfig().categories,
        sample_size=args.beaver_sample_size,
    )


def cache_layers(
    layers: Iterable[int],
    core_datasets: Sequence[str],
    beaver_config: BeaverIngestionConfig,
    batch_size: int,
    val_fraction: float,
    seed: int,
    force: bool,
) -> None:
    slug = beaver_mix_slug(beaver_config, core_datasets)
    layers = sorted(set(int(layer) for layer in layers))

    pending: list[int] = []
    for layer in layers:
        cache_path = layer_cache_path(slug, layer)
        if cache_path.exists() and not force:
            print(f"[llama-beaver-cache] {slug} layer {layer} already cached; skipping.")
        else:
            pending.append(layer)

    if not pending:
        print("[llama-beaver-cache] Nothing to do.")
        return

    train_split, val_split = split_llama_beaver_mix(
        dataset_names=core_datasets,
        beaver_config=beaver_config,
        val_fraction=val_fraction,
        seed=seed,
    )

    tokenizer, model, _device, _dtype = init_model(MODEL_NAME, seed=seed)

    description = ", ".join(core_datasets)
    print(
        f"[llama-beaver-cache] Extracting layers {pending} | core=[{description}] | "
        f"beaver={beaver_config.split} categories={list(beaver_config.categories)}"
    )

    train_outputs = extract_layerwise_detection_activations(
        train_split.dialogues,
        train_split.labels,
        model,
        tokenizer,
        pending,
        batch_size,
        desc="LlamaBeaver[train]",
    )
    val_outputs = extract_layerwise_detection_activations(
        val_split.dialogues,
        val_split.labels,
        model,
        tokenizer,
        pending,
        batch_size,
        desc="LlamaBeaver[val]",
    )

    for layer in pending:
        cache_path = layer_cache_path(slug, layer)
        train_payload = train_outputs[layer]
        val_payload = val_outputs[layer]

        payload = {
            "dataset": slug,
            "layer_index": layer,
            "train": train_payload,
            "val": val_payload,
            "meta": {
                "model_name": MODEL_NAME,
                "core_datasets": list(core_datasets),
                "beaver_split": beaver_config.split,
                "beaver_categories": list(beaver_config.categories),
                "beaver_sample_size": beaver_config.sample_size,
                "val_fraction": val_fraction,
                "seed": seed,
                "train_dialogues": train_split.size,
                "val_dialogues": val_split.size,
            },
        }

        torch.save(payload, cache_path)
        print(f"[llama-beaver-cache] Saved -> {cache_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache activations for the Llama + Beaver negative dataset mix.")
    parser.add_argument("--layer", type=int, help="Single layer to cache.")
    parser.add_argument("--layers", type=int, nargs="+", help="Explicit list of layers to cache.")
    parser.add_argument("--layer-start", type=int, help="Start of range-style layer sweep.")
    parser.add_argument("--layer-step", type=int, default=LAYER_STEP, help="Step size for range-style sweep.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for forward passes.")
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION, help="Validation split proportion.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for shuffling and sampling.")
    parser.add_argument("--core-datasets", nargs="+", default=list(DEFAULT_CORE_DATASETS), help="Base Apollo datasets.")
    parser.add_argument(
        "--beaver-split",
        default=BeaverIngestionConfig().split,
        help="BeaverTails split to use (default: 30k_test).",
    )
    parser.add_argument(
        "--beaver-category",
        action="append",
        help="Category to include from BeaverTails (repeatable). Use 'no_category' to include uncategorized rows.",
    )
    parser.add_argument(
        "--beaver-sample-size",
        type=int,
        default=None,
        help="Optional sample size for BeaverTails ingestion.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing cached files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = _resolve_layers(args.layer, args.layers, args.layer_start, args.layer_step)
    core_datasets = canonicalize_dataset_names(args.core_datasets)
    beaver_config = _build_beaver_config(args)
    cache_layers(
        layers=layers,
        core_datasets=core_datasets,
        beaver_config=beaver_config,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
