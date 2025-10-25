from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence

import torch
from transformers import AutoConfig

from probe_pipeline.utils import init_model, load_filtered_dataset

from .activations import extract_layerwise_detection_activations
from .config import DEFAULT_BATCH_SIZE, DEFAULT_LAYER, DEFAULT_LAYER_SWEEP, LAYER_STEP, MODEL_NAME, RANDOM_SEED
from .eval_datasets import EVAL_DATASETS, EvalSpec, liars_bench_specs, slugify_config
from .paths import eval_layer_cache_path

REPO_ROOT = Path("/workspace/jake/deception-detection")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deception_detection.types import Message  # type: ignore


def chunk_rows(dialogues: Sequence, labels: Sequence[int], chunk_size: int):
    for start in range(0, len(dialogues), chunk_size):
        end = start + chunk_size
        yield dialogues[start:end], labels[start:end], start // chunk_size


def build_dialogue(row: dict) -> list[Message]:
    dialogue: list[Message] = []
    messages = row["messages"]
    last_assistant_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx]["role"] == "assistant":
            last_assistant_idx = idx
            break

    for idx, msg in enumerate(messages):
        dialogue.append(
            Message(
                role=msg["role"],
                content=msg["content"],
                detect=(idx == last_assistant_idx),
            )
        )
    return dialogue


def cache_eval_dataset(
    spec: EvalSpec,
    alias: str,
    layers: Sequence[int],
    batch_size: int,
    sample_size: int | None,
    seed: int,
    chunk_size: int,
    force: bool = False,
) -> None:
    layers = sorted(set(int(layer) for layer in layers))
    pending: list[int] = []
    for layer in layers:
        path = eval_layer_cache_path(alias, layer)
        if path.exists() and not force:
            print(f"[eval-cache] {alias} layer {layer} already cached; skipping.")
        else:
            pending.append(layer)

    if not pending:
        print(f"[eval-cache] {alias}: nothing to do.")
        return

    dataset = load_filtered_dataset(
        spec.dataset_id,
        spec.config,
        split=spec.split,
        filter_model_name=spec.filter_model,
        sample_size=sample_size,
        seed=seed,
    )

    dialogues = [build_dialogue(row) for row in dataset]
    labels = [int(row.get("deceptive", 0)) for row in dataset]

    tokenizer, model, device, _dtype = init_model(MODEL_NAME, seed=seed)

    tmp_root = Path(eval_layer_cache_path(alias, pending[0])).parent / f".tmp_eval_{seed}"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    chunk_records: dict[int, list[dict[str, object]]] = {layer: [] for layer in pending}
    layer_hidden_dim: dict[int, int] = {}

    for chunk_dialogues, chunk_labels, chunk_idx in chunk_rows(dialogues, labels, chunk_size):
        outputs = extract_layerwise_detection_activations(
            chunk_dialogues,
            chunk_labels,
            model,
            tokenizer,
            pending,
            batch_size,
            desc=f"Eval[{alias}]#{chunk_idx}",
        )
        for layer, info in outputs.items():
            acts = info["activations"]
            lbls = info["labels"]
            counts = info["counts"]
            dialogue_labels_chunk = info["dialogue_labels"]
            chunk_path = tmp_root / f"layer{layer:02d}_chunk{chunk_idx:04d}.pt"
            torch.save(
                {
                    "activations": acts,
                    "labels": lbls,
                    "counts": counts,
                    "dialogue_labels": dialogue_labels_chunk,
                },
                chunk_path,
            )
            chunk_records[layer].append(
                {
                    "path": chunk_path,
                    "token_count": int(lbls.size(0)),
                    "dialogue_count": len(counts),
                }
            )
            if acts.numel() > 0 and layer_hidden_dim.get(layer) is None:
                layer_hidden_dim[layer] = acts.size(1)

    for layer in pending:
        cache_path = eval_layer_cache_path(alias, layer)
        hidden_dim = layer_hidden_dim.get(layer, getattr(model.config, "hidden_size", 0))
        records = chunk_records[layer]
        total_tokens = sum(r["token_count"] for r in records)
        total_dialogues = sum(r["dialogue_count"] for r in records)
        counts: list[int] = []
        dialogue_labels = torch.empty(total_dialogues, dtype=torch.long)
        if total_tokens == 0 or hidden_dim == 0:
            activations = torch.empty((0, hidden_dim), dtype=torch.float16)
            label_tensor = torch.empty(0, dtype=torch.long)
            dialogue_offset = 0
            for record in records:
                data = torch.load(record["path"], map_location="cpu")
                chunk_counts = data["counts"]
                counts.extend(chunk_counts)
                chunk_dialogue_labels = data["dialogue_labels"].to(torch.long)
                length = len(chunk_counts)
                dialogue_labels[dialogue_offset : dialogue_offset + length] = chunk_dialogue_labels
                dialogue_offset += length
                Path(record["path"]).unlink(missing_ok=True)
        else:
            activations = torch.empty((total_tokens, hidden_dim), dtype=torch.float16)
            label_tensor = torch.empty(total_tokens, dtype=torch.long)
            offset = 0
            dialogue_offset = 0
            for record in records:
                data = torch.load(record["path"], map_location="cpu")
                length = int(record["token_count"])
                if length > 0:
                    activations[offset : offset + length] = data["activations"]
                    label_tensor[offset : offset + length] = data["labels"]
                    offset += length
                chunk_counts = data["counts"]
                counts.extend(chunk_counts)
                chunk_dialogue_labels = data["dialogue_labels"].to(torch.long)
                dialogue_length = len(chunk_counts)
                dialogue_labels[dialogue_offset : dialogue_offset + dialogue_length] = chunk_dialogue_labels
                dialogue_offset += dialogue_length
                Path(record["path"]).unlink(missing_ok=True)

        torch.save(
            {
                "dataset": alias,
                "layer_index": layer,
                "activations": activations,
                "labels": label_tensor,
                "counts": counts,
                "dialogue_labels": dialogue_labels,
                "meta": {
                    "config": spec.config,
                    "count": len(dialogues),
                    "tokens": int(total_tokens),
                    "sample_size": sample_size,
                },
            },
            cache_path,
        )
        print(f"[eval-cache] Saved -> {cache_path}")

    shutil.rmtree(tmp_root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache evaluation activations for Apollo probes.")
    parser.add_argument(
        "--dataset",
        choices=EVAL_DATASETS,
        nargs="+",
        default=EVAL_DATASETS,
        help="Eval configs to cache.",
    )
    parser.add_argument("--layer", type=int, help="Single layer index to cache.")
    parser.add_argument("--layers", type=int, nargs="+", help="Multiple layer indices to cache.")
    parser.add_argument("--layer-start", type=int, help="Start of regular layer spacing.")
    parser.add_argument("--layer-step", type=int, default=LAYER_STEP, help="Spacing for range-based caching.")
    parser.add_argument("--default-sweep", action="store_true", help="Cache default sweep layers.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_layers(args) -> Sequence[int]:
    if args.default_sweep:
        return DEFAULT_LAYER_SWEEP
    if args.layers:
        return args.layers
    if args.layer_start is not None:
        config = AutoConfig.from_pretrained(MODEL_NAME)
        max_layer = config.num_hidden_layers
        step = args.layer_step or LAYER_STEP
        return [layer for layer in range(args.layer_start, max_layer + 1, step)]
    if args.layer is not None:
        return [args.layer]
    return [DEFAULT_LAYER]


def main() -> None:
    args = parse_args()
    layers = resolve_layers(args)
    specs = {spec.config: spec for spec in liars_bench_specs()}
    for config_name in args.dataset:
        spec = specs.get(config_name)
        if spec is None:
            print(f"[eval-cache] Unknown eval config {config_name}; skipping.")
            continue
        alias = slugify_config(config_name)
        cache_eval_dataset(
            spec,
            alias,
            layers=layers,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            seed=args.seed,
            chunk_size=max(1, args.chunk_size),
            force=args.force,
        )


if __name__ == "__main__":
    main()
