from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from tqdm.auto import tqdm

from probe_pipeline.utils import init_model, pad_batch

from . import CACHE_DIR
from .data import DEFAULT_SPLIT, load_beaver_samples
from .tokenization import tokenize_samples


MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
LAYER_INDEX = 22
DEFAULT_BATCH_SIZE = 2
DEFAULT_SHARD_SIZE = 64


def _collate_batch(tokenized, pad_token_id: int) -> tuple[torch.LongTensor, torch.LongTensor, List[Sequence[int]]]:
    input_ids = pad_batch([item.input_ids for item in tokenized], pad_token_id)
    attention_mask = pad_batch([item.attention_mask for item in tokenized], 0)
    assistant_positions = [list(item.assistant_token_indices) for item in tokenized]
    return input_ids, attention_mask, assistant_positions


def cache_activations(
    split: str = DEFAULT_SPLIT,
    sample_size: int | None = None,
    seed: int = 42,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shard_size: int = DEFAULT_SHARD_SIZE,
    output_dir: Path = CACHE_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, samples = load_beaver_samples(split=split, sample_size=sample_size, seed=seed)
    tokenizer, model, device, _dtype = init_model(MODEL_NAME, seed=seed)
    tokenized = tokenize_samples(samples, tokenizer)

    total_tokens = 0
    shard_entries: list[dict] = []
    current_shard_tokens = 0
    shard_paths: list[dict] = []
    shard_index = 0

    for start in tqdm(range(0, len(tokenized), batch_size), desc="Caching activations"):
        batch = tokenized[start : start + batch_size]
        input_ids, attention_mask, assistant_positions = _collate_batch(batch, tokenizer.pad_token_id)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden = outputs.hidden_states[LAYER_INDEX].to(torch.float32).cpu()

        for idx, sample in enumerate(batch):
            positions = assistant_positions[idx]
            if not positions:
                continue
            hidden_slice = hidden[idx, positions, :].to(torch.float16)
            token_slice = sample.input_ids[positions].clone()

            total_tokens += hidden_slice.size(0)
            current_shard_tokens += hidden_slice.size(0)

            shard_entries.append(
                {
                    "row_id": sample.row_id,
                    "categories": dict(sample.categories),
                    "token_ids": token_slice,
                    "hidden": hidden_slice,
                    "num_tokens": int(hidden_slice.size(0)),
                }
            )

            if len(shard_entries) >= shard_size:
                shard_path = output_dir / f"layer{LAYER_INDEX:02d}_shard{shard_index:04d}.pt"
                torch.save(
                    {"model_name": MODEL_NAME, "layer_index": LAYER_INDEX, "examples": shard_entries},
                    shard_path,
                )
                shard_paths.append(
                    {
                        "path": shard_path.name,
                        "num_examples": len(shard_entries),
                        "total_tokens": current_shard_tokens,
                    }
                )
                shard_entries = []
                current_shard_tokens = 0
                shard_index += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if shard_entries:
        shard_path = output_dir / f"layer{LAYER_INDEX:02d}_shard{shard_index:04d}.pt"
        torch.save({"model_name": MODEL_NAME, "layer_index": LAYER_INDEX, "examples": shard_entries}, shard_path)
        shard_paths.append(
            {
                "path": shard_path.name,
                "num_examples": len(shard_entries),
                "total_tokens": current_shard_tokens,
            }
        )
        current_shard_tokens = 0

    manifest = {
        "model_name": MODEL_NAME,
        "layer_index": LAYER_INDEX,
        "split": split,
        "sample_size": sample_size,
        "seed": seed,
        "total_examples": len(samples),
        "total_tokens": total_tokens,
        "shards": shard_paths,
    }
    manifest_path = output_dir / f"beavertails_layer{LAYER_INDEX:02d}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache BeaverTails assistant activations.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split to load.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional number of rows to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for model forward passes.")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE, help="Number of examples per cached shard.")
    parser.add_argument("--output-dir", type=Path, default=CACHE_DIR, help="Directory to store cached activations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = cache_activations(
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        output_dir=args.output_dir,
    )
    print(f"Wrote cache manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
