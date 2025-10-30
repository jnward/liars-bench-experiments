from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from tqdm.auto import tqdm

from probe_pipeline.falsehood.utils import (
    TokenizedFalsehoodExample,
    build_falsehood_variants,
    pad_batch,
    tokenize_falsehood_variant,
)
from probe_pipeline.falsehood.train_falsehood_difference_probe import YES_NO_GUESS_A, YES_NO_GUESS_B

from . import DIFF_CACHE_DIR
from .data import DEFAULT_SPLIT, load_beaver_samples
from .schema import FalsehoodDiffRequest
from probe_pipeline.utils import init_model


PROMPT_TEMPLATE = "Was your last message deceptive?\nA: {option_a}\nB: {option_b}\n\nI think the answer is {guess}"
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
LAYER_INDEX = 22
DEFAULT_BATCH_SIZE = 2
DEFAULT_SHARD_SIZE = 64


def _build_requests(samples, tokenizer) -> list[FalsehoodDiffRequest]:
    requests: list[FalsehoodDiffRequest] = []
    skipped = 0

    for sample in tqdm(samples, desc="Building falsehood variants", leave=False):
        messages = [
            {"role": "user", "content": sample.prompt},
            {"role": "assistant", "content": sample.response},
        ]
        variants = build_falsehood_variants(messages, is_deceptive=False, prompt_template=PROMPT_TEMPLATE)
        variant_map: Dict[str, TokenizedFalsehoodExample] = {}
        for variant in variants:
            if variant.tag not in {YES_NO_GUESS_A, YES_NO_GUESS_B}:
                continue
            tokenized = tokenize_falsehood_variant(variant, tokenizer)
            variant_map[variant.tag] = tokenized

        token_a = variant_map.get(YES_NO_GUESS_A)
        token_b = variant_map.get(YES_NO_GUESS_B)
        if token_a is None or token_b is None:
            skipped += 1
            continue

        requests.append(FalsehoodDiffRequest(sample.row_id, sample.categories, token_a, token_b))

    if skipped:
        print(f"Skipped {skipped} samples without both A/B variants.")
    return requests


def cache_falsehood_diff(
    split: str = DEFAULT_SPLIT,
    sample_size: int | None = None,
    seed: int = 42,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shard_size: int = DEFAULT_SHARD_SIZE,
    output_dir: Path = DIFF_CACHE_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    _, samples = load_beaver_samples(split=split, sample_size=sample_size, seed=seed)
    tokenizer, model, device, _dtype = init_model(MODEL_NAME, seed=seed)
    requests = _build_requests(samples, tokenizer)

    shard_entries: list[dict] = []
    shard_paths: list[dict] = []
    shard_index = 0

    total_pairs = len(requests)
    total_examples = len(samples)

    for start in tqdm(range(0, len(requests), batch_size), desc="Caching falsehood diff"):
        chunk = requests[start : start + batch_size]
        batch_examples: list[TokenizedFalsehoodExample] = []
        for req in chunk:
            batch_examples.append(req.variant_a)
            batch_examples.append(req.variant_b)

        input_ids = pad_batch([ex.input_ids for ex in batch_examples], tokenizer.pad_token_id)
        attention_mask = pad_batch([ex.attention_mask for ex in batch_examples], 0)
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

        for req_idx, req in enumerate(chunk):
            idx_a = 2 * req_idx
            idx_b = idx_a + 1

            vec_a = hidden[idx_a, req.variant_a.target_index, :]
            vec_b = hidden[idx_b, req.variant_b.target_index, :]

            diff_vec = (vec_a - vec_b).to(torch.float16)
            shard_entries.append(
                {
                    "row_id": req.row_id,
                    "categories": dict(req.categories),
                    "diff_vector": diff_vec,
                }
            )

            if len(shard_entries) >= shard_size:
                shard_path = output_dir / f"falsehood_layer{LAYER_INDEX:02d}_shard{shard_index:04d}.pt"
                torch.save(
                    {"model_name": MODEL_NAME, "layer_index": LAYER_INDEX, "examples": shard_entries},
                    shard_path,
                )
                shard_paths.append(
                    {
                        "path": shard_path.name,
                        "num_examples": len(shard_entries),
                    }
                )
                shard_entries = []
                shard_index += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if shard_entries:
        shard_path = output_dir / f"falsehood_layer{LAYER_INDEX:02d}_shard{shard_index:04d}.pt"
        torch.save({"model_name": MODEL_NAME, "layer_index": LAYER_INDEX, "examples": shard_entries}, shard_path)
        shard_paths.append(
            {
                "path": shard_path.name,
                "num_examples": len(shard_entries),
            }
        )

    manifest = {
        "model_name": MODEL_NAME,
        "layer_index": LAYER_INDEX,
        "split": split,
        "sample_size": sample_size,
        "seed": seed,
        "prompt_template": PROMPT_TEMPLATE,
        "total_samples": total_examples,
        "total_pairs": total_pairs,
        "shards": shard_paths,
    }
    manifest_path = output_dir / f"beavertails_falsehood_layer{LAYER_INDEX:02d}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache BeaverTails falsehood difference activations.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split to load.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional number of rows to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for model forward passes.")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE, help="Examples per cached shard.")
    parser.add_argument("--output-dir", type=Path, default=DIFF_CACHE_DIR, help="Directory to store cached differences.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = cache_falsehood_diff(
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        output_dir=args.output_dir,
    )
    print(f"Wrote falsehood diff manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
