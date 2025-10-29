from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..config import DEFAULT_LAYER_INDEX
from ..evaluate import evaluate_diff_probe


def load_prompt_config(config_path: Path) -> List[Dict[str, Any]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Prompt config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Config file must contain a non-empty 'prompts' list.")
    normalized: List[Dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            raise ValueError(f"Prompt entry #{idx} is not an object: {prompt!r}")
        question = prompt.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Prompt entry #{idx} missing valid 'question' text: {prompt!r}")
        name = prompt.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError(f"Prompt entry #{idx} has non-string name: {prompt!r}")
        normalized.append({
            "name": name,
            "question": question,
        })
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate diff probes across multiple prompt questions.")
    parser.add_argument("dataset", help="Training dataset slug (e.g., repe_honesty__plain+roleplaying__plain).")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config containing prompt list.")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["user", "assistant", "combined", "all"],
        default=["all"],
        help="Probe variants to evaluate for each prompt.",
    )
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER_INDEX, help="Layer index of the trained probes.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Specific Liars Bench configs to evaluate. Defaults to all configured splits.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for forward passes.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional cap on examples per dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Number of dialogues to process before flushing intermediate tensors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_prompt_config(args.config)

    total = len(prompts)
    for idx, prompt in enumerate(prompts, start=1):
        name = prompt.get("name")
        question = prompt["question"]
        display_name = name or (question[:60] + ("…" if len(question) > 60 else ""))
        print(f"[prompt {idx}/{total}] Evaluating '{display_name}'")
        result = evaluate_diff_probe(
            dataset_slug=args.dataset,
            variants=args.variants,
            layer=args.layer,
            dataset_filters=args.datasets,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            seed=args.seed,
            chunk_size=args.chunk_size,
            prompt_name=name,
            prompt_text=question,
        )
        print(
            f"  -> Saved summary to {result['summary_path']} (variants: {', '.join(result.get('evaluated_variants', []))})"
        )


if __name__ == "__main__":
    main()
