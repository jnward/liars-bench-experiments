from __future__ import annotations

import argparse

from ..config import DEFAULT_LAYER_INDEX
from ..evaluate import evaluate_diff_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained diff probe on Liars Bench datasets.")
    parser.add_argument("dataset", help="Dataset slug used during training (e.g., repe_honesty__plain+roleplaying__plain)")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["user", "assistant", "combined", "all"],
        default=["all"],
        help="Probe variants to evaluate. Use 'all' to run user, assistant, and combined together.",
    )
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER_INDEX, help="Layer index of the trained probe.")
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
    parser.add_argument(
        "--question",
        type=str,
        help="Custom question to append before the A/B choices (defaults to deception question).",
    )
    parser.add_argument(
        "--prompt-name",
        type=str,
        help="Optional name/slug for the custom question (used for output directories).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_slug = args.dataset
    results = evaluate_diff_probe(
        dataset_slug=dataset_slug,
        variants=args.variants,
        layer=args.layer,
        dataset_filters=args.datasets,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        seed=args.seed,
        chunk_size=args.chunk_size,
        prompt_name=args.prompt_name,
        prompt_text=args.question,
    )
    evaluated = ", ".join(results.get("evaluated_variants", []))
    prompt_info = results.get("prompt")
    prompt_desc = "default" if not prompt_info else prompt_info.get("slug") or prompt_info.get("name") or "custom"
    print(f"[diff-eval] Summary saved to {results['summary_path']} (prompt={prompt_desc}, variants: {evaluated})")


if __name__ == "__main__":
    main()
