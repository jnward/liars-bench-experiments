from __future__ import annotations

import argparse

from ..config import DEFAULT_LAYER_INDEX, TRAIN_DATASETS
from ..train import train_diff_probe
from ...data import dataset_slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a difference-based deception probe.")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=list(TRAIN_DATASETS),
        help="Dataset name(s) used when caching diff vectors.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER_INDEX,
        help="Layer index used in the cached diff vectors.",
    )
    parser.add_argument(
        "--variant",
        choices=["user", "assistant", "combined"],
        default="user",
        help="Feature variant to train on.",
    )
    parser.add_argument("--reg-strength", type=float, default=1.0, help="Logistic regression C parameter.")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum solver iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for logistic regression.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = args.dataset
    if len(dataset_names) == 1 and "+" in dataset_names[0]:
        dataset_names = [part for part in dataset_names[0].split("+") if part]
    slug = dataset_slug(dataset_names)
    metrics = train_diff_probe(
        dataset_slug=slug,
        variant=args.variant,
        layer=args.layer,
        reg_strength=args.reg_strength,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    print(f"[diff-train] Trained probe saved to {metrics['probe_path']}")
    print(f"[diff-train] Metrics stored at {metrics['results_path']}")


if __name__ == "__main__":
    main()

