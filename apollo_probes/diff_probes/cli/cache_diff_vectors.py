from __future__ import annotations

import argparse

from ..cache import cache_diff_vectors
from ..config import DEFAULT_LAYER_INDEX, TRAIN_DATASETS
from ...config import DEFAULT_BATCH_SIZE, RANDOM_SEED, VAL_FRACTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache difference-probe activations for deception prompts.")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=list(TRAIN_DATASETS),
        help="Training dataset name(s); supply multiple entries or a '+' separated slug.",
    )
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER_INDEX, help="Layer index to cache.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for forward passes.")
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION, help="Validation split fraction.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for shuffling dialogues.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_arg = args.dataset
    if len(dataset_arg) == 1 and "+" in dataset_arg[0]:
        dataset_arg = [part for part in dataset_arg[0].split("+") if part]
    cache_diff_vectors(
        dataset_args=dataset_arg,
        layer=args.layer,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
