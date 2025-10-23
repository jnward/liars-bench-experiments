from __future__ import annotations

import argparse
from pathlib import Path

from . import DIFF_CACHE_DIR, DIFF_PLOTS_DIR, DIFF_SCORES_DIR
from .falsehood_diff_cache import cache_falsehood_diff
from .falsehood_diff_score import score_falsehood_diff
from .plot_violin import plot_violin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BeaverTails falsehood difference probe experiment.")
    parser.add_argument("--split", default="30k_test", help="Dataset split to load.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional random sample size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for activation extraction.")
    parser.add_argument("--shard-size", type=int, default=64, help="Number of examples per cache shard.")
    parser.add_argument("--probe-path", type=Path, action="append", help="Additional probe path(s) to evaluate.")
    parser.add_argument("--no-default-probes", action="store_true", help="Disable default diff probes from repository.")
    parser.add_argument("--skip-cache", action="store_true", help="Reuse existing cache manifest without recomputing activations.")
    parser.add_argument("--skip-scoring", action="store_true", help="Skip probe scoring (use existing scores).")
    parser.add_argument("--skip-plot", action="store_true", help="Skip violin plot generation.")
    parser.add_argument("--manifest-path", type=Path, default=None, help="Explicit manifest path (overrides default).")
    parser.add_argument("--scores-dir", type=Path, default=None, help="Directory to save/read probe scores.")
    parser.add_argument("--scores-path", type=Path, action="append", help="Existing score file to plot (repeatable).")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Directory to save plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = args.manifest_path
    if not args.skip_cache:
        manifest_path = cache_falsehood_diff(
            split=args.split,
            sample_size=args.sample_size,
            seed=args.seed,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            output_dir=DIFF_CACHE_DIR,
        )
    elif manifest_path is None:
        manifest_path = DIFF_CACHE_DIR / "beavertails_falsehood_layer22_manifest.json"

    scores_dir = args.scores_dir or DIFF_SCORES_DIR
    plot_dir = args.plot_dir or DIFF_PLOTS_DIR

    if not args.skip_scoring:
        outputs = score_falsehood_diff(
            manifest_path=manifest_path,
            probe_paths=args.probe_path,
            output_dir=scores_dir,
            include_defaults=not args.no_default_probes,
        )
        score_paths = list(outputs.values())
    else:
        if args.scores_path:
            score_paths = list(args.scores_path)
        else:
            score_paths = sorted(scores_dir.glob("*_scores.jsonl"))

    if not args.skip_plot:
        if not score_paths:
            print("No falsehood diff score files available for plotting.")
            return
        for path in score_paths:
            output = plot_violin(
                scores_path=path,
                output_dir=plot_dir,
            )
            print(f"Wrote falsehood diff violin -> {output}")


if __name__ == "__main__":
    main()
