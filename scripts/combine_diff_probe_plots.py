"""Generate combined ROC and violin plots for diff probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from apollo_probes.diff_probes.plotting import generate_combined_plots

DEFAULT_DATASET_SLUG = "repe_honesty__plain+roleplaying__plain"
DEFAULT_LAYER = 34


def load_summary(results_dir: Path) -> dict:
    summary_path = results_dir / "multi_variant_evaluation.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary JSON not found at {summary_path}. Run the multi-variant evaluation first.")
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine diff-probe plots across datasets for each variant.")
    parser.add_argument("dataset", nargs="?", default=DEFAULT_DATASET_SLUG, help="Training dataset slug.")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER, help="Layer index used for evaluation.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("apollo_probes/results/diff_probes"),
        help="Root directory containing diff-probe results.",
    )
    parser.add_argument(
        "--prompt-slug",
        type=str,
        help="Optional prompt slug; if provided, plots are read from prompts/<slug>/ under the layer directory.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Specific variants to process (default: all variants present in the summary).",
    )
    args = parser.parse_args()

    results_dir = args.results_root / args.dataset / f"layer{args.layer:02d}"
    if args.prompt_slug:
        results_dir = results_dir / "prompts" / args.prompt_slug
    summary = load_summary(results_dir)
    variants = args.variants or list(summary.get("variants", {}).keys())
    if not variants:
        raise ValueError("No variants found in the summary; run evaluation first.")

    generated_paths = generate_combined_plots(results_dir, summary, variants)
    if generated_paths:
        print("[combine] Generated:")
        for path in generated_paths:
            print(f"  - {path}")
    else:
        print("[combine] No plots generated (missing score data?).")


if __name__ == "__main__":
    main()
