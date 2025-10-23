from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from . import PLOTS_DIR, SCORES_DIR

LAYER_INDEX = 22


def _read_scores(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fin:
        return [json.loads(line) for line in fin]


def _expand_labels(categories: dict | None) -> list[str]:
    if not categories:
        return ["no_category"]
    labels = [str(name) for name, flag in categories.items() if flag]
    return labels or ["no_category"]


def build_category_scores(records: list[dict]) -> dict[str, list[float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for record in records:
        score = float(record["score"])
        labels = _expand_labels(record.get("categories"))
        for label in labels:
            buckets[label].append(score)
    return buckets


def plot_violin(
    scores_path: Path,
    output_dir: Path,
    title: str | None = None,
) -> Path:
    records = _read_scores(scores_path)
    if not records:
        raise ValueError(f"No records found in {scores_path}")
    probe_name = str(records[0].get("probe") or scores_path.stem)

    buckets = build_category_scores(records)
    if not buckets:
        raise ValueError(f"No category scores found in {scores_path}")

    labels = sorted(buckets.keys())
    data = [buckets[label] for label in labels]

    all_scores = [score for bucket in data for score in bucket]
    global_min = min(all_scores)
    global_max = max(all_scores)
    span = max(global_max - global_min, 1e-6)
    margin = max(0.05 * span, 1e-3)

    plt.figure(figsize=(max(10, len(labels) * 0.7), 6))
    parts = plt.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.7)

    if global_min <= 0.0 <= global_max:
        plt.axhline(0.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.ylim(global_min - margin, global_max + margin)
    plt.xticks(
        ticks=range(1, len(labels) + 1),
        labels=[f"{label}\n(n={len(buckets[label])})" for label in labels],
        rotation=45,
        ha="right",
    )
    plt.ylabel("Probe score (avg over assistant tokens)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    if title is None:
        title = f"{probe_name} Probe Scores by Category (Layer {LAYER_INDEX})"
    plt.title(title)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{probe_name}_layer{LAYER_INDEX:02d}_category_violin.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    summary_path = output_path.with_suffix(".json")
    summary = {
        "scores_path": str(scores_path),
        "layer_index": LAYER_INDEX,
        "probe": probe_name,
        "categories": {
            label: {
                "count": len(buckets[label]),
                "mean": float(sum(buckets[label]) / max(len(buckets[label]), 1)),
            }
            for label in labels
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot violin distributions of probe scores by BeaverTails category.")
    parser.add_argument(
        "--scores-path",
        type=Path,
        nargs="+",
        default=None,
        help="One or more JSONL score files to visualize. Defaults to all files in beaver_tails/scores.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOTS_DIR,
        help="Directory to write violin plots.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom title. When multiple scores are provided this applies to all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.scores_path is None:
        scores = sorted(SCORES_DIR.glob("*_scores.jsonl"))
    else:
        scores = list(args.scores_path)

    if not scores:
        raise SystemExit("No score files found to plot.")

    for scores_path in scores:
        output = plot_violin(
            scores_path=scores_path,
            output_dir=args.output_dir,
            title=args.title,
        )
        print(f"Wrote violin plot -> {output}")


if __name__ == "__main__":
    main()
