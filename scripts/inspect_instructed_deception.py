from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probe_pipeline.utils import load_filtered_dataset

DATASET_ID = "Cadenza-Labs/liars-bench"
CONFIG = "instructed-deception"
SPLIT = "test"
FILTER_MODEL = "llama-v3.3-70b-instruct"


def main() -> None:
    dataset = load_filtered_dataset(
        DATASET_ID,
        CONFIG,
        SPLIT,
        FILTER_MODEL,
        sample_size=None,
        seed=42,
    )
    print(f"Total rows: {len(dataset)}")

    deceptive_counts = Counter()
    correct_option_counts = Counter()
    instructed_counts = Counter()

    for row in dataset:
        deceptive_counts[row.get("deceptive", False)] += 1
        meta = row.get("meta")
        if isinstance(meta, str):
            try:
                meta = ast.literal_eval(meta)
            except Exception:
                meta = None
        if isinstance(meta, dict):
            correct_option = meta.get("correct_option")
            instructed = meta.get("instructed_deceptiveness")
            correct_option_counts[correct_option] += 1
            instructed_counts[instructed] += 1

    print("Deceptive label counts:")
    for key, value in sorted(deceptive_counts.items()):
        print(f"  {key}: {value}")

    print("correct_option counts:")
    for key, value in sorted(correct_option_counts.items()):
        print(f"  {key}: {value}")

    print("instructed_deceptiveness counts:")
    for key, value in sorted(instructed_counts.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
