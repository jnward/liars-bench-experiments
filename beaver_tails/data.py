from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from datasets import Dataset, load_dataset

from .schema import BeaverSample, CategoryMap


DATASET_ID = "PKU-Alignment/BeaverTails"
DEFAULT_SPLIT = "30k_test"


def _normalize_categories(raw: dict | None) -> CategoryMap:
    if raw is None:
        return {}
    return {str(key): bool(value) for key, value in raw.items()}


def _row_identifier(row: dict, index: int) -> str:
    for key in ("id", "uid", "example_id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return f"row-{index}"


def load_beaver_samples(
    split: str = DEFAULT_SPLIT,
    sample_size: int | None = None,
    seed: int = 42,
) -> tuple[Dataset, List[BeaverSample]]:
    """
    Load BeaverTails rows and convert them into BeaverSample objects.

    Parameters
    ----------
    split:
        Dataset split to load (default: ``30k_test``).
    sample_size:
        Optional number of examples to subsample (random shuffle with ``seed``).
    seed:
        RNG seed used for deterministic shuffling when ``sample_size`` is set.

    Returns
    -------
    hf_dataset:
        The Hugging Face dataset instance after any sampling.
    samples:
        List of :class:`BeaverSample` objects.
    """

    dataset: Dataset = load_dataset(DATASET_ID, split=split)
    if sample_size is not None:
        sample_size = min(sample_size, len(dataset))
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    samples: List[BeaverSample] = []
    for idx, row in enumerate(dataset):
        prompt = row.get("prompt")
        response = row.get("response")
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise ValueError(f"Row {idx} missing prompt/response text.")
        categories = _normalize_categories(row.get("category"))
        samples.append(
            BeaverSample(
                row_id=_row_identifier(row, idx),
                prompt=prompt,
                response=response,
                categories=categories,
            )
        )

    return dataset, samples
