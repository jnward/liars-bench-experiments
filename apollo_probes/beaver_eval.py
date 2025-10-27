from __future__ import annotations

from typing import List, Sequence, Tuple

from beaver_tails.data import DEFAULT_SPLIT as BEAVER_DEFAULT_SPLIT, load_beaver_samples
from beaver_tails.schema import BeaverSample

from deception_detection.types import Dialogue, Message

BEAVER_POSITIVE_CATEGORIES: Tuple[str, ...] = ("hate_speech", "drug_abuse", "violence")
BEAVER_POSITIVE_SET = set(BEAVER_POSITIVE_CATEGORIES)
BEAVER_NEGATIVE_TAG = "no_category"


def beaver_slug(split: str, sample_size: int | None, seed: int) -> str:
    slug = f"beaver_tails_{split}"
    if sample_size is not None:
        slug += f"_n{sample_size}"
    slug += f"_seed{seed}"
    return slug


def _normalize_category_name(raw_name: str) -> str:
    return raw_name.split(",", 1)[0].strip()


def _select_category_tags(sample: BeaverSample) -> tuple[bool, list[str]]:
    positive_tags: list[str] = []
    seen: set[str] = set()
    for raw_name, flag in sample.categories.items():
        if not flag:
            continue
        normalized = _normalize_category_name(raw_name)
        if normalized in BEAVER_POSITIVE_SET and normalized not in seen:
            positive_tags.append(normalized)
            seen.add(normalized)
    if positive_tags:
        return True, positive_tags
    has_any_category = any(sample.categories.values())
    if has_any_category:
        return False, []
    return False, [BEAVER_NEGATIVE_TAG]


def build_beaver_dialogues(
    split: str = BEAVER_DEFAULT_SPLIT,
    sample_size: int | None = None,
    seed: int = 42,
) -> tuple[list[Dialogue], list[int], list[list[str]], list[str]]:
    _dataset, samples = load_beaver_samples(split=split, sample_size=sample_size, seed=seed)
    dialogues: list[Dialogue] = []
    labels: list[int] = []
    categories: list[list[str]] = []
    row_ids: list[str] = []

    for sample in samples:
        is_positive, tags = _select_category_tags(sample)
        if not is_positive and tags == []:
            # sample belongs to other harmful categories we are ignoring
            continue
        if is_positive:
            label = 1
        else:
            label = 0
        dialogue = [
            Message(role="user", content=sample.prompt, detect=False),
            Message(role="assistant", content=sample.response, detect=True),
        ]
        dialogues.append(dialogue)
        labels.append(label)
        categories.append(tags)
        row_ids.append(sample.row_id)

    return dialogues, labels, categories, row_ids
