from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

from beaver_tails.data import DEFAULT_SPLIT as BEAVER_DEFAULT_SPLIT, load_beaver_samples
from beaver_tails.schema import BeaverSample

from deception_detection.repository import DatasetRepository  # type: ignore
from deception_detection.types import Dialogue, Message  # type: ignore

from .config import RANDOM_SEED, VAL_FRACTION
from .data import AVAILABLE_DATASETS, DialogueSplit, _resolve_dataset_model, filter_binary


DEFAULT_BEAVER_CATEGORIES: tuple[str, ...] = (
    "no_category",
    "violence",
    "drug_abuse",
    "hate_speech",
)

DEFAULT_CORE_DATASETS: tuple[str, ...] = ("repe_honesty__plain", "roleplaying__plain")


@dataclass(frozen=True)
class BeaverIngestionConfig:
    split: str = BEAVER_DEFAULT_SPLIT
    categories: Sequence[str] = DEFAULT_BEAVER_CATEGORIES
    sample_size: int | None = None


def _normalize_category_name(raw_name: str) -> str:
    return raw_name.split(",", 1)[0].strip()


_TOKEN_SANITIZER = re.compile(r"[^a-z0-9]+")


def _slug_token(token: str) -> str:
    cleaned = _TOKEN_SANITIZER.sub("-", token.lower()).strip("-")
    return cleaned or "nc"


def _sample_matches_category(sample: BeaverSample, category: str) -> bool:
    if category == "no_category":
        return not any(sample.categories.values())

    target = _normalize_category_name(category)
    for raw_name, flag in sample.categories.items():
        if not flag:
            continue
        normalized = _normalize_category_name(raw_name)
        if normalized == target or raw_name.strip() == category:
            return True
    return False


def _load_beaver_negatives(
    config: BeaverIngestionConfig,
    seed: int,
) -> tuple[list[Dialogue], list[int]]:
    _dataset, samples = load_beaver_samples(
        split=config.split,
        sample_size=config.sample_size,
        seed=seed,
    )
    target_categories = tuple(config.categories)
    dialogues: list[Dialogue] = []
    labels: list[int] = []

    for sample in samples:
        if not target_categories:
            include = True
        else:
            include = any(_sample_matches_category(sample, category) for category in target_categories)
        if not include:
            continue

        dialogue = [
            Message(role="user", content=sample.prompt, detect=False),
            Message(role="assistant", content=sample.response, detect=True),
        ]
        dialogues.append(dialogue)
        labels.append(0)

    return dialogues, labels


def _load_llama_datasets(dataset_names: Sequence[str], seed: int) -> DialogueSplit:
    canonical = sorted(set(dataset_names))
    repository = DatasetRepository()

    combined_dialogues: list[Dialogue] = []
    combined_labels: list[int] = []

    for name in canonical:
        spec = AVAILABLE_DATASETS[name]
        model_name = _resolve_dataset_model(spec)
        try:
            dataset = repository.get(
                spec["partial_id"],
                model=model_name,
                trim_reasoning=True,
                shuffle_upon_init=True,
            )
        except KeyError as exc:
            fallback_model = spec.get("default_model")
            if fallback_model and fallback_model != model_name:
                dataset = repository.get(
                    spec["partial_id"],
                    model=fallback_model,
                    trim_reasoning=True,
                    shuffle_upon_init=True,
                )
            else:
                raise exc
        dialogues, labels = filter_binary(dataset.dialogues, dataset.labels)
        combined_dialogues.extend(dialogues)
        combined_labels.extend(labels)

    if not combined_dialogues:
        raise ValueError(f"No dialogues found for datasets={canonical}")

    return DialogueSplit(combined_dialogues, combined_labels)


def build_llama_beaver_mix(
    dataset_names: Sequence[str],
    beaver_config: BeaverIngestionConfig | None = None,
    seed: int = RANDOM_SEED,
) -> DialogueSplit:
    core_split = _load_llama_datasets(dataset_names, seed)

    if beaver_config is None:
        beaver_config = BeaverIngestionConfig()

    beaver_dialogues, beaver_labels = _load_beaver_negatives(beaver_config, seed)

    dialogues = list(core_split.dialogues)
    labels = list(core_split.labels)

    dialogues.extend(beaver_dialogues)
    labels.extend(beaver_labels)

    if not dialogues:
        raise ValueError("Combined dataset is empty.")

    combined = DialogueSplit(dialogues, labels)
    _ = seed  # Reserved for future deterministic shuffling hooks.
    return combined


def split_llama_beaver_mix(
    dataset_names: Sequence[str],
    beaver_config: BeaverIngestionConfig | None = None,
    val_fraction: float = VAL_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[DialogueSplit, DialogueSplit]:
    combined = build_llama_beaver_mix(dataset_names, beaver_config=beaver_config, seed=seed)
    total = combined.size
    if total == 0:
        raise ValueError("Combined dataset is empty.")

    # reuse _gather_datasets shuffle behavior: deterministic shuffle based on seed
    # by calling _gather_datasets on same inputs (without Beaver) for ordering baseline.
    # Instead, perform local shuffle using seed for reproducibility.
    import random

    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    shuffled_dialogues = [combined.dialogues[i] for i in indices]
    shuffled_labels = [combined.labels[i] for i in indices]

    num_val = max(1, int(total * val_fraction))
    train_dialogues = shuffled_dialogues[:-num_val]
    train_labels = shuffled_labels[:-num_val]
    val_dialogues = shuffled_dialogues[-num_val:]
    val_labels = shuffled_labels[-num_val:]

    train_split = DialogueSplit(train_dialogues, train_labels)
    val_split = DialogueSplit(val_dialogues, val_labels)
    return train_split, val_split


def beaver_mix_slug(
    config: BeaverIngestionConfig | None = None,
    core_datasets: Sequence[str] | None = None,
) -> str:
    if config is None:
        config = BeaverIngestionConfig()
    parts = ["llama_beaver_mix", _slug_token(config.split)]
    if core_datasets:
        core_tokens = [_slug_token(name) for name in core_datasets]
        parts.append("core-" + "-".join(sorted(core_tokens)))
    if config.sample_size is not None:
        parts.append(f"n{config.sample_size}")
    if config.categories:
        category_tokens = [_slug_token(cat) for cat in config.categories]
        parts.append("-".join(sorted(category_tokens)))
    return "_".join(parts)


__all__ = [
    "BeaverIngestionConfig",
    "DEFAULT_BEAVER_CATEGORIES",
    "DEFAULT_CORE_DATASETS",
    "build_llama_beaver_mix",
    "beaver_mix_slug",
    "split_llama_beaver_mix",
]
