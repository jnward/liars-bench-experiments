from __future__ import annotations

import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from .config import VAL_FRACTION, RANDOM_SEED, DEFAULT_DATASET, APOLLO_CONFIG

REPO_ROOT = Path("/workspace/jake/deception-detection")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deception_detection.repository import DatasetRepository  # type: ignore
from deception_detection.types import Dialogue, Label  # type: ignore


AVAILABLE_DATASETS = {
    "repe_honesty__plain": {"partial_id": "repe_honesty__plain", "model": "prewritten"},
    # Roleplaying data can come from different backbone rollouts.
    "roleplaying__plain": {
        "partial_id": "roleplaying__plain",
        "models": {
            "llama": "llama-70b-3.3",
            "qwen": "qwen-2.5-72b-instruct",
        },
        "default_model": "llama-70b-3.3",
    },
    "got_cities__plain": {"partial_id": "got_cities__plain", "model": "prewritten"},
    "got_larger_than__qa": {"partial_id": "got_larger_than__qa", "model": "prewritten"},
}


def _resolve_dataset_model(spec: dict) -> str:
    model_map = spec.get("models")
    if model_map:
        return model_map.get(APOLLO_CONFIG.key, spec.get("default_model", "prewritten"))
    return spec.get("model", "prewritten")


def canonicalize_dataset_names(dataset_names: Sequence[str] | None) -> list[str]:
    if not dataset_names:
        dataset_names = [DEFAULT_DATASET]
    if len(dataset_names) == 1 and "+" in dataset_names[0]:
        dataset_names = [name for name in dataset_names[0].split("+") if name]
    cleaned: list[str] = []
    for name in dataset_names:
        name = name.strip()
        if name not in AVAILABLE_DATASETS:
            raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(AVAILABLE_DATASETS)}")
        cleaned.append(name)
    return sorted(set(cleaned))


def dataset_slug(dataset_names: Sequence[str]) -> str:
    return "+".join(dataset_names)


def prepare_datasets(dataset_args: Sequence[str] | None) -> tuple[list[str], str]:
    names = canonicalize_dataset_names(dataset_args)
    return names, dataset_slug(names)


@dataclass
class DialogueSplit:
    dialogues: List[Dialogue]
    labels: List[int]

    @property
    def size(self) -> int:
        return len(self.dialogues)


def label_to_int(label: Label) -> int:
    if label == Label.HONEST:
        return 0
    if label == Label.DECEPTIVE:
        return 1
    raise ValueError(f"Unsupported label {label}")


def filter_binary(dialogues: Sequence[Dialogue], labels: Sequence[Label]) -> Tuple[List[Dialogue], List[int]]:
    filtered_dialogues: List[Dialogue] = []
    filtered_labels: List[int] = []
    for dialogue, label in zip(dialogues, labels):
        if label in (Label.HONEST, Label.DECEPTIVE):
            filtered_dialogues.append(dialogue)
            filtered_labels.append(label_to_int(label))
    return filtered_dialogues, filtered_labels


def _gather_datasets(dataset_names: Sequence[str], seed: int) -> DialogueSplit:
    canonical = canonicalize_dataset_names(dataset_names)
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
                warnings.warn(
                    f"Dataset '{spec['partial_id']}' missing for model '{model_name}',"
                    f" falling back to '{fallback_model}'.",
                    RuntimeWarning,
                )
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

    total = len(combined_dialogues)
    if total == 0:
        raise ValueError(f"No binary-labeled dialogues available for datasets {canonical}")

    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    shuffled_dialogues = [combined_dialogues[i] for i in indices]
    shuffled_labels = [combined_labels[i] for i in indices]
    return DialogueSplit(shuffled_dialogues, shuffled_labels)


def load_dataset_splits(
    dataset_names: Sequence[str],
    val_fraction: float = VAL_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[DialogueSplit, DialogueSplit]:
    combined = _gather_datasets(dataset_names, seed)
    total = combined.size

    num_val = max(1, int(total * val_fraction))
    train_dialogues = combined.dialogues[:-num_val]
    train_labels = combined.labels[:-num_val]
    val_dialogues = combined.dialogues[-num_val:]
    val_labels = combined.labels[-num_val:]

    train_split = DialogueSplit(train_dialogues, train_labels)
    val_split = DialogueSplit(val_dialogues, val_labels)
    return train_split, val_split


def load_full_dataset(dataset_names: Sequence[str], seed: int = RANDOM_SEED) -> DialogueSplit:
    return _gather_datasets(dataset_names, seed)
