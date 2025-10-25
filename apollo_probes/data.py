from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from .config import VAL_FRACTION, RANDOM_SEED, DEFAULT_DATASET

REPO_ROOT = Path("/workspace/jake/deception-detection")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deception_detection.repository import DatasetRepository  # type: ignore
from deception_detection.types import Dialogue, Label  # type: ignore


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


def load_dataset_splits(
    dataset_name: str = DEFAULT_DATASET,
    val_fraction: float = VAL_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[DialogueSplit, DialogueSplit]:
    repository = DatasetRepository()
    dataset = repository.get(
        dataset_name,
        model="prewritten",
        trim_reasoning=True,
        shuffle_upon_init=True,
    )

    dialogues, labels = filter_binary(dataset.dialogues, dataset.labels)
    total = len(dialogues)
    if total == 0:
        raise ValueError(f"No binary-labeled dialogues available for dataset {dataset_name}")

    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    dialogues = [dialogues[i] for i in indices]
    labels = [labels[i] for i in indices]

    num_val = max(1, int(total * val_fraction))
    train_dialogues = dialogues[:-num_val]
    train_labels = labels[:-num_val]
    val_dialogues = dialogues[-num_val:]
    val_labels = labels[-num_val:]

    train_split = DialogueSplit(train_dialogues, train_labels)
    val_split = DialogueSplit(val_dialogues, val_labels)
    return train_split, val_split
