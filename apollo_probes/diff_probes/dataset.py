from __future__ import annotations

from typing import Generator, Iterable, Tuple

from deception_detection.types import Dialogue  # type: ignore

from ..config import RANDOM_SEED
from ..data import canonicalize_dataset_names, load_full_dataset
from .types import DialogueInfo


def iter_dialogues(
    dataset_names: Iterable[str],
    seed: int = RANDOM_SEED,
) -> Generator[Tuple[DialogueInfo, Dialogue], None, None]:
    """
    Yield (info, dialogue) pairs for each dataset requested.

    The dialogue indices correspond to the shuffled order produced by the dataset loader.
    """
    canonical = canonicalize_dataset_names(list(dataset_names))
    for dataset_name in canonical:
        split = load_full_dataset([dataset_name], seed=seed)
        for idx, (dialogue, label) in enumerate(zip(split.dialogues, split.labels, strict=False)):
            info = DialogueInfo(
                dataset=dataset_name,
                dialogue_index=idx,
                label=int(label),
                meta={},
            )
            yield info, dialogue

