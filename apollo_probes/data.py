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
from deception_detection.types import Dialogue, Label, Message  # type: ignore

from probe_pipeline.utils import load_filtered_dataset


AVAILABLE_DATASETS = {
    "repe_honesty__plain": {"partial_id": "repe_honesty__plain", "model": "prewritten"},
    # Roleplaying data comes from llama-3.3 rollouts so we pick up full assistant responses.
    "roleplaying__plain": {"partial_id": "roleplaying__plain", "model": "llama-70b-3.3"},
    "got_cities__plain": {"partial_id": "got_cities__plain", "model": "prewritten"},
    "got_larger_than__qa": {"partial_id": "got_larger_than__qa", "model": "prewritten"},
    # BeaverTails safety datasets (harmful vs harmless)
    "beavertails__hate_speech": {"type": "beaver", "category": "hate_speech", "sample_size": 600},
    "beavertails__drug_abuse": {"type": "beaver", "category": "drug_abuse", "sample_size": 600},
    "beavertails__violence": {"type": "beaver", "category": "violence", "sample_size": 600},
    "beavertails__combined": {"type": "beaver", "category": "combined", "sample_size": 600},
    # Liars Bench datasets (stratified sampling, max 500 per dataset)
    "liars-bench__convincing-game": {"type": "liars_bench", "config": "convincing-game", "sample_size": 500},
    "liars-bench__hpkr": {"type": "liars_bench", "config": "harm-pressure-knowledge-report", "sample_size": 500},
    "liars-bench__instructed-deception": {"type": "liars_bench", "config": "instructed-deception", "sample_size": 500},
    "liars-bench__insider-trading-report": {"type": "liars_bench", "config": "insider-trading/report", "sample_size": 500},
    "liars-bench__insider-trading-confirm": {"type": "liars_bench", "config": "insider-trading/confirmation", "sample_size": 500},
}


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


def _load_beavertails_dataset(category: str, sample_size: int, seed: int) -> DialogueSplit:
    """Load BeaverTails dataset with specified category and sample size.

    Parameters
    ----------
    category : str
        One of: "hate_speech", "drug_abuse", "violence", "combined"
    sample_size : int
        Total number of samples (half harmful, half harmless)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    DialogueSplit
        Combined dialogues and labels (not yet split into train/val)
    """
    from .beaver_eval import build_beaver_dialogues, BEAVER_POSITIVE_CATEGORIES, BEAVER_NEGATIVE_TAG

    # Load all samples from 330k_train split
    all_dialogues, all_labels, all_categories, _ = build_beaver_dialogues(
        split='330k_train',
        sample_size=None,  # Load all
        seed=seed
    )

    # Separate into pure category samples
    TARGET_CATEGORIES = set(BEAVER_POSITIVE_CATEGORIES)

    # Collect pure samples for each category
    pure_samples = {cat: [] for cat in TARGET_CATEGORIES}
    pure_samples['no_category'] = []

    for dialogue, label, tags in zip(all_dialogues, all_labels, all_categories):
        # Pure samples: exactly one tag
        if len(tags) == 1:
            tag = tags[0]
            if tag in TARGET_CATEGORIES:
                pure_samples[tag].append((dialogue, label))
            elif tag == BEAVER_NEGATIVE_TAG:
                pure_samples['no_category'].append((dialogue, label))

    # Sample based on category
    rng = random.Random(seed)
    num_per_class = sample_size // 2

    if category == "combined":
        # 100 from each harmful category + 300 no_category
        num_per_harmful = 100
        harmful_dialogues = []
        harmful_labels = []

        for cat in BEAVER_POSITIVE_CATEGORIES:
            samples = pure_samples[cat]
            rng.shuffle(samples)
            selected = samples[:num_per_harmful]
            for d, l in selected:
                harmful_dialogues.append(d)
                harmful_labels.append(l)

        # 300 no_category
        no_cat_samples = pure_samples['no_category']
        rng.shuffle(no_cat_samples)
        selected = no_cat_samples[:300]
        harmless_dialogues = [d for d, l in selected]
        harmless_labels = [l for d, l in selected]
    else:
        # Single category: num_per_class harmful + num_per_class harmless
        harmful_samples = pure_samples[category]
        rng.shuffle(harmful_samples)
        selected = harmful_samples[:num_per_class]
        harmful_dialogues = [d for d, l in selected]
        harmful_labels = [l for d, l in selected]

        no_cat_samples = pure_samples['no_category']
        rng.shuffle(no_cat_samples)
        selected = no_cat_samples[:num_per_class]
        harmless_dialogues = [d for d, l in selected]
        harmless_labels = [l for d, l in selected]

    # Combine and shuffle
    combined_dialogues = harmful_dialogues + harmless_dialogues
    combined_labels = harmful_labels + harmless_labels

    indices = list(range(len(combined_dialogues)))
    rng.shuffle(indices)
    shuffled_dialogues = [combined_dialogues[i] for i in indices]
    shuffled_labels = [combined_labels[i] for i in indices]

    return DialogueSplit(shuffled_dialogues, shuffled_labels)


def _build_dialogue_from_liars_bench(row: dict) -> Dialogue:
    """Convert a Liars Bench row to a Dialogue."""
    dialogue: list[Message] = []
    messages = row["messages"]

    # Find the last assistant message
    last_assistant_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx]["role"] == "assistant":
            last_assistant_idx = idx
            break

    # Build dialogue with detect flag on last assistant message
    for idx, msg in enumerate(messages):
        dialogue.append(
            Message(
                role=msg["role"],
                content=msg["content"],
                detect=(idx == last_assistant_idx),
            )
        )
    return dialogue


def _load_liars_bench_dataset(config: str, sample_size: int, seed: int) -> DialogueSplit:
    """Load Liars Bench dataset with stratified sampling.

    Parameters
    ----------
    config : str
        Config name (e.g., "convincing-game", "insider-trading/report")
    sample_size : int
        Maximum number of samples (stratified by label)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    DialogueSplit
        Combined dialogues and labels (not yet split into train/val)
    """
    # Load the full dataset (load_filtered_dataset handles insider-trading/report splitting)
    dataset = load_filtered_dataset(
        dataset_id="Cadenza-Labs/liars-bench",
        config=config,  # Pass full config, load_filtered_dataset handles "/" splitting
        split="test",
        filter_model_name="llama-v3.3-70b-instruct",
        sample_size=None,  # Load all first, then stratify ourselves
        seed=seed
    )

    # Convert to dialogues and labels
    all_dialogues: list[Dialogue] = []
    all_labels: list[int] = []

    for row in dataset:
        dialogue = _build_dialogue_from_liars_bench(row)
        label = int(row.get("deceptive", 0))
        all_dialogues.append(dialogue)
        all_labels.append(label)

    # Stratified sampling: maintain ratio of honest (0) to deceptive (1)
    if len(all_dialogues) <= sample_size:
        # If dataset is smaller than sample_size, use all samples
        return DialogueSplit(all_dialogues, all_labels)

    # Separate by label
    honest_indices = [i for i, label in enumerate(all_labels) if label == 0]
    deceptive_indices = [i for i, label in enumerate(all_labels) if label == 1]

    # Calculate proportions
    total = len(all_dialogues)
    honest_ratio = len(honest_indices) / total
    deceptive_ratio = len(deceptive_indices) / total

    # Calculate stratified sample sizes
    num_honest = int(sample_size * honest_ratio)
    num_deceptive = sample_size - num_honest

    # Ensure we don't try to sample more than available
    num_honest = min(num_honest, len(honest_indices))
    num_deceptive = min(num_deceptive, len(deceptive_indices))

    # Sample from each group
    rng = random.Random(seed)
    sampled_honest = rng.sample(honest_indices, num_honest)
    sampled_deceptive = rng.sample(deceptive_indices, num_deceptive)

    # Combine and shuffle
    sampled_indices = sampled_honest + sampled_deceptive
    rng.shuffle(sampled_indices)

    sampled_dialogues = [all_dialogues[i] for i in sampled_indices]
    sampled_labels = [all_labels[i] for i in sampled_indices]

    return DialogueSplit(sampled_dialogues, sampled_labels)


def _gather_datasets(dataset_names: Sequence[str], seed: int) -> DialogueSplit:
    canonical = canonicalize_dataset_names(dataset_names)
    repository = DatasetRepository()

    combined_dialogues: list[Dialogue] = []
    combined_labels: list[int] = []
    for name in canonical:
        spec = AVAILABLE_DATASETS[name]

        # Handle BeaverTails datasets separately
        if spec.get("type") == "beaver":
            beaver_split = _load_beavertails_dataset(
                category=spec["category"],
                sample_size=spec["sample_size"],
                seed=seed
            )
            combined_dialogues.extend(beaver_split.dialogues)
            combined_labels.extend(beaver_split.labels)
        # Handle Liars Bench datasets
        elif spec.get("type") == "liars_bench":
            liars_split = _load_liars_bench_dataset(
                config=spec["config"],
                sample_size=spec["sample_size"],
                seed=seed
            )
            combined_dialogues.extend(liars_split.dialogues)
            combined_labels.extend(liars_split.labels)
        else:
            # Handle standard datasets from repository
            dataset = repository.get(
                spec["partial_id"],
                model=spec.get("model", "prewritten"),
                trim_reasoning=True,
                shuffle_upon_init=True,
            )
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
