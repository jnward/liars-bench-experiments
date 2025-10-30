from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .config import EVAL_FILTER_MODEL


DATASET_ID = "Cadenza-Labs/liars-bench"

EVAL_DATASETS = [
    "convincing-game",
    "harm-pressure-knowledge-report",
    "instructed-deception",
    "insider-trading/report",
    "insider-trading/confirmation",
    "soft-trigger",
]


@dataclass
class EvalSpec:
    dataset_id: str  # e.g., "Cadenza-Labs/liars-bench"
    config: str
    split: str = "test"
    filter_model: str = EVAL_FILTER_MODEL


def liars_bench_specs() -> list[EvalSpec]:
    specs: list[EvalSpec] = []
    for cfg in EVAL_DATASETS:
        specs.append(EvalSpec(dataset_id=DATASET_ID, config=cfg))
    return specs


def slugify_config(config: str) -> str:
    return config.replace("/", "_")
