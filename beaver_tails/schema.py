from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch


CategoryMap = Dict[str, bool]


@dataclass(slots=True)
class BeaverSample:
    row_id: str
    prompt: str
    response: str
    categories: CategoryMap


@dataclass(slots=True)
class TokenizedBeaverSample:
    row_id: str
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    assistant_token_indices: Sequence[int]
    categories: CategoryMap


@dataclass(slots=True)
class ActivationRecord:
    row_id: str
    categories: CategoryMap
    hidden: torch.Tensor  # shape: [tokens, hidden_dim]
    token_ids: torch.LongTensor  # shape: [tokens]


@dataclass(slots=True)
class CacheShardMeta:
    shard_path: Path
    num_examples: int
    total_tokens: int


@dataclass(slots=True)
class CacheManifest:
    model_name: str
    layer_index: int
    sample_size: int | None
    seed: int
    split: str
    shard_metas: List[CacheShardMeta]
