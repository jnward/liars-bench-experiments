from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from deception_detection.types import Dialogue  # type: ignore


@dataclass
class DialogueInfo:
    dataset: str
    dialogue_index: int
    label: int
    meta: Dict[str, Any]


@dataclass
class VariantActivations:
    variant: str
    speaker: str
    choice: str
    pooled: torch.Tensor
    token_activations: Optional[torch.Tensor] = None


@dataclass
class DifferenceVectors:
    user: torch.Tensor
    assistant: Optional[torch.Tensor]
    combined: Optional[torch.Tensor]


@dataclass
class VariantRequest:
    info: DialogueInfo
    variant_key: str
    choice: str
    speaker: str
    dialogue: Dialogue


@dataclass
class VariantResult:
    request: VariantRequest
    pooled: torch.Tensor
    target_hidden: torch.Tensor
    token_index: int
