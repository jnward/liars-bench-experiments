from __future__ import annotations

from typing import Sequence

import torch


def pool_dialogue_activations(
    activations: torch.Tensor,
    counts: Sequence[int],
    dialogue_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Aggregate token-level activations into per-dialogue averages.

    Returns pooled activations, labels, and number of skipped dialogues.
    """
    if activations.dim() != 2:
        activations = activations.view(activations.size(0), -1)
    dim = activations.size(-1)
    pooled: list[torch.Tensor] = []
    pooled_labels: list[int] = []
    offset = 0
    skipped = 0
    for idx, count in enumerate(counts):
        if count is None:
            count = 0
        if count <= 0:
            skipped += 1
            continue
        span = activations[offset : offset + count].to(torch.float32)
        offset += count
        if span.numel() == 0:
            skipped += 1
            continue
        pooled.append(span.mean(dim=0, keepdim=True))
        pooled_labels.append(int(dialogue_labels[idx].item()))
    if pooled:
        pooled_tensor = torch.cat(pooled, dim=0)
        label_tensor = torch.tensor(pooled_labels, dtype=torch.long)
    else:
        pooled_tensor = torch.empty((0, dim), dtype=torch.float32)
        label_tensor = torch.empty(0, dtype=torch.long)
    return pooled_tensor, label_tensor, skipped
