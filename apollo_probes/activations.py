from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import torch
from tqdm.auto import tqdm

from .detection import apply_chat_template_batch, create_detection_mask

import sys

REPO_ROOT = Path("/workspace/jake/deception-detection")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deception_detection.types import Dialogue  # type: ignore


def extract_layerwise_detection_activations(
    dialogues: Sequence[Dialogue],
    labels: Sequence[int],
    model,
    tokenizer,
    layer_indices: Sequence[int],
    batch_size: int,
    desc: str,
) -> dict[int, dict[str, object]]:
    if not layer_indices:
        raise ValueError("layer_indices must be non-empty.")

    uniq_layers = sorted(set(layer_indices))
    device = next(model.parameters()).device
    hidden_size = getattr(model.config, "hidden_size", None)

    per_layer_vectors: dict[int, list[torch.Tensor]] = {layer: [] for layer in uniq_layers}
    per_layer_labels: dict[int, list[int]] = {layer: [] for layer in uniq_layers}
    per_layer_counts: dict[int, list[int]] = {layer: [] for layer in uniq_layers}
    per_layer_dialogue_labels: dict[int, list[int]] = {layer: [] for layer in uniq_layers}

    for start in tqdm(range(0, len(dialogues), batch_size), desc=desc):
        batch_dialogues = dialogues[start : start + batch_size]
        batch_labels = labels[start : start + batch_size]

        formatted = apply_chat_template_batch(batch_dialogues, tokenizer)
        tokens = tokenizer(
            formatted,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        detection_mask = create_detection_mask(batch_dialogues, formatted, tokens)

        outputs = model(
            input_ids=tokens["input_ids"].to(device),
            attention_mask=tokens["attention_mask"].to(device),
            output_hidden_states=True,
            use_cache=False,
        )

        for layer in uniq_layers:
            layer_slot = layer + 1  # account for embeddings
            if layer_slot >= len(outputs.hidden_states):
                raise ValueError(
                    f"Layer index {layer} unavailable (model has {len(outputs.hidden_states) - 1} layers)"
                )
            layer_hidden = outputs.hidden_states[layer_slot].to(torch.float32).cpu()
            if hidden_size is None:
                hidden_size = layer_hidden.size(-1)

            for sample_idx, lbl in enumerate(batch_labels):
                mask = detection_mask[sample_idx].bool()
                count = int(mask.sum().item())
                per_layer_counts[layer].append(count)
                per_layer_dialogue_labels[layer].append(int(lbl))
                if count == 0:
                    continue
                selected = layer_hidden[sample_idx][mask]
                per_layer_vectors[layer].append(selected)
                per_layer_labels[layer].extend([int(lbl)] * count)

        del outputs, tokens, detection_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result: dict[int, dict[str, object]] = {}
    for layer in uniq_layers:
        vectors = per_layer_vectors[layer]
        label_list = per_layer_labels[layer]
        if vectors:
            acts = torch.cat(vectors, dim=0).to(torch.float16)
        else:
            dim = hidden_size or getattr(model.config, "hidden_size", 0)
            acts = torch.empty((0, dim), dtype=torch.float16)
        label_tensor = torch.tensor(label_list, dtype=torch.long)
        result[layer] = {
            "activations": acts,
            "labels": label_tensor,
            "counts": list(per_layer_counts[layer]),
            "dialogue_labels": torch.tensor(per_layer_dialogue_labels[layer], dtype=torch.long),
        }
    return result
