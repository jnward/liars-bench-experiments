from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm

from ..activations import apply_chat_template_batch, create_detection_mask
from .config import PROMPT_VARIANTS
from .prompts import build_prompt_variants
from .types import DialogueInfo, VariantRequest, VariantResult


@dataclass
class DifferenceGroup:
    labels: torch.Tensor
    dialogue_indices: List[int]
    user_diff: torch.Tensor
    assistant_diff: torch.Tensor
    combined_diff: torch.Tensor
    variant_hidden: Optional[Dict[str, torch.Tensor]] = None


def prepare_requests(entries: Sequence[Tuple[DialogueInfo, object]]) -> List[VariantRequest]:
    requests: List[VariantRequest] = []
    for info, dialogue in entries:
        variants = build_prompt_variants(dialogue)
        for variant_info in PROMPT_VARIANTS:
            variant_dialogue = variants[variant_info.variant]
            requests.append(
                VariantRequest(
                    info=info,
                    variant_key=variant_info.variant,
                    choice=variant_info.choice,
                    speaker=variant_info.speaker,
                    dialogue=variant_dialogue,
                )
            )
    return requests


def extract_variant_results(
    requests: Sequence[VariantRequest],
    layer_index: int,
    batch_size: int,
    tokenizer,
    model,
) -> List[VariantResult]:
    if not requests:
        return []

    device = next(model.parameters()).device
    hidden_size = getattr(model.config, "hidden_size", None)
    results: List[VariantResult] = []

    for start in tqdm(range(0, len(requests), batch_size), desc="Variants", leave=False):
        batch_requests = requests[start : start + batch_size]
        batch_dialogues = [req.dialogue for req in batch_requests]
        formatted = apply_chat_template_batch(batch_dialogues, tokenizer)
        tokens = tokenizer(
            formatted,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        detection_mask = create_detection_mask(batch_dialogues, formatted, tokens)

        with torch.no_grad():
            outputs = model(
                input_ids=tokens["input_ids"].to(device),
                attention_mask=tokens["attention_mask"].to(device),
                output_hidden_states=True,
                use_cache=False,
            )

        layer_slot = layer_index + 1
        if layer_slot >= len(outputs.hidden_states):
            raise ValueError(
                f"Layer index {layer_index} exceeds available hidden states ({len(outputs.hidden_states) - 1})"
            )
        hidden = outputs.hidden_states[layer_slot].to(torch.float32).cpu()

        for idx, req in enumerate(batch_requests):
            mask = detection_mask[idx].nonzero(as_tuple=False).flatten()
            if mask.numel() == 0:
                raise ValueError(f"No detection tokens found for variant {req.variant_key}")
            token_index = int(mask[-1].item())
            token_hidden = hidden[idx, token_index, :].to(torch.float16)
            if mask.numel() == 1:
                pooled = token_hidden
            else:
                pooled = hidden[idx, mask, :].mean(dim=0).to(torch.float16)
            results.append(
                VariantResult(
                    request=req,
                    pooled=pooled,
                    target_hidden=token_hidden,
                    token_index=token_index,
                )
            )

        del outputs, tokens, detection_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if hidden_size is not None and results:
        for res in results:
            if res.target_hidden.numel() != hidden_size:
                raise ValueError("Hidden size mismatch encountered while extracting variants.")
    return results


def assemble_difference_vectors(results: Sequence[VariantResult], store_variant_hidden: bool = True) -> DifferenceGroup:
    by_dialogue: Dict[Tuple[str, int], Dict[str, VariantResult]] = {}
    labels_map: Dict[Tuple[str, int], int] = {}
    for result in results:
        key = (result.request.info.dataset, result.request.info.dialogue_index)
        by_dialogue.setdefault(key, {})[result.request.variant_key] = result
        labels_map[key] = result.request.info.label

    required_keys = {variant.variant for variant in PROMPT_VARIANTS}
    dialogue_indices: List[int] = []
    label_list: List[int] = []
    user_vectors: List[torch.Tensor] = []
    assistant_vectors: List[torch.Tensor] = []
    combined_vectors: List[torch.Tensor] = []
    variant_hidden: Optional[Dict[str, List[torch.Tensor]]] = (
        {variant.variant: [] for variant in PROMPT_VARIANTS} if store_variant_hidden else None
    )

    for (dataset, dialogue_index), variant_map in by_dialogue.items():
        if set(variant_map.keys()) != required_keys:
            continue
        user_a = variant_map["user_A"].target_hidden.to(torch.float32)
        user_b = variant_map["user_B"].target_hidden.to(torch.float32)
        assistant_a = variant_map["assistant_A"].target_hidden.to(torch.float32)
        assistant_b = variant_map["assistant_B"].target_hidden.to(torch.float32)

        user_diff = user_a - user_b
        assistant_diff = assistant_a - assistant_b
        combined = torch.cat([user_diff, assistant_diff], dim=-1)

        if store_variant_hidden and variant_hidden is not None:
            for key, res in variant_map.items():
                variant_hidden[key].append(res.target_hidden.to(torch.float32))

        dialogue_indices.append(dialogue_index)
        label_list.append(labels_map[(dataset, dialogue_index)])
        user_vectors.append(user_diff)
        assistant_vectors.append(assistant_diff)
        combined_vectors.append(combined)

    if not user_vectors:
        raise ValueError("No complete variant groups available for difference assembly.")

    def stack(tensors: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(tensors, dim=0).to(torch.float16)

    variant_hidden_stacked: Optional[Dict[str, torch.Tensor]] = None
    if store_variant_hidden and variant_hidden is not None:
        variant_hidden_stacked = {key: stack(vecs) for key, vecs in variant_hidden.items()}

    return DifferenceGroup(
        labels=torch.tensor(label_list, dtype=torch.long),
        dialogue_indices=dialogue_indices,
        user_diff=stack(user_vectors),
        assistant_diff=stack(assistant_vectors),
        combined_diff=stack(combined_vectors),
        variant_hidden=variant_hidden_stacked,
    )
