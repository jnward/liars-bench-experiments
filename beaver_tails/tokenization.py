from __future__ import annotations

from typing import List, Sequence

import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from probe_pipeline.utils import _get_assistant_header_ids, locate_last_assistant_indices

from .schema import BeaverSample, TokenizedBeaverSample


def _extract_token_ids(encoded) -> List[int]:
    if isinstance(encoded, dict):
        token_ids = encoded.get("input_ids")
    else:
        token_ids = encoded

    if token_ids is None:
        raise ValueError("Tokenizer output missing input_ids.")

    if isinstance(token_ids, torch.Tensor):
        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze(0)
        token_ids = token_ids.tolist()

    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError(f"Unexpected nested token id structure: {token_ids}")
        token_ids = token_ids[0]

    return list(token_ids)


def tokenize_samples(
    samples: Sequence[BeaverSample],
    tokenizer: AutoTokenizer,
) -> list[TokenizedBeaverSample]:
    header_ids = _get_assistant_header_ids(tokenizer)
    tokenized: list[TokenizedBeaverSample] = []

    for sample in tqdm(samples, desc="Tokenizing BeaverTails", leave=False):
        messages = [
            {"role": "user", "content": sample.prompt},
            {"role": "assistant", "content": sample.response},
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        token_ids = _extract_token_ids(encoded)
        assistant_indices = locate_last_assistant_indices(token_ids, header_ids)
        if not assistant_indices:
            raise ValueError(f"Assistant indices not found for row {sample.row_id}")

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        tokenized.append(
            TokenizedBeaverSample(
                row_id=sample.row_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                assistant_token_indices=assistant_indices,
                categories=sample.categories,
            )
        )

    return tokenized
