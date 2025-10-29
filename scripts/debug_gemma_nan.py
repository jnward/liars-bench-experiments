#!/usr/bin/env python

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probe_pipeline.model_configs import get_model_config  # noqa: E402
from probe_pipeline.utils import (  # noqa: E402
    init_model,
    tokenize_conversation,
    _get_assistant_header_ids,
)


def main() -> None:
    load_dotenv()
    config = get_model_config(os.environ.get("PROBE_MODEL", "gemma_test"))
    dataset_path = Path(
        "/root/huggingface/datasets/Cadenza-Labs___liars-bench/convincing-game/0.0.0/"
        "b2ad937423b3b3b71de0ba3640f0545b536bd558/liars-bench-test.arrow"
    )
    ds = Dataset.from_file(str(dataset_path))
    row = next(row for row in ds if row.get("model") == config.filter_model_name)
    tokenizer, model, device, _dtype = init_model(config.model_name, seed=0)
    header_ids = _get_assistant_header_ids(tokenizer)

    convo = tokenize_conversation(row, tokenizer, header_ids)
    print("Assistant token indices:", convo.assistant_token_indices[:10])

    input_ids = convo.input_ids.unsqueeze(0).to(device)
    attention_mask = convo.attention_mask.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden = outputs.hidden_states[config.layer_start].to(torch.float32)
    positions = convo.assistant_token_indices
    vecs = hidden[0, positions, :]
    print("Hidden stats:", torch.isnan(vecs).any().item(), vecs.mean().item())


if __name__ == "__main__":
    main()
