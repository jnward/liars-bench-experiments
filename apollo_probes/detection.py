from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence

import torch

REPO_ROOT = Path("/workspace/jake/deception-detection")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deception_detection.types import Dialogue  # type: ignore


def _role_headers(tokenizer) -> dict[str, str]:
    template = getattr(tokenizer, "chat_template", "") or ""
    if "<|im_start|>" in template or "qwen" in tokenizer.name_or_path.lower():
        return {
            "system": "<|im_start|>system\n",
            "user": "<|im_start|>user\n",
            "assistant": "<|im_start|>assistant\n",
        }
    # Llama-style headers
    return {
        "system": "<|start_header_id|>system<|end_header_id|>\n\n",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    }


def _end_marker(tokenizer) -> str:
    template = getattr(tokenizer, "chat_template", "") or ""
    if "<|im_end|>" in template or "qwen" in tokenizer.name_or_path.lower():
        return "<|im_end|>"
    return "<|eot_id|>"


def dialogue_to_messages(dialogue: Dialogue) -> list[dict[str, str]]:
    return [{"role": msg.role, "content": msg.content} for msg in dialogue]


def apply_chat_template_batch(dialogues: Sequence[Dialogue], tokenizer) -> list[str]:
    messages_batch = [dialogue_to_messages(d) for d in dialogues]
    return [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        for messages in messages_batch
    ]


def create_detection_mask(
    dialogues: Sequence[Dialogue],
    formatted_dialogues: Sequence[str],
    tokenizer,
    tokenizer_output,
) -> torch.Tensor:
    detection_mask = torch.zeros_like(tokenizer_output["input_ids"]).bool()
    headers = _role_headers(tokenizer)
    end_marker = _end_marker(tokenizer)

    for batch_idx, dialogue in enumerate(dialogues):
        char_idx = 0
        formatted = formatted_dialogues[batch_idx]
        for message in dialogue:
            header = headers.get(message.role)
            if header is None:
                continue

            header_start = formatted.find(header, char_idx)
            if header_start == -1:
                continue

            start_char = header_start + len(header)
            end_marker_idx = formatted.find(end_marker, start_char)
            if end_marker_idx == -1:
                end_marker_idx = len(formatted)

            if getattr(message, "detect", False) and start_char < end_marker_idx:
                start_tok = tokenizer_output.char_to_token(batch_idx, start_char)
                end_tok = tokenizer_output.char_to_token(batch_idx, max(start_char, end_marker_idx - 1))
                if start_tok is not None and end_tok is not None:
                    detection_mask[batch_idx, start_tok : end_tok + 1] = True

            char_idx = end_marker_idx + len(end_marker)

            if char_idx < len(formatted) and formatted[char_idx] == "\n":
                char_idx += 1

    return detection_mask
