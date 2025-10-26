from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Sequence

import torch

REPO_ROOT = Path("/workspace/jake/deception-detection")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deception_detection.types import Dialogue  # type: ignore


BEGIN_PATTERN = r"((<\|pad\|>)*(<\|begin_of_text\|>))?"
HEADER_PATTERN = r"<\|start_header_id\|>(system|user|assistant)<\|end_header_id\|>\n\n"
EOT_PATTERN = r"<\|eot_id\|>"
DATE_INFO = r"(Cutting Knowledge Date: December 2023\nToday Date: \d\d \w\w\w 202[45]\n\n)?"
PREFIX_REGEX = re.compile(
    rf"({BEGIN_PATTERN}{HEADER_PATTERN}{DATE_INFO})?({EOT_PATTERN}{HEADER_PATTERN})?|(\n\n)?"
)


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
    tokenizer_output,
) -> torch.Tensor:
    detection_mask = torch.zeros_like(tokenizer_output["input_ids"]).bool()

    for batch_idx, dialogue in enumerate(dialogues):
        char_idx = 0
        formatted = formatted_dialogues[batch_idx]

        for message in dialogue:
            match = PREFIX_REGEX.match(formatted[char_idx:])
            if match is None:
                continue

            start_char_idx = char_idx + match.end()
            content_to_find = message.content.strip()
            if not content_to_find:
                char_idx = start_char_idx
                continue

            content_start = formatted.find(content_to_find, start_char_idx)
            if content_start == -1 or content_start != start_char_idx:
                char_idx = start_char_idx + len(message.content)
                continue

            end_char_idx = start_char_idx + len(content_to_find)

            if getattr(message, "detect", False):
                start_tok = tokenizer_output.char_to_token(batch_idx, start_char_idx)
                end_tok = tokenizer_output.char_to_token(batch_idx, max(start_char_idx, end_char_idx - 1))
                if start_tok is not None and end_tok is not None:
                    detection_mask[batch_idx, start_tok : end_tok + 1] = True

            char_idx = end_char_idx

    return detection_mask
