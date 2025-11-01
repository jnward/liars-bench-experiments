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


BEGIN_PATTERN = r"((<\|pad\|>)*(<\|begin_of_text\|>|<\|startoftext\|>))?"
# Llama pattern: <|start_header_id|>role<|end_header_id|>
LLAMA_HEADER_PATTERN = r"<\|start_header_id\|>(system|user|assistant)<\|end_header_id\|>\n\n"
# GPT-OSS pattern: <|start|>role<|message|> or <|start|>role<|channel|>channel_name<|message|>
GPTOSS_HEADER_PATTERN = r"<\|start\|>(system|user|assistant)(<\|channel\|>\w+)?<\|message\|>"
HEADER_PATTERN = rf"({LLAMA_HEADER_PATTERN}|{GPTOSS_HEADER_PATTERN})"
EOT_PATTERN = r"(<\|eot_id\|>|<\|end\|>|<\|return\|>)"
DATE_INFO = r"(Cutting Knowledge Date: December 2023\nToday Date: \d\d \w\w\w 202[45]\n\n)?"
# System message for GPT-OSS includes knowledge cutoff, date, reasoning, and channel info
GPTOSS_SYSTEM_INFO = r"(You are ChatGPT.*?# Valid channels:.*?message\.\n)?"
PREFIX_REGEX = re.compile(
    rf"({BEGIN_PATTERN}{HEADER_PATTERN}({DATE_INFO}|{GPTOSS_SYSTEM_INFO}))?({EOT_PATTERN}{HEADER_PATTERN})?|(\n\n)?"
)


def dialogue_to_messages(dialogue: Dialogue) -> list[dict[str, str]]:
    return [{"role": msg.role, "content": msg.content} for msg in dialogue]


def apply_chat_template_batch(dialogues: Sequence[Dialogue], tokenizer) -> list[str]:
    messages_batch = [dialogue_to_messages(d) for d in dialogues]
    return [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        for messages in messages_batch
    ]


def _get_assistant_header_for_detection(tokenizer) -> list[int]:
    """Get assistant header token IDs for detection masking."""
    # GPT-OSS models use <|start|>assistant
    if 'gpt_oss' in getattr(tokenizer, 'name_or_path', '').lower():
        return tokenizer.encode("<|start|>assistant", add_special_tokens=False)
    # Llama models use <|start_header_id|>assistant<|end_header_id|>
    else:
        return tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False)


def _locate_last_assistant_tokens(input_ids: list[int], header_ids: list[int]) -> list[int]:
    """Find token indices for the last assistant message."""
    matches = []
    header_len = len(header_ids)
    for pos in range(0, len(input_ids) - header_len + 1):
        if input_ids[pos:pos + header_len] == header_ids:
            matches.append(pos)
    if not matches:
        return []
    # Use last match
    chosen_idx = matches[-1]
    # Return all tokens from header to end of sequence
    return list(range(chosen_idx, len(input_ids)))


def create_detection_mask(
    dialogues: Sequence[Dialogue],
    formatted_dialogues: Sequence[str],
    tokenizer_output,
) -> torch.Tensor:
    detection_mask = torch.zeros_like(tokenizer_output["input_ids"]).bool()

    # Get tokenizer object (tokenizer_output is a BatchEncoding from tokenizer())
    # We need the tokenizer itself to get the header IDs
    # For now, we'll use a heuristic based on the formatted dialogue structure

    # Check if this is GPT-OSS format
    is_gptoss = any('<|start|>assistant' in fmt for fmt in formatted_dialogues)

    for batch_idx, dialogue in enumerate(dialogues):
        # Find which messages have detect=True
        detect_messages = [msg for msg in dialogue if getattr(msg, "detect", False)]
        if not detect_messages:
            continue

        # For GPT-OSS, use token-based matching
        if is_gptoss:
            from transformers import AutoTokenizer
            # Get the tokenizer from the output (hacky but works)
            # Actually, we need to get it passed in or use the global one
            # For now, let's use a different approach - just use char_to_token on assistant content
            formatted = formatted_dialogues[batch_idx]

            # Find the last occurrence of assistant content
            # This is the message we want to detect
            detect_msg = detect_messages[-1]  # Usually only one
            content = detect_msg.content.strip()

            # Find where this content appears in the formatted text
            # Look for it after an assistant header
            assistant_marker = "<|start|>assistant"
            assistant_pos = formatted.rfind(assistant_marker)
            if assistant_pos != -1:
                # Find the content after the message marker
                message_marker = "<|message|>"
                message_pos = formatted.find(message_marker, assistant_pos)
                if message_pos != -1:
                    content_start = message_pos + len(message_marker)
                    # Find where content actually starts (skip whitespace)
                    while content_start < len(formatted) and formatted[content_start] in ' \n':
                        content_start += 1

                    # Find end of content (before <|end|> or <|return|>)
                    content_end = formatted.find("<|end|>", content_start)
                    if content_end == -1:
                        content_end = formatted.find("<|return|>", content_start)
                    if content_end == -1:
                        content_end = len(formatted)

                    # Convert char positions to token positions
                    start_tok = tokenizer_output.char_to_token(batch_idx, content_start)
                    end_tok = tokenizer_output.char_to_token(batch_idx, max(content_start, content_end - 1))

                    if start_tok is not None and end_tok is not None:
                        detection_mask[batch_idx, start_tok:end_tok + 1] = True
        else:
            # Original Llama logic
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
