#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path
import sys
from dotenv import load_dotenv

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probe_pipeline.model_configs import get_model_config  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


def format_block(label: str, content: str | None) -> str:
    if not content:
        return f"{label}: (none)"
    wrapped = textwrap.indent(content, prefix="    ")
    return f"{label}:\n{wrapped}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect chat-template and assistant header behaviour for a probe model."
    )
    parser.add_argument(
        "--model",
        "-m",
        default=os.environ.get("PROBE_MODEL", "llama"),
        help="Model key defined in probe_pipeline.model_configs (default: %(default)s).",
    )
    args = parser.parse_args()

    config = get_model_config(args.model)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    chat_template = getattr(tokenizer, "chat_template", "") or ""
    print(format_block("Chat template", chat_template))

    print(format_block("Special tokens map", repr(tokenizer.special_tokens_map)))

    header_override = config.assistant_header_override or os.environ.get(
        "PROBE_ASSISTANT_HEADER_OVERRIDE"
    )
    if header_override:
        header_str = header_override
        print(f"Assistant header (override from config): {header_str!r}")
    else:
        # Simple heuristic mirroring utils._get_assistant_header_ids
        name_lower = tokenizer.name_or_path.lower()
        if "<|im_start|>" in chat_template or "qwen" in name_lower:
            header_str = "<|im_start|>assistant"
        elif "<start_of_turn>" in chat_template or "gemma" in name_lower:
            header_str = "<start_of_turn>model"
        else:
            header_str = "<|start_header_id|>assistant<|end_header_id|>"
        print(f"Assistant header (heuristic): {header_str!r}")

    encoded = tokenizer.encode(header_str, add_special_tokens=False)
    print(f"Header token IDs ({len(encoded)} tokens): {encoded}")

    sample_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "State your model family."},
        {"role": "assistant", "content": "I am an alignment probe."},
    ]

    rendered = tokenizer.apply_chat_template(
        sample_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    print(format_block("Rendered sample dialogue", rendered))

    encoded_chat = tokenizer.apply_chat_template(
        sample_messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    token_ids = encoded_chat["input_ids"] if isinstance(encoded_chat, dict) else encoded_chat
    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    print(f"Encoded dialogue token count: {len(token_ids)}")
    print(f"First 40 token IDs: {token_ids[:40]}")


if __name__ == "__main__":
    main()
