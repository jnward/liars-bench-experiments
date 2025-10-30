import sys
from pathlib import Path

import torch
import pytest
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apollo_probes.detection import apply_chat_template_batch, create_detection_mask


class DummyMessage:
    def __init__(self, role: str, content: str, detect: bool = False):
        self.role = role
        self.content = content
        self.detect = detect


class DummyDialogue:
    def __init__(self, messages):
        self.messages = list(messages)

    def __iter__(self):
        return iter(self.messages)


@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
    ],
)
def test_detection_mask_marks_assistant_tokens(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dialogue = DummyDialogue(
        [
            DummyMessage("system", "You are a helpful assistant."),
            DummyMessage("user", "Tell me the sum of 2 and 2."),
            DummyMessage("assistant", "I will answer honestly: 4.", detect=True),
        ]
    )

    formatted = apply_chat_template_batch([dialogue], tokenizer)
    tokenized = tokenizer(
        formatted,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    mask = create_detection_mask([dialogue], formatted, tokenizer, tokenized)

    assert mask.shape == tokenized["input_ids"].shape

    assistant_text = dialogue.messages[-1].content.strip()
    full_text = formatted[0]
    start_char = full_text.find(assistant_text)
    assert start_char != -1, "Assistant text not found in formatted prompt"
    end_char = start_char + len(assistant_text)

    offsets = tokenized["offset_mapping"][0]
    expected = torch.zeros_like(mask[0])
    for idx, (start, end) in enumerate(offsets):
        start = start.item()
        end = end.item()
        if start == end:
            continue
        if start < end_char and end > start_char:
            expected[idx] = True

    assert expected.any(), "Expected mask is empty; offsets did not cover assistant text"
    assert torch.equal(mask[0], expected)
