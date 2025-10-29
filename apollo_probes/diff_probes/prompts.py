from __future__ import annotations

from typing import Dict

from deception_detection.types import Dialogue, Message  # type: ignore

from .config import APPENDED_USER_PROMPT_TEMPLATE, FORCED_ASSISTANT_TEMPLATE, PROMPT_VARIANTS


def _clone_without_detection(dialogue: Dialogue) -> Dialogue:
    """Return a deep copy of the dialogue with detect flags cleared."""
    cloned: Dialogue = []
    for message in dialogue:
        cloned.append(Message(role=message.role, content=message.content, detect=False))
    return cloned


def build_prompt_variants(dialogue: Dialogue) -> Dict[str, Dialogue]:
    """
    Construct appended dialogues for all user/assistant A/B variants.

    Returns a mapping from variant name (e.g. ``user_A``) to the new dialogue.
    """
    variants: Dict[str, Dialogue] = {}

    # User variants: appended user message marked for detection.
    for info in PROMPT_VARIANTS:
        if info.speaker == "user":
            content = APPENDED_USER_PROMPT_TEMPLATE.format(choice=info.choice)
            new_dialogue = _clone_without_detection(dialogue)
            new_dialogue.append(Message(role="user", content=content, detect=True))
            variants[info.variant] = new_dialogue

    # Assistant variants: append user message (no detection) then forced assistant reply.
    for info in PROMPT_VARIANTS:
        if info.speaker == "assistant":
            user_content = APPENDED_USER_PROMPT_TEMPLATE.format(choice=info.choice)
            new_dialogue = _clone_without_detection(dialogue)
            new_dialogue.append(Message(role="user", content=user_content, detect=False))
            assistant_content = FORCED_ASSISTANT_TEMPLATE.format(choice=info.choice)
            new_dialogue.append(Message(role="assistant", content=assistant_content, detect=True))
            variants[info.variant] = new_dialogue

    return variants
