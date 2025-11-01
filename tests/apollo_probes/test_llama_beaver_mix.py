from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DECEPTION_ROOT = Path("/workspace/jake/deception-detection")
if str(DECEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(DECEPTION_ROOT))

from deception_detection.types import Dialogue, Message  # noqa: E402

from apollo_probes.data import DialogueSplit  # noqa: E402
from apollo_probes import llama_beaver_mix as lbm  # noqa: E402


def _build_dialogue(text: str) -> Dialogue:
    return [
        Message(role="system", content="sys", detect=False),
        Message(role="user", content="prompt", detect=False),
        Message(role="assistant", content=text, detect=True),
    ]


def _make_core_split() -> DialogueSplit:
    dialogues = [_build_dialogue("core-honest"), _build_dialogue("core-deceptive")]
    labels = [0, 1]
    return DialogueSplit(dialogues, labels)


def test_build_llama_beaver_mix_injects_negatives(monkeypatch: pytest.MonkeyPatch) -> None:
    core_split = _make_core_split()
    monkeypatch.setattr(lbm, "_load_llama_datasets", lambda names, seed: core_split)

    beaver_samples = [
        lbm.BeaverSample(
            row_id="neg-privacy",
            prompt="privacy prompt",
            response="privacy response",
            categories={"privacy_violation": True},
        ),
        lbm.BeaverSample(
            row_id="neg-nocat",
            prompt="nocat prompt",
            response="nocat response",
            categories={},
        ),
        lbm.BeaverSample(
            row_id="filtered",
            prompt="filter prompt",
            response="filter response",
            categories={"hate_speech": True},
        ),
    ]

    monkeypatch.setattr(
        lbm,
        "load_beaver_samples",
        lambda split, sample_size, seed: (None, beaver_samples),
    )

    config = lbm.BeaverIngestionConfig(categories=("no_category", "privacy_violation"))
    combined = lbm.build_llama_beaver_mix(
        dataset_names=("repe_honesty__plain",),
        beaver_config=config,
        seed=123,
    )

    # core dialogues preserved
    assert combined.size == 4  # 2 core + 2 beaver negatives
    assert combined.labels.count(1) == 1
    assert combined.labels.count(0) == 3

    # ensure Beaver negatives appended with detect flag preserved
    neg_dialogues = combined.dialogues[-2:]
    for dialogue in neg_dialogues:
        assert dialogue[-1].role == "assistant"
        assert dialogue[-1].detect is True


def test_beaver_mix_slug_encodes_config() -> None:
    config = lbm.BeaverIngestionConfig(
        split="custom_split",
        categories=("no_category", "privacy_violation"),
        sample_size=100,
    )
    slug = lbm.beaver_mix_slug(config, core_datasets=("repe_honesty__plain", "roleplaying__plain"))
    assert "custom-split" in slug
    assert "core-repe-honesty-plain-roleplaying-plain" in slug
    assert "n100" in slug
    assert "no-category-privacy-violation" in slug
