# %% [markdown]
# Insider Trading: Report ↔ Confirmation Alignment
#
# Simple sanity notebook (script) for the Liars Bench insider-trading split.
# We align the initial report turns with their confirmation follow-ups by matching
# on the transcript prefix, then surface a few cases where the model tells the
# truth in the report but lies when challenged directly.
#
# Run with:
# ```
# uv run python insider-trading_matching.py
# ```
# Make sure `HF_TOKEN` is exported (e.g., via `.env`) so the gated dataset can be loaded.

# %%
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Tuple

from datasets import Dataset, load_dataset
from dotenv import load_dotenv


# %%
load_dotenv()


# %% Utility structures
@dataclass
class TurnRecord:
    index: int
    model: str
    deceptive: bool
    meta: dict
    messages: list[dict[str, str]]

    @property
    def turn_name(self) -> str:
        return self.meta["turn_name"]


def parse_meta(meta_str: str) -> dict:
    """`meta` is stored as the repr of a dict; convert it back."""
    return ast.literal_eval(meta_str)


def message_signature(messages: Iterable[dict]) -> Tuple[Tuple[str, str], ...]:
    """Return a hashable signature for a full conversation transcript."""
    return tuple((m["role"], m["content"]) for m in messages)


def build_turn_records(dataset: Dataset) -> list[TurnRecord]:
    records: list[TurnRecord] = []
    for row in dataset:
        records.append(
            TurnRecord(
                index=row["index"],
                model=row["model"],
                deceptive=bool(row["deceptive"]),
                meta=parse_meta(row["meta"]),
                messages=row["messages"],
            )
        )
    return records


def match_reports_to_confirmations(records: list[TurnRecord]) -> list[tuple[TurnRecord, TurnRecord]]:
    """Return list of (report, confirmation) pairs sharing the same conversation history."""
    report_lookup: dict[tuple[str, Tuple[Tuple[str, str], ...]], TurnRecord] = {}
    for record in records:
        if record.turn_name != "report":
            continue
        sig = (record.model, message_signature(record.messages))
        report_lookup[sig] = record

    pairs: list[tuple[TurnRecord, TurnRecord]] = []
    missing_prefix = 0
    for record in records:
        if record.turn_name != "confirmation":
            continue
        prefix_sig = (record.model, message_signature(record.messages[:-2]))
        report = report_lookup.get(prefix_sig)
        if report is None:
            missing_prefix += 1
            continue
        pairs.append((report, record))

    if missing_prefix:
        print(f"Warning: {missing_prefix} confirmation rows did not match any report.")
    print(f"Matched {len(pairs)} report ↔ confirmation pairs")
    return pairs


def summarize_pairs(pairs: list[tuple[TurnRecord, TurnRecord]]) -> None:
    counts = Counter((rep.deceptive, conf.deceptive) for rep, conf in pairs)
    print("Joint label counts (report_label, confirmation_label):")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")


def print_truth_report_false_confirm_examples(
    pairs: list[tuple[TurnRecord, TurnRecord]], limit: int = 3
) -> None:
    print(f"\nShowing up to {limit} cases where report is truthful but confirmation is deceptive:\n")
    shown = 0
    for report, confirmation in pairs:
        if report.deceptive or not confirmation.deceptive:
            continue
        print(f"Report idx={report.index} (model={report.model}) deceptive={report.deceptive}")
        print(report.messages[-1]["content"])
        print(f"\nConfirmation idx={confirmation.index} deceptive={confirmation.deceptive}")
        print(confirmation.messages[-1]["content"])
        print("\n" + "-" * 80 + "\n")
        shown += 1
        if shown >= limit:
            break

    if shown == 0:
        print("No examples found.")


# %% Load dataset
dataset = load_dataset("Cadenza-Labs/liars-bench", "insider-trading", split="test")
print(f"Loaded insider-trading split with {len(dataset)} rows")


# %% Build turn records
records = build_turn_records(dataset)
print(f"Constructed {len(records)} turn records")


# %% Match report ↔ confirmation pairs
pairs = match_reports_to_confirmations(records)


# %% Summaries & examples
summarize_pairs(pairs)
print_truth_report_false_confirm_examples(pairs, limit=3)

# %%
