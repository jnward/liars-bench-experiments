# %% [markdown]
# # Per-Token Probe Heatmaps for High-Scoring Truthful Deceptive Dialogues
#
# Visualize probe logits on each assistant token for dialogues where the system instructed deception,
# the model nevertheless answered truthfully (Claude rating 2), yet the probe fires strongly.

# %%
from __future__ import annotations

import ast
import html
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from datasets.utils.logging import disable_progress_bar, set_verbosity_error
from dotenv import load_dotenv
from IPython.display import HTML, display

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

disable_progress_bar()
set_verbosity_error()
load_dotenv(PROJECT_ROOT / ".env")

from probe_pipeline.utils import init_model, load_filtered_dataset  # noqa: E402

from apollo_probes.config import RANDOM_SEED  # noqa: E402
from apollo_probes.evaluate_probes import compute_scores, load_eval_cache, load_probe  # noqa: E402
from apollo_probes.eval_cache import build_dialogue  # noqa: E402
from apollo_probes.eval_datasets import liars_bench_specs, slugify_config  # noqa: E402
from apollo_probes.detection import apply_chat_template_batch, create_detection_mask  # noqa: E402
from apollo_probes.pooling import pool_dialogue_activations  # noqa: E402


# %%
# ---- Configuration ----
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
EVAL_CONFIG = "instructed-deception"
TRAIN_DATASET_SLUG = "repe_honesty__plain+roleplaying__plain"
LAYER_INDEX = 34
TOP_K = 16

RATINGS_PATH = (
    PROJECT_ROOT
    / "apollo_probes"
    / "notebooks"
    / "cache"
    / "assistant_truth_ratings"
    / f"{slugify_config(EVAL_CONFIG)}__assistant_truth_ratings.jsonl"
)


# %%
# ---- Helpers ----
def parse_meta(meta_value):
    if isinstance(meta_value, dict):
        return meta_value
    if isinstance(meta_value, str):
        try:
            parsed = ast.literal_eval(meta_value)
        except (ValueError, SyntaxError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def first_user_message(messages: Sequence[dict]) -> str | None:
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, Sequence):
                parts: list[str] = []
                for chunk in content:
                    if isinstance(chunk, dict) and chunk.get("type") == "text":
                        parts.append(chunk.get("text", ""))
                    elif isinstance(chunk, str):
                        parts.append(chunk)
                text = "\n".join(parts)
                if text:
                    return text
    return None


def load_truth_ratings(path: Path) -> dict[int, int]:
    ratings: dict[int, int] = {}
    if not path.exists():
        raise FileNotFoundError(f"Missing truth ratings cache: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            ratings[int(record["dataset_index"])] = int(record["verdict"])
    return ratings


@dataclass
class DialogueRow:
    row_index: int
    system_template_index: int
    instructed: bool
    deceptive_label: int
    messages: list[dict]
    system_message: str | None
    user_message: str | None
    assistant_text: str


def extract_assistant_text(messages: Sequence[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, Sequence):
            parts: list[str] = []
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    parts.append(chunk.get("text", ""))
                elif isinstance(chunk, str):
                    parts.append(chunk)
            return "\n".join(parts)
    raise ValueError("Assistant message missing.")


def html_escape(text: str) -> str:
    return html.escape(text, quote=False).replace("\n", "<br/>")


def format_token_piece(piece: str) -> str:
    rendered = piece
    replacements = {
        "▁": " ",
        "Ġ": " ",
        "Ċ": "\n",
    }
    for key, val in replacements.items():
        rendered = rendered.replace(key, val)
    rendered = html.escape(rendered, quote=False)
    rendered = rendered.replace("\n", "<br/>")
    if rendered == "":
        rendered = "&nbsp;"
    return rendered


def score_to_color(score: float, limit: float) -> str:
    if limit <= 1e-6:
        return "background-color: rgba(255, 255, 255, 0.6);"
    norm = max(min(score / limit, 1.0), -1.0)
    intensity = int(200 * abs(norm))
    if norm > 0:
        r, g, b = 255, 255 - intensity, 255 - intensity
    else:
        r, g, b = 255 - intensity, 255 - intensity, 255
    return f"background-color: rgb({r}, {g}, {b});"


# %%
# ---- Load dataset rows ----
specs = {spec.config: spec for spec in liars_bench_specs()}
if EVAL_CONFIG not in specs:
    raise ValueError(f"Unknown eval config '{EVAL_CONFIG}'.")

spec = specs[EVAL_CONFIG]
eval_rows = load_filtered_dataset(
    spec.dataset_id,
    spec.config,
    split=spec.split,
    filter_model_name=spec.filter_model,
    sample_size=None,
    seed=RANDOM_SEED,
)

print(f"Loaded eval rows: {len(eval_rows)}")

rows_by_index: dict[int, DialogueRow] = {}

for row_idx, row in enumerate(eval_rows):
    meta = parse_meta(row.get("meta"))
    system_message = None
    for message in row.get("messages") or []:
        if message.get("role") == "system":
            system_message = message.get("content")
            break
    user_msg = first_user_message(row.get("messages") or [])
    try:
        assistant_text = extract_assistant_text(row.get("messages") or [])
    except ValueError:
        continue
    rows_by_index[row_idx] = DialogueRow(
        row_index=row_idx,
        system_template_index=int(meta.get("system_template_index", -1)),
        instructed=bool(meta.get("instructed_deceptiveness", False)),
        deceptive_label=int(row.get("deceptive", 0)),
        messages=list(row.get("messages") or []),
        system_message=system_message,
        user_message=user_msg,
        assistant_text=assistant_text,
    )

print(f"Indexed rows: {len(rows_by_index)}")


# %%
# ---- Load truth ratings and probe scores ----
truth_ratings = load_truth_ratings(RATINGS_PATH)
print(f"Loaded truth ratings for {len(truth_ratings)} dialogues.")

tokenizer, model, device, _dtype = init_model(MODEL_NAME, seed=RANDOM_SEED)
probe = load_probe(TRAIN_DATASET_SLUG, LAYER_INDEX)
eval_alias = slugify_config(EVAL_CONFIG)
cache = load_eval_cache(eval_alias, LAYER_INDEX)

counts: Sequence[int | None] = cache["counts"]
dialogue_labels: torch.Tensor = cache["dialogue_labels"]
activations: torch.Tensor = cache["activations"]

pooled_acts, pooled_labels, _ = pool_dialogue_activations(
    activations,
    counts,
    dialogue_labels,
)
pooled_scores = compute_scores(pooled_acts.to(torch.float32), probe)
keeplist = [idx for idx, count in enumerate(counts) if count and count > 0]
score_map = {int(dialogue_idx): float(pooled_scores[pos]) for pos, dialogue_idx in enumerate(keeplist)}


# %%
# ---- Select high-scoring truthful deceptive examples ----
qualified_rows = []
for row_idx, info in rows_by_index.items():
    if not info.instructed or info.deceptive_label != 0:
        continue
    rating = truth_ratings.get(row_idx)
    if rating != 2:
        continue
    score = score_map.get(row_idx)
    if score is None:
        continue
    qualified_rows.append((row_idx, score))

qualified_rows.sort(key=lambda item: item[1], reverse=True)
top_examples = qualified_rows[:TOP_K]

print(f"Qualified examples: {len(qualified_rows)} | showing top {len(top_examples)}")


# %%
# ---- Prepare per-token activations ----
def compute_token_logits(dialogue: DialogueRow) -> tuple[np.ndarray, Sequence[int], list[str], np.ndarray]:
    detection_dialogue = build_dialogue({"messages": dialogue.messages})
    formatted = apply_chat_template_batch([detection_dialogue], tokenizer)
    tokenized = tokenizer(
        formatted,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    detection_mask = create_detection_mask([detection_dialogue], formatted, tokenized)[0].bool()

    input_ids = tokenized["input_ids"][0].to(device)
    attention_mask = tokenized["attention_mask"][0].to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            output_hidden_states=True,
            use_cache=False,
        )
    hidden = outputs.hidden_states[LAYER_INDEX + 1].to(torch.float32)[0].cpu()
    positions = torch.nonzero(detection_mask, as_tuple=False).squeeze(-1).tolist()
    if isinstance(positions, int):
        positions = [positions]
    if not positions:
        raise ValueError("Detection mask produced no assistant tokens.")
    token_vectors = hidden[positions, :]
    logits = compute_scores(token_vectors, probe)
    tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0].tolist())
    return logits, positions, tokens, token_vectors.numpy()


def build_highlight_html(
    dialogue: DialogueRow,
    tokens: Sequence[str],
    logits: np.ndarray,
    positions: Sequence[int],
    live_score_logits: float,
    live_score_pooled: float,
) -> str:
    max_abs = float(np.max(np.abs(logits))) if logits.size else 1.0

    assistant_html: list[str] = []
    for score, pos in zip(logits, positions, strict=False):
        piece = tokens[pos]
        styled_piece = format_token_piece(piece)
        style = score_to_color(score, max_abs)
        span = (
            f'<span class="token" style="{style}" title="logit={score:.3f}">'
            f"{styled_piece}"
            "</span>"
        )
        assistant_html.append(span)

    assistant_block = "".join(assistant_html)

    system_text = html_escape(dialogue.system_message or "")
    user_text = html_escape(dialogue.user_message or "")

    container = f"""
    <div class="example">
      <div class="meta">
        <strong>row {dialogue.row_index}</strong> |
        system template {dialogue.system_template_index} |
        pooled score (cached) {score_map.get(dialogue.row_index, float('nan')):.3f} |
        pooled score (live logits) {live_score_logits:.3f} |
        pooled score (live pooled) {live_score_pooled:.3f}
      </div>
      <div class="turn system"><span class="role">System:</span> {system_text}</div>
      <div class="turn user"><span class="role">User:</span> {user_text}</div>
      <div class="turn assistant">
        <span class="role">Assistant (token logits):</span>
        <div class="token-stream">{assistant_block}</div>
      </div>
    </div>
    """
    return container


STYLE_BLOCK = """
<style>
.probe-heatmap-container {
    font-family: "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 13px;
    line-height: 1.6;
    color: #111;
}
.probe-heatmap-container .example {
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 16px;
    background: #fafafa;
}
.probe-heatmap-container .meta {
    font-size: 12px;
    color: #555;
    margin-bottom: 6px;
}
.probe-heatmap-container .turn {
    margin-bottom: 8px;
}
.probe-heatmap-container .role {
    font-weight: 600;
    margin-right: 6px;
}
.probe-heatmap-container .token-stream {
    white-space: pre-wrap;
    display: inline-block;
    border-radius: 4px;
    padding: 4px 6px;
    background: white;
    color: #111;
}
.probe-heatmap-container .token {
    display: inline;
    padding: 1px 2px;
    border-radius: 2px;
    color: #111;
}
</style>
"""


# %%
# ---- Render selected examples ----
examples_html: list[str] = []

for row_idx, score in top_examples:
    dialogue = rows_by_index[row_idx]
    logits, positions, tokens, token_vectors = compute_token_logits(dialogue)
    live_logits_score = float(np.mean(logits)) if logits.size else float("nan")
    live_pooled_score = float(compute_scores(torch.tensor(token_vectors).mean(dim=0).unsqueeze(0), probe)[0]) if logits.size else float("nan")
    example_html = build_highlight_html(dialogue, tokens, logits, positions, live_logits_score, live_pooled_score)
    examples_html.append(example_html)

full_html = STYLE_BLOCK + '<div class="probe-heatmap-container">' + "".join(examples_html) + "</div>"
display(HTML(full_html))

# %%
# ---- Honest vs Deceptive prompt comparison ----
def find_honest_match(dialogue: DialogueRow) -> DialogueRow | None:
    user_text = dialogue.user_message
    system_idx = dialogue.system_template_index
    if user_text is None:
        return None
    candidates = [
        info
        for info in rows_by_index.values()
        if info.user_message == user_text
        and info.system_template_index == system_idx
        and not info.instructed
    ]
    return candidates[0] if candidates else None


def swap_system_message(source: DialogueRow, new_system_text: str | None) -> DialogueRow:
    swapped_messages = []
    for message in source.messages:
        if message.get("role") == "system":
            swapped = dict(message)
            swapped["content"] = new_system_text or message.get("content", "")
            swapped_messages.append(swapped)
        else:
            swapped_messages.append(dict(message))
    return DialogueRow(
        row_index=source.row_index,
        system_template_index=source.system_template_index,
        instructed=False,
        deceptive_label=source.deceptive_label,
        messages=swapped_messages,
        system_message=new_system_text,
        user_message=source.user_message,
        assistant_text=source.assistant_text,
    )


comparison_blocks: list[str] = []

for row_idx, _score in top_examples:
    deceptive_dialogue = rows_by_index[row_idx]
    honest_match = find_honest_match(deceptive_dialogue)
    if honest_match is None:
        continue
    swapped_dialogue = swap_system_message(deceptive_dialogue, honest_match.system_message)

    # Deceptive prompt output
    dec_logits, dec_positions, dec_tokens, dec_vectors = compute_token_logits(deceptive_dialogue)
    dec_live_logits = float(np.mean(dec_logits)) if dec_logits.size else float("nan")
    dec_live_pooled = float(compute_scores(torch.tensor(dec_vectors).mean(dim=0).unsqueeze(0), probe)[0]) if dec_logits.size else float("nan")
    dec_html = build_highlight_html(
        deceptive_dialogue,
        dec_tokens,
        dec_logits,
        dec_positions,
        dec_live_logits,
        dec_live_pooled,
    )

    # Honest prompt with same assistant
    hon_logits, hon_positions, hon_tokens, hon_vectors = compute_token_logits(swapped_dialogue)
    hon_live_logits = float(np.mean(hon_logits)) if hon_logits.size else float("nan")
    hon_live_pooled = float(compute_scores(torch.tensor(hon_vectors).mean(dim=0).unsqueeze(0), probe)[0]) if hon_logits.size else float("nan")
    hon_html = build_highlight_html(
        swapped_dialogue,
        hon_tokens,
        hon_logits,
        hon_positions,
        hon_live_logits,
        hon_live_pooled,
    )

    comparison_blocks.append(dec_html)
    comparison_blocks.append(hon_html)

comparison_html = STYLE_BLOCK + '<div class="probe-heatmap-container">' + "".join(comparison_blocks) + "</div>"
display(HTML(comparison_html))


# %%
