# %% [markdown]
# # Insider-Trading Report – Probe Heatmaps for Misclassified Dialogues
#
# Visualize per-token probe logits for false positives/false negatives on the
# `insider-trading/report` evaluation split.

# %%
from __future__ import annotations

import ast
import html
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
EVAL_CONFIG = "insider-trading/report"
TRAIN_DATASET_SLUG = "repe_honesty__plain+roleplaying__plain"
LAYER_INDEX = 34
TOP_FALSE_POSITIVES = 8
TOP_FALSE_NEGATIVES = 8
THRESHOLD = 0.0


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


def html_escape(text: str | None) -> str:
    if not text:
        return ""
    return html.escape(text, quote=False).replace("\n", "<br/>")


def format_token_piece(piece: str) -> str:
    replacements = {"▁": " ", "Ġ": " ", "Ċ": "\n"}
    rendered = piece
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    rendered = html.escape(rendered, quote=False)
    rendered = rendered.replace("\n", "<br/>")
    return rendered or "&nbsp;"


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


# %%
# ---- Load dataset rows ----
specs = {spec.config: spec for spec in liars_bench_specs()}
if EVAL_CONFIG not in specs:
    raise ValueError(f"Unknown eval config '{EVAL_CONFIG}'.")

spec = specs[EVAL_CONFIG]

os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache_home"))
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

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
    user_text = first_user_message(row.get("messages") or [])
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
        user_message=user_text,
        assistant_text=assistant_text,
    )

print(f"Indexed rows: {len(rows_by_index)}")


# %%
# ---- Load eval cache & probe ----
alias = slugify_config(EVAL_CONFIG)
cache = load_eval_cache(alias, LAYER_INDEX)
counts: Sequence[int | None] = cache["counts"]
dialogue_labels: torch.Tensor = cache["dialogue_labels"]
activations: torch.Tensor = cache["activations"]

tokenizer, model, device, _dtype = init_model(MODEL_NAME, seed=RANDOM_SEED)
probe = load_probe(TRAIN_DATASET_SLUG, LAYER_INDEX)

pooled_acts, pooled_labels, _ = pool_dialogue_activations(
    activations,
    counts,
    dialogue_labels,
)
pooled_scores = compute_scores(pooled_acts.to(torch.float32), probe)

kept_indices = [idx for idx, count in enumerate(counts) if count and count > 0]
score_map = {int(idx): float(pooled_scores[pos]) for pos, idx in enumerate(kept_indices)}
label_map = {
    int(idx): int(pooled_labels[pos].item()) for pos, idx in enumerate(kept_indices)
}


# %%
# ---- Identify false positives / false negatives ----
false_positives: list[int] = []
false_negatives: list[int] = []

for idx, info in rows_by_index.items():
    score = score_map.get(idx)
    label = label_map.get(idx)
    if score is None or label is None:
        continue
    predicted = 1 if score > THRESHOLD else 0
    if predicted == 1 and label == 0:
        false_positives.append(idx)
    elif predicted == 0 and label == 1:
        false_negatives.append(idx)

false_positives.sort(key=lambda i: score_map[i], reverse=True)
false_negatives.sort(key=lambda i: score_map[i])

selected_fp = false_positives[:TOP_FALSE_POSITIVES]
selected_fn = false_negatives[:TOP_FALSE_NEGATIVES]

print(
    f"False positives: {len(false_positives)} total | showing top {len(selected_fp)}"
)
print(
    f"False negatives: {len(false_negatives)} total | showing top {len(selected_fn)}"
)


# %%
# ---- Token-level logits ----
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
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"), torch.no_grad():
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
    return logits, positions, tokens, token_vectors


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
.probe-heatmap-container .tag {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 11px;
    margin-right: 6px;
}
.probe-heatmap-container .tag-fp {
    background: #ffe5e5;
    color: #a00;
}
.probe-heatmap-container .tag-fn {
    background: #e5f0ff;
    color: #005;
}
.probe-heatmap-container .tag-sample {
    background: #f0f0f0;
    color: #444;
}
.probe-heatmap-container .tag-pos {
    background: #e8ffe8;
    color: #045c04;
}
</style>
"""


def build_highlight_html(
    dialogue: DialogueRow,
    tokens: Sequence[str],
    logits: np.ndarray,
    positions: Sequence[int],
    live_logits_score: float,
    live_pooled_score: float,
    tag: str,
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
    cached_score = score_map.get(dialogue.row_index, float("nan"))
    truth_label = label_map.get(dialogue.row_index, None)
    predicted_label = 1 if cached_score > THRESHOLD else 0

    tag_class = {"FP": "tag-fp", "FN": "tag-fn", "Sample": "tag-sample", "Sample L1": "tag-pos"}.get(tag, "tag-sample")

    container = f"""
    <div class="example">
      <div class="meta">
        <span class="tag {tag_class}">{tag}</span>
        <strong>row {dialogue.row_index}</strong> |
        truth label {truth_label} | predicted {predicted_label} |
        pooled score (cached) {cached_score:.3f} |
        pooled score (live logits) {live_logits_score:.3f} |
        pooled score (live pooled) {live_pooled_score:.3f}
      </div>
      <div class="turn assistant">
        <span class="role">Assistant (token logits):</span>
        <div class="token-stream">{assistant_block}</div>
      </div>
    </div>
    """
    return container


blocks: list[str] = []

for idx in selected_fp:
    dialogue = rows_by_index[idx]
    logits, positions, tokens, token_vectors = compute_token_logits(dialogue)
    live_logits_score = float(np.mean(logits)) if logits.size else float("nan")
    live_pooled_score = (
        float(compute_scores(token_vectors.mean(dim=0, keepdim=True), probe)[0])
        if logits.size
        else float("nan")
    )
    blocks.append(
        build_highlight_html(
            dialogue,
            tokens,
            logits,
            positions,
            live_logits_score,
            live_pooled_score,
            tag="FP",
        )
    )

for idx in selected_fn:
    dialogue = rows_by_index[idx]
    logits, positions, tokens, token_vectors = compute_token_logits(dialogue)
    live_logits_score = float(np.mean(logits)) if logits.size else float("nan")
    live_pooled_score = (
        float(compute_scores(token_vectors.mean(dim=0, keepdim=True), probe)[0])
        if logits.size
        else float("nan")
    )
    blocks.append(
        build_highlight_html(
            dialogue,
            tokens,
            logits,
            positions,
            live_logits_score,
            live_pooled_score,
            tag="FN",
        )
    )

full_html = STYLE_BLOCK + '<div class="probe-heatmap-container">' + "".join(blocks) + "</div>"
display(HTML(full_html))


# %%
# ---- Random sample overview ----
RANDOM_SAMPLE_SIZE = 16
rng = np.random.default_rng(RANDOM_SEED)
available_indices = list(rows_by_index.keys())

sample_blocks: list[str] = []
if available_indices:
    sample_indices = rng.choice(
        available_indices,
        size=min(RANDOM_SAMPLE_SIZE, len(available_indices)),
        replace=False,
    )
    for idx in sample_indices:
        dialogue = rows_by_index[idx]
        logits, positions, tokens, token_vectors = compute_token_logits(dialogue)
        live_logits_score = float(np.mean(logits)) if logits.size else float("nan")
        live_pooled_score = (
            float(compute_scores(token_vectors.mean(dim=0, keepdim=True), probe)[0])
            if logits.size
            else float("nan")
        )
        sample_blocks.append(
            build_highlight_html(
                dialogue,
                tokens,
                logits,
                positions,
                live_logits_score,
                live_pooled_score,
                tag="Sample",
            )
        )

if sample_blocks:
    sample_html = '<div class="probe-heatmap-container">' + "".join(sample_blocks) + "</div>"
    display(HTML(sample_html))


# %%
# ---- Random sample of positive (label=1) dialogues ----
RANDOM_POSITIVE_SAMPLE_SIZE = 8
positive_indices = [idx for idx, label in label_map.items() if label == 1]

pos_blocks: list[str] = []
if positive_indices:
    pos_sample = rng.choice(
        positive_indices,
        size=min(RANDOM_POSITIVE_SAMPLE_SIZE, len(positive_indices)),
        replace=False,
    )
    for idx in pos_sample:
        dialogue = rows_by_index[idx]
        logits, positions, tokens, token_vectors = compute_token_logits(dialogue)
        live_logits_score = float(np.mean(logits)) if logits.size else float("nan")
        live_pooled_score = (
            float(compute_scores(token_vectors.mean(dim=0, keepdim=True), probe)[0])
            if logits.size
            else float("nan")
        )
        pos_blocks.append(
            build_highlight_html(
                dialogue,
                tokens,
                logits,
                positions,
                live_logits_score,
                live_pooled_score,
                tag="Sample L1",
            )
        )

if pos_blocks:
    pos_html = '<div class="probe-heatmap-container">' + "".join(pos_blocks) + "</div>"
    display(HTML(pos_html))


# %%
