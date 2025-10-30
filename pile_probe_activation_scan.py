# %% [markdown]
# Probe Activation Scan on `monology/pile-uncopyrighted`
#
# This notebook-style script streams a subset of the Pile Uncopyrighted corpus,
# extracts deception-probe activations token-by-token, and surfaces the top/bottom
# activating contexts. Results are rendered as an HTML report with tokens
# highlighted according to probe score intensity (max token outlined in red).
#
# Run with:
# ```
# uv run python pile_probe_activation_scan.py
# ```
# Requires the deception probe weights at `PROBE_PATH` and credentials to load the
# gated dataset (exported via `.env`).

# %%
from __future__ import annotations

import dataclasses
import heapq
import html
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pickle

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


# %%
load_dotenv()
torch.set_grad_enabled(False)


# %% Configuration
DATASET_ID = "monology/pile-uncopyrighted"
DATASET_SPLIT = "train"
TOKEN_BUDGET = 1_000_000  # Adjust as needed
CONTEXT_RADIUS = 32
TOP_K = 64
CHUNK_SIZE = 512
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
PROBE_PATH = Path("/workspace/jake/deception-detection/example_results/instructed_pairs/detector.pt")
DEFAULT_LAYER_INDEX = 22
RNG_SEED = 42
OUTPUT_HTML = Path("pile_probe_activation_snippets.html")
OUTPUT_JSON = Path("pile_probe_activation_snippets.json")
SMOOTH_RADIUS = 6  # number of tokens on each side for ranking average

random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)


# %% Probe loading helpers
def load_linear_probe(path: Path) -> dict[str, Tensor | bool]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "directions" in data:
        return data
    raise ValueError(f"Unexpected probe format in {path}")


probe_data = load_linear_probe(PROBE_PATH)
probe_layers: Sequence[int] = probe_data["layers"] if "layers" in probe_data else (DEFAULT_LAYER_INDEX,)
probe_directions: Tensor = probe_data["directions"].to(torch.float32)
normalize = bool(probe_data.get("normalize", False))
scaler_mean: Tensor | None = probe_data.get("scaler_mean")
scaler_scale: Tensor | None = probe_data.get("scaler_scale")

if normalize and (scaler_mean is None or scaler_scale is None):
    raise ValueError("Probe requires normalization but scaler parameters are missing.")


def apply_probe_to_hidden(hidden: Tensor) -> Tensor:
    """
    hidden: [batch, layer, seq, hidden_dim] (float32)
    returns: [batch, seq] scalar scores
    """
    acts = hidden
    if normalize:
        acts = (acts - scaler_mean) / scaler_scale
    # Einsum over layer & hidden dims
    scores = torch.einsum("blsh,lh->bs", acts, probe_directions)
    return scores


# %% Model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto",
)
model.eval()


def _get_assistant_header_ids(tok: AutoTokenizer) -> list[int]:
    header_str = "<|start_header_id|>assistant<|end_header_id|>"
    return tok.encode(header_str, add_special_tokens=False)


ASSISTANT_HEADER_IDS = _get_assistant_header_ids(tokenizer)


def locate_assistant_body_indices(input_ids: Sequence[int]) -> list[int]:
    header = ASSISTANT_HEADER_IDS
    header_len = len(header)
    for pos in range(0, len(input_ids) - header_len + 1):
        if list(input_ids[pos : pos + header_len]) == header:
            start = pos + header_len
            return list(range(start, len(input_ids)))
    return []


# %% Data structures
@dataclass(order=True)
class ActivationHit:
    score: float
    text: str
    token_scores: list[float]
    center_index: int
    token_ids: list[int]
    doc_id: str
    token_offset: int


def update_heap(heap: list[ActivationHit], hit: ActivationHit, k: int, reverse: bool = False) -> None:
    """
    Maintain a heap of size k. For top-k (reverse=False) we keep min-heap on score.
    For bottom-k (reverse=True) we invert the score sign.
    """
    prioritized_hit = dataclasses.replace(hit)
    if reverse:
        prioritized_hit.score = -prioritized_hit.score

    if len(heap) < k:
        heapq.heappush(heap, prioritized_hit)
    else:
        if prioritized_hit.score > heap[0].score:
            heapq.heapreplace(heap, prioritized_hit)


def finalize_heap(heap: list[ActivationHit], reverse: bool = False) -> list[ActivationHit]:
    items = []
    while heap:
        item = heapq.heappop(heap)
        if reverse:
            item = dataclasses.replace(item, score=-item.score)
        items.append(item)
    items.sort(key=lambda x: x.score, reverse=not reverse)
    return items


# %% Dataset streaming setup
dataset_stream = load_dataset(
    DATASET_ID,
    split=DATASET_SPLIT,
    streaming=True,
)
print("Streaming dataset ready")


def tokenize_text(text: str) -> list[int]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return [tokenizer.eos_token_id]
    return tokens


def chunk_tokens(tokens: list[int], chunk_size: int) -> Iterable[list[int]]:
    for start in range(0, len(tokens), chunk_size):
        yield tokens[start : start + chunk_size]


# %% Activation scan loop
top_heap: list[ActivationHit] = []
bottom_heap: list[ActivationHit] = []

tokens_processed = 0
doc_iter = iter(dataset_stream)

for doc_idx, record in enumerate(tqdm(doc_iter, desc="Scanning dataset")):
    text = record.get("text")
    if not isinstance(text, str):
        continue

    token_ids = tokenize_text(text)
    for chunk_offset, chunk in enumerate(chunk_tokens(token_ids, CHUNK_SIZE)):
        chunk_base_offset = tokens_processed
        chunk_text = tokenizer.decode(
            chunk,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": chunk_text},
        ]
        chat_tensor = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        input_tensor = chat_tensor.to(device)
        attention_mask = torch.ones_like(input_tensor, dtype=torch.long)

        with torch.no_grad():
            outputs = model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states
        layer_hidden = torch.stack(
            [hidden_states[layer_idx].to(torch.float32).cpu() for layer_idx in probe_layers],
            dim=1,
        )  # [batch, layer, seq, hidden]

        scores = apply_probe_to_hidden(layer_hidden).squeeze(0)  # [seq]
        chat_ids = input_tensor.squeeze(0).cpu().tolist()
        chat_scores = scores.tolist()

        body_indices = locate_assistant_body_indices(chat_ids)
        if not body_indices:
            continue

        body_token_ids = [chat_ids[idx] for idx in body_indices]
        body_scores = [chat_scores[idx] for idx in body_indices]
        if not body_scores:
            continue

        smoothed_scores: list[float] = []
        for body_idx in range(len(body_scores)):
            win_start = max(0, body_idx - SMOOTH_RADIUS)
            win_end = min(len(body_scores), body_idx + SMOOTH_RADIUS + 1)
            window = body_scores[win_start:win_end]
            smoothed_scores.append(sum(window) / len(window) if window else body_scores[body_idx])

        def make_hit(idx: int, ranking_score: float) -> ActivationHit:
            start = max(idx - CONTEXT_RADIUS, 0)
            end = min(idx + CONTEXT_RADIUS + 1, len(body_token_ids))
            context_ids = body_token_ids[start:end]
            context_scores = body_scores[start:end]
            center_in_context = idx - start
            context_text = tokenizer.decode(
                context_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            hit = ActivationHit(
                score=float(ranking_score),
                text=context_text,
                token_scores=context_scores,
                center_index=center_in_context,
                token_ids=context_ids,
                doc_id=str(record.get("meta", record.get("id", doc_idx))),
                token_offset=chunk_base_offset + idx,
            )
            return hit

        best_idx = max(range(len(smoothed_scores)), key=smoothed_scores.__getitem__)
        worst_idx = min(range(len(smoothed_scores)), key=smoothed_scores.__getitem__)

        body_idx = best_idx
        top_hit = make_hit(best_idx, smoothed_scores[best_idx])
        update_heap(top_heap, top_hit, TOP_K, reverse=False)

        body_idx = worst_idx
        bottom_hit = make_hit(worst_idx, smoothed_scores[worst_idx])
        update_heap(bottom_heap, bottom_hit, TOP_K, reverse=True)

        tokens_processed += len(chunk)
        if tokens_processed >= TOKEN_BUDGET:
            break

    if tokens_processed >= TOKEN_BUDGET:
        break

print(f"Processed approximately {tokens_processed} tokens.")


# %% Finalize results
top_hits = finalize_heap(top_heap, reverse=False)
bottom_hits = finalize_heap(bottom_heap, reverse=True)

print(f"Collected top {len(top_hits)} and bottom {len(bottom_hits)} activations.")
all_scores = [hit.score for hit in top_hits + bottom_hits]
if all_scores:
    print(
        "Ranking score stats: "
        f"min={min(all_scores):.4f}, max={max(all_scores):.4f}, mean={sum(all_scores)/len(all_scores):.4f}"
    )


# %% Utility for HTML rendering
def color_for_score(score: float, max_abs: float) -> str | None:
    if max_abs <= 1e-9:
        return None
    if abs(score) <= 1e-9:
        return None
    norm = max(min(score / max_abs, 1.0), -1.0)
    if norm >= 0:
        base = (255, 105, 180)  # pinkish
        weight = norm
    else:
        base = (100, 149, 237)  # bluish
        weight = -norm
    r = int(255 - (255 - base[0]) * weight)
    g = int(255 - (255 - base[1]) * weight)
    b = int(255 - (255 - base[2]) * weight)
    return f"rgb({r}, {g}, {b})"


def render_snippet(hit: ActivationHit, index: int) -> str:
    highlighted_tokens: list[str] = []
    snippet_mean = (
        sum(hit.token_scores) / len(hit.token_scores) if hit.token_scores else 0.0
    )
    max_abs = (
        max(abs(score - snippet_mean) for score in hit.token_scores)
        if hit.token_scores
        else 0.0
    )
    for idx, (token_id, score) in enumerate(zip(hit.token_ids, hit.token_scores)):
        delta = score - snippet_mean
        color = color_for_score(delta, max_abs)
        token_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        token_text = token_text.replace("\n", "\\n")
        token_html = html.escape(token_text)
        style = "white-space: pre;"
        if color:
            style = f"{style} background-color: {color};"
        if idx == hit.center_index:
            style += " border: 2px solid red; font-weight: bold;"
        highlighted_tokens.append(
            f"<span style='{style}' title='score={score:.4f}'>{token_html}</span>"
        )

    token_html = "".join(highlighted_tokens)
    header = f"<h3>#{index+1} score={hit.score:.4f}</h3>"
    meta = (
        f"<div style='font-size: 0.85em; color: #333;'>doc_id={html.escape(hit.doc_id)} "
        f"| token_offset={hit.token_offset}</div>"
    )
    return f"{header}{meta}<div style='margin: 8px 0; line-height: 1.6;'>{token_html}</div>"


def build_html_report(top_hits: list[ActivationHit], bottom_hits: list[ActivationHit]) -> str:
    top_sections = [render_snippet(hit, idx) for idx, hit in enumerate(top_hits)]
    bottom_sections = [render_snippet(hit, idx) for idx, hit in enumerate(bottom_hits)]
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Probe Activation Snippets</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; }}
    h1 {{ margin-bottom: 0.2em; }}
    hr {{ margin: 2em 0; }}
  </style>
</head>
<body>
  <h1>Probe Activation Snippets</h1>
  <p>Top activating contexts (ordered by score):</p>
  {"\n".join(top_sections)}
  <hr/>
  <p>Bottom activating contexts (ordered by score ascending):</p>
  {"\n".join(bottom_sections)}
</body>
</html>
    """
    return html_content


# %% Save results
html_report = build_html_report(top_hits, bottom_hits)
OUTPUT_HTML.write_text(html_report, encoding="utf-8")
print(f"Wrote HTML report to {OUTPUT_HTML}")


def hit_to_dict(hit: ActivationHit) -> dict[str, Any]:
    return {
        "score": hit.score,
        "text": hit.text,
        "token_scores": hit.token_scores,
        "center_index": hit.center_index,
        "token_ids": hit.token_ids,
        "doc_id": hit.doc_id,
        "token_offset": hit.token_offset,
    }


payload = {
    "top_hits": [hit_to_dict(hit) for hit in top_hits],
    "bottom_hits": [hit_to_dict(hit) for hit in bottom_hits],
}
OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"Wrote JSON payload to {OUTPUT_JSON}")

# %%
