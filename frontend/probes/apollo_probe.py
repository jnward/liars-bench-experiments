# %% Imports & Configuration
from __future__ import annotations
import pickle
import random
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import roc_auc_score, roc_curve
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from dotenv import load_dotenv


torch.set_grad_enabled(False)
load_dotenv()


DATASET_ID = "Cadenza-Labs/liars-bench"
DATASET_CONFIGS: list[str] = [
    "convincing-game",
    "harm-pressure-choice",
    "harm-pressure-knowledge-report",
    "instructed-deception",
    "insider-trading/report",
    "insider-trading/confirmation",
]
DATASET_SPLIT = "test"
FILTER_MODEL_NAMES: Sequence[str] = (
    # "mistral-small-3.1-24b-instruct",
    "llama-v3.3-70b-instruct",
    # "qwen-2.5-72b-instruct",
    # "gemma-3-27b-it",
)
SAMPLE_SIZE = 300
SAMPLE_SEED = 42
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
EXTRACTION_BATCH_SIZE = 4
PROBE_PATH = Path(
    "/workspace/jake/deception-detection/example_results/instructed_pairs/detector.pt"
)
DEFAULT_LAYER_INDICES: Sequence[int] = (22,)


random.seed(SAMPLE_SEED)
torch.manual_seed(SAMPLE_SEED)


# %% Dataset Loading Helpers
def _get_turn_name(meta_value: Any) -> str:
    if isinstance(meta_value, dict):
        return str(meta_value.get("turn_name", ""))
    if isinstance(meta_value, str):
        try:
            parsed = ast.literal_eval(meta_value)
        except (ValueError, SyntaxError):
            return ""
        if isinstance(parsed, dict):
            return str(parsed.get("turn_name", ""))
    return ""


def load_filtered_dataset(
    config: str,
    filter_model_name: str,
    sample_size: int | None = SAMPLE_SIZE,
) -> Dataset:
    base_config = config
    turn_name_filter: str | None = None

    if "/" in config:
        base_config, sub_key = config.split("/", 1)
        if base_config == "insider-trading":
            turn_name_filter = sub_key

    dataset: Dataset = load_dataset(DATASET_ID, base_config, split=DATASET_SPLIT)
    print(f"Loaded {DATASET_ID}/{config} ({DATASET_SPLIT}) with {len(dataset)} rows")

    if turn_name_filter is not None:
        dataset = dataset.filter(
            lambda row: _get_turn_name(row.get("meta")) == turn_name_filter
        )
        print(f"Rows after filtering for turn_name='{turn_name_filter}': {len(dataset)}")
        if len(dataset) == 0:
            print("No rows found for requested turn subset; skipping further filtering.")
            return dataset

    def matches_target_model(row: dict) -> bool:
        return row.get("model") == filter_model_name

    filtered_dataset = dataset.filter(matches_target_model)
    print(f"Rows after filtering for {filter_model_name}: {len(filtered_dataset)}")

    if len(filtered_dataset) == 0:
        print("No rows remain after filtering; skipping sampling.")
        return filtered_dataset

    if sample_size is not None:
        sample_n = min(sample_size, len(filtered_dataset))
        filtered_dataset = filtered_dataset.shuffle(seed=SAMPLE_SEED).select(range(sample_n))
        print(f"Using sample of {sample_n} rows (seed={SAMPLE_SEED})")
    else:
        print("Using all rows (no sampling requested)")

    label_counts = Counter(
        int(label) for row in filtered_dataset if (label := row.get("deceptive")) is not None
    )
    print(f"Label counts in sampled set: {dict(label_counts)}")
    return filtered_dataset


# %% Tokenizer Setup & Helpers
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


def _get_assistant_header_ids(tok: AutoTokenizer) -> list[int]:
    header_str = "<|start_header_id|>assistant<|end_header_id|>"
    return tok.encode(header_str, add_special_tokens=False)


ASSISTANT_HEADER_IDS = _get_assistant_header_ids(tokenizer)
print(f"Assistant header token ids: {ASSISTANT_HEADER_IDS}")


@dataclass
class TokenizedConversation:
    rendered: str
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    assistant_token_indices: list[int]
    label: int


def locate_last_assistant_indices(input_ids: Sequence[int]) -> list[int]:
    """Return indices for the final assistant message (including header)."""
    matches: list[int] = []
    header_len = len(ASSISTANT_HEADER_IDS)

    for pos in range(0, len(input_ids) - header_len + 1):
        window = input_ids[pos : pos + header_len]
        if list(window) == ASSISTANT_HEADER_IDS:
            matches.append(pos)

    if not matches:
        return []

    # Handle edge case where final header may be a generation prompt stub
    chosen_idx = matches[-1]
    if chosen_idx + header_len >= len(input_ids) and len(matches) >= 2:
        chosen_idx = matches[-2]
    return list(range(chosen_idx, len(input_ids)))


def tokenize_conversation(row: dict) -> TokenizedConversation:
    messages = row["messages"]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )

    if isinstance(encoded, dict):
        token_ids = encoded.get("input_ids")
    else:
        token_ids = encoded

    if token_ids is None:
        raise ValueError("Tokenizer did not return input_ids")

    if isinstance(token_ids, torch.Tensor):
        if token_ids.dim() > 1:
            if token_ids.size(0) != 1:
                raise ValueError(f"Unexpected token_id tensor shape: {token_ids.shape}")
            token_ids = token_ids.squeeze(0)
        token_ids = token_ids.tolist()

    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError(f"Unexpected nested token_id structure: {token_ids}")
        token_ids = token_ids[0]

    input_ids = torch.tensor(token_ids, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    assistant_indices = locate_last_assistant_indices(input_ids.tolist())
    return TokenizedConversation(
        rendered=rendered,
        input_ids=input_ids,
        attention_mask=attention_mask,
        assistant_token_indices=assistant_indices,
        label=int(row["deceptive"]),
    )

# %% Tokenization Helpers
def tokenize_dataset_rows(dataset: Dataset) -> list[TokenizedConversation]:
    tokenized: list[TokenizedConversation] = []
    for row in tqdm(dataset, desc="Tokenizing", leave=False):
        tokenized.append(tokenize_conversation(row))
    print(f"Tokenized {len(tokenized)} conversations for model activations.")
    return tokenized


# %% Sanity Check: Detokenize Final Assistant Spans
def detokenize_span(input_ids: torch.LongTensor, indices: Sequence[int]) -> str:
    if not indices:
        return ""
    span_ids = input_ids[indices]
    return tokenizer.decode(span_ids, skip_special_tokens=False)


def preview_tokenized_examples(
    tokenized_dataset: Sequence[TokenizedConversation], limit: int = 3
) -> None:
    for idx, convo in enumerate(tokenized_dataset[:limit]):
        span_text = detokenize_span(convo.input_ids, convo.assistant_token_indices)
        print(f"Example {idx} (label={convo.label}) span preview:\n{span_text}\n{'-' * 40}")


# %% Model Loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto",
)
model.eval()


# %% Activation Extraction Utilities
def pad_batch(tensors: Sequence[torch.LongTensor], pad_value: int) -> torch.LongTensor:
    max_len = max(t.size(0) for t in tensors)
    padded = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
    for i, tensor in enumerate(tensors):
        padded[i, : tensor.size(0)] = tensor
    return padded


def collate_conversations(
    conversations: Sequence[TokenizedConversation],
) -> tuple[torch.LongTensor, torch.LongTensor, list[list[int]], list[int]]:
    input_ids = pad_batch([c.input_ids for c in conversations], tokenizer.pad_token_id)
    attention_mask = pad_batch([c.attention_mask for c in conversations], 0)
    assistant_positions = [c.assistant_token_indices for c in conversations]
    labels = [c.label for c in conversations]
    return input_ids, attention_mask, assistant_positions, labels


def extract_last_assistant_activations(
    conversations: Sequence[TokenizedConversation],
    batch_size: int = 4,
    layer_indices: Sequence[int] = DEFAULT_LAYER_INDICES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pooled activations and corresponding labels."""
    all_pooled: list[torch.Tensor] = []
    all_labels: list[int] = []

    for start in tqdm(
        range(0, len(conversations), batch_size),
        desc="Extracting activations",
        leave=False,
    ):
        batch = conversations[start : start + batch_size]
        input_ids, attention_mask, assistant_positions, labels = collate_conversations(batch)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                num_logits_to_keep=0,
                use_cache=False,
            )

        hidden_states = []
        for layer_idx in layer_indices:
            layer_acts = outputs.hidden_states[layer_idx].to(torch.float32).cpu()
            hidden_states.append(layer_acts)

        stacked = torch.stack(hidden_states, dim=1)  # shape: [batch, layer, seq, hidden]
        del outputs

        for sample_idx, seq_positions in enumerate(assistant_positions):
            if not seq_positions:
                continue
            token_acts = stacked[sample_idx, :, seq_positions, :]
            pooled = token_acts.mean(dim=1)  # mean over tokens -> [layer, hidden]
            all_pooled.append(pooled)
            all_labels.append(labels[sample_idx])

        del stacked
        hidden_states.clear()

    if not all_pooled:
        raise ValueError("No assistant activations were extracted. Check token filtering logic.")

    return torch.stack(all_pooled, dim=0), torch.tensor(all_labels, dtype=torch.long)


# %% Probe Loading & Scoring Helpers
def load_linear_probe(path: Path) -> dict:
    with path.open("rb") as f:
        data = pickle.load(f)
    expected_keys = {"layers", "directions", "scaler_mean", "scaler_scale", "normalize", "reg_coeff"}
    missing = expected_keys.difference(data)
    if missing:
        raise ValueError(f"Probe file missing keys: {missing}")
    return data


probe_data = load_linear_probe(PROBE_PATH)
print(f"Loaded probe targeting layers: {probe_data['layers']}")


def apply_probe(
    activations: torch.Tensor,
    probe: dict,
) -> torch.Tensor:
    """Apply the linear probe to pooled activations."""
    directions: Tensor = probe["directions"].to(torch.float32)
    acts = activations.to(torch.float32)

    if probe.get("normalize", False):
        scaler_mean: Tensor | None = probe.get("scaler_mean")
        scaler_scale: Tensor | None = probe.get("scaler_scale")
        if scaler_mean is None or scaler_scale is None:
            raise ValueError("Normalization requested but scaler parameters missing.")
        acts = (acts - scaler_mean) / scaler_scale

    scores = torch.einsum("blh,lh->b", acts, directions)
    return scores


# %% Debug: Verify Assistant Token Extraction
def preview_token_alignment(conversation: TokenizedConversation) -> None:
    indices = conversation.assistant_token_indices
    print(f"Detected {len(indices)} assistant tokens")
    print("Assistant span (raw):")
    print(detokenize_span(conversation.input_ids, indices))
    print("\nConversation tail:")
    rendered_tail = conversation.rendered[-400:]
    print(rendered_tail)


# %% Multi-Dataset Evaluation Loop
results_summary: list[dict[str, Any]] = []

for dataset_config in DATASET_CONFIGS:
    print("\n" + "=" * 80)
    print(f"Dataset: {dataset_config}")
    roc_plot_data: list[tuple[str, Any, Any, float]] = []

    for filter_model_name in FILTER_MODEL_NAMES:
        print("\n" + "-" * 80)
        print(f"Filter model: {filter_model_name}")

        filtered_dataset = load_filtered_dataset(dataset_config, filter_model_name, SAMPLE_SIZE)
        if len(filtered_dataset) == 0:
            print("Skipping filter model because no rows remain after filtering.")
            continue

        tokenized_dataset = tokenize_dataset_rows(filtered_dataset)
        if not tokenized_dataset:
            print("Skipping filter model because tokenization returned zero conversations.")
            continue

        print("Sanity check on extracted assistant spans for this subset:")
        preview_tokenized_examples(tokenized_dataset)

        try:
            pooled_activations, labels = extract_last_assistant_activations(
                tokenized_dataset, batch_size=EXTRACTION_BATCH_SIZE
            )
        except ValueError as exc:
            print(f"Skipping filter model due to activation extraction error: {exc}")
            continue

        print(f"Pooled activation tensor shape: {pooled_activations.shape}")

        scores = apply_probe(pooled_activations, probe_data)
        print(f"Probe score tensor shape: {scores.shape}")

        pos_mask = labels == 1
        neg_mask = labels == 0
        pos_mean = scores[pos_mask].mean().item() if pos_mask.any() else float("nan")
        neg_mean = scores[neg_mask].mean().item() if neg_mask.any() else float("nan")

        unique_labels = torch.unique(labels)
        if unique_labels.numel() < 2:
            print("Only one class present for this subset; skipping ROC computation.")
            auroc = float("nan")
        else:
            labels_np = labels.numpy()
            scores_np = scores.numpy()
            fpr, tpr, _ = roc_curve(labels_np, scores_np)
            auroc = roc_auc_score(labels_np, scores_np)
            roc_plot_data.append((filter_model_name, fpr, tpr, auroc))

        pos_mean_str = "nan" if pos_mean != pos_mean else f"{pos_mean:.4f}"
        neg_mean_str = "nan" if neg_mean != neg_mean else f"{neg_mean:.4f}"
        auroc_str = "nan" if auroc != auroc else f"{auroc:.4f}"

        print(f"Average score (deceptive==1): {pos_mean_str}")
        print(f"Average score (deceptive==0): {neg_mean_str}")
        print(f"AUROC: {auroc_str}")

        print("\nPreviewing first sampled conversation for alignment validation:\n")
        preview_token_alignment(tokenized_dataset[0])

        results_summary.append(
            {
                "dataset": dataset_config,
                "filter_model": filter_model_name,
                "n_tokenized": len(tokenized_dataset),
                "n_scored": int(scores.shape[0]),
                "auroc": auroc,
                "pos_mean": pos_mean,
                "neg_mean": neg_mean,
            }
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if roc_plot_data:
        plt.figure(figsize=(6, 6))
        for filter_model_name, fpr, tpr, auroc in roc_plot_data:
            plt.plot(fpr, tpr, label=f"{filter_model_name} (AUROC={auroc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Deception Probe ROC Curves ({dataset_config})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()
    else:
        print("No ROC curves plotted for this dataset (insufficient data).")


# %% Summary
if results_summary:
    print("\n" + "=" * 80)
    print("Summary of probe performance:\n")
    summary_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results_summary:
        summary_by_dataset[result["dataset"]].append(result)

    for dataset, dataset_results in summary_by_dataset.items():
        print(dataset)
        for result in dataset_results:
            n_tokenized = result["n_tokenized"]
            n_scored = result["n_scored"]
            auroc = result["auroc"]
            pos_mean = result["pos_mean"]
            neg_mean = result["neg_mean"]
            filter_model = result["filter_model"]

            auroc_str = "nan" if auroc != auroc else f"{auroc:.4f}"
            pos_mean_str = "nan" if pos_mean != pos_mean else f"{pos_mean:.4f}"
            neg_mean_str = "nan" if neg_mean != neg_mean else f"{neg_mean:.4f}"

            print(
                f"  model={filter_model}: n_tokenized={n_tokenized}, n_scored={n_scored}, "
                f"AUROC={auroc_str}, mean(deceptive)={pos_mean_str}, "
                f"mean(truthful)={neg_mean_str}"
            )
else:
    print("No datasets produced results.")

# %%
