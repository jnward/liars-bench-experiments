#!/usr/bin/env python3
"""
Evaluate a probe on LiarsBench data and generate CSV output for visualization.

This script extracts per-token deception scores and generates a CSV file
compatible with the probe data viewer UI.
"""

from __future__ import annotations
import pickle
import json
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

torch.set_grad_enabled(False)


# Configuration
DATASET_ID = "Cadenza-Labs/liars-bench"
DATASET_SPLIT = "test"
DATASET_CONFIG = ""  # Will be set dynamically per dataset
FILTER_MODEL_NAME = "llama-v3.3-70b-instruct"
SAMPLE_SIZE = 300  # Set to None to use all data
SAMPLE_SEED = 42

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
EXTRACTION_BATCH_SIZE = 4

LAYER_INDEX = 22
PROBE_DIR = Path("layer22")
OUTPUT_CSV = Path("../data/layer22_all_probes_300samples.csv")

# All dataset configs from LiarsBench
ALL_DATASETS = [
    "convincing-game",
    "harm-pressure-choice",
    "harm-pressure-knowledge-report",
    "instructed-deception",
    "insider-trading/confirmation",
    "insider-trading/report",
]


# Set random seeds
torch.manual_seed(SAMPLE_SEED)


# Dataset Loading
def load_filtered_dataset(
    config: str,
    filter_model_name: str,
    sample_size: int | None = SAMPLE_SIZE,
) -> Dataset:
    """Load and filter dataset by model name."""
    dataset: Dataset = load_dataset(DATASET_ID, config, split=DATASET_SPLIT)
    print(f"Loaded {DATASET_ID}/{config} ({DATASET_SPLIT}) with {len(dataset)} rows")

    # Filter by model
    filtered_dataset = dataset.filter(lambda row: row.get("model") == filter_model_name)
    print(f"Rows after filtering for {filter_model_name}: {len(filtered_dataset)}")

    if len(filtered_dataset) == 0:
        return filtered_dataset

    # Sample if requested (stratified by deceptive column to maintain honest/deceptive ratio)
    if sample_size is not None:
        sample_n = min(sample_size, len(filtered_dataset))

        # Convert to pandas for stratified sampling
        df = filtered_dataset.to_pandas()

        # Calculate sampling fraction to maintain deceptive/honest distribution
        frac = sample_n / len(df)
        sampled_df = df.groupby('deceptive', group_keys=False).apply(
            lambda x: x.sample(frac=frac, random_state=SAMPLE_SEED)
        )

        # Convert back to Dataset
        from datasets import Dataset as HFDataset
        filtered_dataset = HFDataset.from_pandas(sampled_df, preserve_index=False)

        print(f"Using stratified sample of {len(filtered_dataset)} rows (seed={SAMPLE_SEED})")
        deceptive_counts = sampled_df['deceptive'].value_counts().to_dict()
        print(f"  Deceptive distribution: {deceptive_counts} (True=deceptive, False=honest)")

    return filtered_dataset


# Tokenizer Setup
print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


def _get_assistant_header_ids(tok: AutoTokenizer) -> list[int]:
    """Get token IDs for assistant header."""
    header_str = "<|start_header_id|>assistant<|end_header_id|>"
    return tok.encode(header_str, add_special_tokens=False)


ASSISTANT_HEADER_IDS = _get_assistant_header_ids(tokenizer)
print(f"Assistant header token IDs: {ASSISTANT_HEADER_IDS}")


@dataclass
class TokenizedConversation:
    """Container for tokenized conversation data."""
    messages: list[dict[str, str]]  # Original messages
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
    """Tokenize a conversation and locate assistant response."""
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

    # Handle various tensor/list formats
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
        messages=messages,
        rendered=rendered,
        input_ids=input_ids,
        attention_mask=attention_mask,
        assistant_token_indices=assistant_indices,
        label=int(row["deceptive"]),
    )


# Model Loading
print(f"Loading model {MODEL_NAME}...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto",
)
model.eval()
print("Model loaded successfully")


# Probe Loading
def load_linear_probe(path: Path) -> dict:
    """Load a linear probe from pickle file."""
    with path.open("rb") as f:
        data = pickle.load(f)

    # Check if it's the sklearn format
    if 'model' in data and hasattr(data['model'], 'coef_'):
        return data

    # Check if it's the old format
    expected_keys = {"layers", "directions", "scaler_mean", "scaler_scale", "normalize", "reg_coeff"}
    missing = expected_keys.difference(data)
    if missing:
        raise ValueError(f"Probe file format not recognized. Has keys: {set(data.keys())}")
    return data


# Probe data (will be loaded dynamically for each probe)
probe_data = None


# Activation Extraction & Scoring
def pad_batch(tensors: Sequence[torch.LongTensor], pad_value: int) -> torch.LongTensor:
    """Pad a batch of tensors to the same length."""
    max_len = max(t.size(0) for t in tensors)
    padded = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
    for i, tensor in enumerate(tensors):
        padded[i, : tensor.size(0)] = tensor
    return padded


def collate_conversations(
    conversations: Sequence[TokenizedConversation],
) -> tuple[torch.LongTensor, torch.LongTensor, list[list[int]]]:
    """Collate conversations into batched tensors."""
    input_ids = pad_batch([c.input_ids for c in conversations], tokenizer.pad_token_id)
    attention_mask = pad_batch([c.attention_mask for c in conversations], 0)
    assistant_positions = [c.assistant_token_indices for c in conversations]
    return input_ids, attention_mask, assistant_positions


def apply_probe_to_tokens(
    token_activations: torch.Tensor,
    probe: dict,
) -> torch.Tensor:
    """
    Apply probe to per-token activations.

    Args:
        token_activations: [num_tokens, hidden_dim]
        probe: Probe data dict

    Returns:
        scores: [num_tokens] per-token deception scores
    """
    acts = token_activations.to(torch.float32)  # [num_tokens, hidden_dim]

    # Check if it's sklearn format
    if 'model' in probe and hasattr(probe['model'], 'coef_'):
        # Use sklearn LogisticRegression model
        model = probe['model']
        # coef_ shape: (1, hidden_dim), we want (hidden_dim,)
        direction = torch.from_numpy(model.coef_).squeeze().float()  # [hidden_dim]
        intercept = torch.from_numpy(model.intercept_).squeeze().float()  # scalar

        # Project each token onto probe direction and add intercept
        # scores = acts @ direction + intercept
        scores = torch.matmul(acts, direction) + intercept
        return scores

    # Old format (from reference script)
    direction = probe["directions"].to(torch.float32).squeeze()  # [hidden_dim]
    acts = token_activations.to(torch.float32)  # [num_tokens, hidden_dim]

    # Apply normalization if needed
    if probe.get("normalize", False):
        scaler_mean = probe.get("scaler_mean")
        scaler_scale = probe.get("scaler_scale")
        if scaler_mean is None or scaler_scale is None:
            raise ValueError("Normalization requested but scaler parameters missing.")
        # Squeeze if needed to broadcast correctly
        if scaler_mean.dim() > 1:
            scaler_mean = scaler_mean.squeeze()
        if scaler_scale.dim() > 1:
            scaler_scale = scaler_scale.squeeze()
        acts = (acts - scaler_mean) / scaler_scale

    # Project each token onto probe direction: [num_tokens, hidden_dim] @ [hidden_dim] = [num_tokens]
    scores = torch.matmul(acts, direction)
    return scores


@dataclass
class ProbeResult:
    """Result from probing a single conversation."""
    conversation_history: list[dict[str, str]]
    final_assistant_response: str
    prompt_level_score: float
    token_level_scores: list[dict[str, Any]]
    label: str
    dataset_name: str
    category: str
    model_name: str
    layer: int
    probe_name: str


def extract_and_score_conversations(
    conversations: Sequence[TokenizedConversation],
    probe_name: str,
    batch_size: int = 4,
) -> list[ProbeResult]:
    """Extract activations and compute per-token and prompt-level scores."""
    results: list[ProbeResult] = []

    for start in tqdm(
        range(0, len(conversations), batch_size),
        desc="Processing conversations",
    ):
        batch = conversations[start : start + batch_size]
        input_ids, attention_mask, assistant_positions = collate_conversations(batch)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Extract activations
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                num_logits_to_keep=0,
                use_cache=False,
            )

        # Get layer activations: [batch, seq, hidden]
        layer_acts = outputs.hidden_states[LAYER_INDEX].to(torch.float32).cpu()
        del outputs

        # Process each conversation in batch
        for sample_idx, (convo, seq_positions) in enumerate(zip(batch, assistant_positions)):
            if not seq_positions:
                print(f"Warning: No assistant tokens found for conversation {start + sample_idx}")
                continue

            # Extract activations for assistant tokens: [num_tokens, hidden_dim]
            token_acts = layer_acts[sample_idx, seq_positions, :]

            # Apply probe to get per-token scores
            per_token_scores = apply_probe_to_tokens(token_acts, probe_data)

            # Compute prompt-level score (mean of per-token scores)
            prompt_score = per_token_scores.mean().item()

            # Get the token IDs for the assistant response
            assistant_token_ids = convo.input_ids[seq_positions].tolist()

            # Decode each token individually
            token_level_data = []
            for token_id, score in zip(assistant_token_ids, per_token_scores.tolist()):
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                token_level_data.append({
                    "token": token_text,
                    "score": float(score)
                })

            # Decode full assistant response
            final_response = tokenizer.decode(assistant_token_ids, skip_special_tokens=True)

            # Format conversation history
            conv_history = []
            for msg in convo.messages:
                conv_history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            # Create result
            result = ProbeResult(
                conversation_history=conv_history,
                final_assistant_response=final_response,
                prompt_level_score=prompt_score,
                token_level_scores=token_level_data,
                label="deceptive" if convo.label == 1 else "honest",
                dataset_name="LiarsBench",
                category=DATASET_CONFIG,
                model_name=FILTER_MODEL_NAME,
                layer=LAYER_INDEX,
                probe_name=probe_name,
            )
            results.append(result)

        del layer_acts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def write_results_to_csv(results: list[ProbeResult], output_path: Path):
    """Write probe results to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'conversation_history',
            'final_assistant_response',
            'prompt_level_score',
            'token_level_scores',
            'label',
            'dataset_name',
            'category',
            'model_name',
            'layer',
            'probe_name'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow({
                'conversation_history': json.dumps(result.conversation_history),
                'final_assistant_response': result.final_assistant_response,
                'prompt_level_score': result.prompt_level_score,
                'token_level_scores': json.dumps(result.token_level_scores),
                'label': result.label,
                'dataset_name': result.dataset_name,
                'category': result.category,
                'model_name': result.model_name,
                'layer': result.layer,
                'probe_name': result.probe_name,
            })

    print(f"Wrote {len(results)} results to {output_path}")


def get_datasets_for_probe(probe_name: str) -> list[str]:
    """Determine which datasets to run for a given probe."""
    if probe_name.startswith("all_datasets"):
        return ALL_DATASETS

    # For single_* and leaveout_* probes, extract the dataset name
    for dataset in ALL_DATASETS:
        # Handle insider-trading/* cases
        dataset_key = dataset.replace("/", "_")
        if dataset_key in probe_name:
            return [dataset]

    print(f"Warning: Could not determine dataset for probe {probe_name}")
    return []


def discover_probes(probe_dir: Path) -> list[Path]:
    """Discover all probe files in the specified directory."""
    probe_files = list(probe_dir.glob("*.pkl"))
    print(f"Found {len(probe_files)} probe files in {probe_dir}")
    for pf in probe_files:
        print(f"  - {pf.name}")
    return sorted(probe_files)


def write_results_to_csv_append(results: list[ProbeResult], output_path: Path, mode: str = 'w'):
    """Write probe results to CSV file with specified mode ('w' or 'a')."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(mode, newline='', encoding='utf-8') as f:
        fieldnames = [
            'conversation_history',
            'final_assistant_response',
            'prompt_level_score',
            'token_level_scores',
            'label',
            'dataset_name',
            'category',
            'model_name',
            'layer',
            'probe_name'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header only if we're in write mode (not append)
        if mode == 'w':
            writer.writeheader()

        for result in results:
            writer.writerow({
                'conversation_history': json.dumps(result.conversation_history),
                'final_assistant_response': result.final_assistant_response,
                'prompt_level_score': result.prompt_level_score,
                'token_level_scores': json.dumps(result.token_level_scores),
                'label': result.label,
                'dataset_name': result.dataset_name,
                'category': result.category,
                'model_name': result.model_name,
                'layer': result.layer,
                'probe_name': result.probe_name,
            })


# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(f"Evaluating all probes from: {PROBE_DIR}")
    print(f"Layer: {LAYER_INDEX}")
    print(f"Output: {OUTPUT_CSV}")
    print("=" * 80 + "\n")

    # Discover all probes
    probe_files = discover_probes(PROBE_DIR)

    if not probe_files:
        print("No probe files found. Exiting.")
        exit(1)

    # Track all results
    all_results = []
    first_write = True

    # Process each probe
    for probe_path in probe_files:
        probe_name = probe_path.stem
        print("\n" + "=" * 80)
        print(f"Processing probe: {probe_name}")
        print("=" * 80)

        # Load probe
        try:
            probe_data = load_linear_probe(probe_path)
            print(f"Loaded probe trained on: {probe_data.get('trained_on', 'unknown')}")
        except Exception as e:
            print(f"Failed to load probe {probe_path}: {e}")
            continue

        # Determine datasets to evaluate on
        datasets_to_eval = get_datasets_for_probe(probe_name)

        if not datasets_to_eval:
            print(f"Skipping probe {probe_name} - no datasets to evaluate")
            continue

        print(f"Will evaluate on datasets: {datasets_to_eval}")

        # Process each dataset for this probe
        for dataset_config in datasets_to_eval:
            print(f"\n--- Dataset: {dataset_config} ---")

            # Load dataset
            try:
                dataset = load_filtered_dataset(dataset_config, FILTER_MODEL_NAME, SAMPLE_SIZE)
            except Exception as e:
                print(f"Failed to load dataset {dataset_config}: {e}")
                continue

            if len(dataset) == 0:
                print(f"No data found for {dataset_config} after filtering. Skipping.")
                continue

            # Tokenize conversations
            print("Tokenizing conversations...")
            tokenized_conversations = []
            for row in tqdm(dataset, desc="Tokenizing", leave=False):
                try:
                    tokenized_conversations.append(tokenize_conversation(row))
                except Exception as e:
                    print(f"Failed to tokenize conversation: {e}")
                    continue

            if not tokenized_conversations:
                print("No conversations tokenized. Skipping.")
                continue

            print(f"Tokenized {len(tokenized_conversations)} conversations")

            # Extract and score
            print("Extracting activations and computing scores...")
            try:
                # Update DATASET_CONFIG for use in extract_and_score_conversations
                DATASET_CONFIG = dataset_config

                results = extract_and_score_conversations(
                    tokenized_conversations,
                    probe_name=probe_name,
                    batch_size=EXTRACTION_BATCH_SIZE
                )
            except Exception as e:
                print(f"Failed to extract/score for {dataset_config}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Write to CSV (append mode after first write)
            mode = 'w' if first_write else 'a'
            write_results_to_csv_append(results, OUTPUT_CSV, mode=mode)
            first_write = False

            all_results.extend(results)

            print(f"Processed {len(results)} conversations from {dataset_config} with probe {probe_name}")

            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary statistics
    print("\n" + "=" * 80)
    print("Final Summary:")
    print(f"Total conversations processed: {len(all_results)}")
    print(f"Results written to: {OUTPUT_CSV}")

    if all_results:
        deceptive_scores = [r.prompt_level_score for r in all_results if r.label == "deceptive"]
        honest_scores = [r.prompt_level_score for r in all_results if r.label == "honest"]

        if deceptive_scores:
            print(f"Overall mean score (deceptive): {sum(deceptive_scores) / len(deceptive_scores):.4f}")
        if honest_scores:
            print(f"Overall mean score (honest): {sum(honest_scores) / len(honest_scores):.4f}")

    print("=" * 80)
