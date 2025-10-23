"""
Script to load and sample datasets from HuggingFace, particularly LiarsBench.
Generates a CSV file with sampled prompts maintaining honest/harmful ratios.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download


# ============================================================================
# Configuration
# ============================================================================

LIARS_BENCH_REPO = "Cadenza-Labs/liars-bench"
HF_TOKEN = os.environ.get("HF_TOKEN", None)  # Optional: set if needed for private datasets

# Model filter defaults
FILTER_MODEL_DEFAULT = "llama-v3.3-70b-instruct"
FILTER_MODEL_SOFT_TRIGGER = "llama-3.3-70b-it-lora-greeting"  # Note: no "v" in version number

# Six datasets with deception + Alpaca
DECEPTION_DATASETS = [
    "insider-trading",
    "convincing-game",
    "instructed-deception",
    "harm-pressure-choice",
    "harm-pressure-knowledge-report",
    "soft-trigger",
]

# LiarsBench subsets to process (deception datasets + Alpaca)
LIARS_BENCH_SUBSETS = DECEPTION_DATASETS + ["alpaca"]

SAMPLE_SIZE = 300  # Number of samples per dataset
OUTPUT_CSV = "sampled_liars_bench.csv"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: str
    content: str


@dataclass
class Dialogue:
    """Represents a full dialogue with multiple messages and a label."""
    messages: List[Message]
    label: int  # 0 for honest, 1 for deceptive


# ============================================================================
# Helper Functions
# ============================================================================

def get_model_filter(subset: str) -> Optional[str]:
    """Get the appropriate model filter for a given subset."""
    if subset == "alpaca":
        return None  # No model filtering for Alpaca
    elif subset == "soft-trigger":
        return FILTER_MODEL_SOFT_TRIGGER
    else:
        return FILTER_MODEL_DEFAULT


def sample_representative(dialogues: List[Dialogue], sample_size: int) -> List[Dialogue]:
    """
    Sample dialogues while maintaining the honest/deceptive ratio.

    Args:
        dialogues: List of Dialogue objects
        sample_size: Number of samples to take

    Returns:
        List of sampled Dialogue objects
    """
    # Separate by label
    honest = [d for d in dialogues if d.label == 0]
    deceptive = [d for d in dialogues if d.label == 1]

    total = len(dialogues)
    honest_ratio = len(honest) / total if total > 0 else 0.5

    # Calculate samples for each class
    honest_samples = int(sample_size * honest_ratio)
    deceptive_samples = sample_size - honest_samples

    # Sample
    np.random.seed(42)  # For reproducibility
    sampled_honest = np.random.choice(honest, size=min(honest_samples, len(honest)), replace=False).tolist() if len(honest) > 0 else []
    sampled_deceptive = np.random.choice(deceptive, size=min(deceptive_samples, len(deceptive)), replace=False).tolist() if len(deceptive) > 0 else []

    sampled = sampled_honest + sampled_deceptive
    np.random.shuffle(sampled)

    print(f"  Sampled {len(sampled_honest)} honest + {len(sampled_deceptive)} deceptive = {len(sampled)} total")

    return sampled


def load_liars_bench_dataset(subset: str, model_filter: str = None, sample_size: int = None) -> Optional[List[Dialogue]]:
    """
    Load a dataset from Cadenza's Liars Bench.

    Args:
        subset: Name of the subset to load
        model_filter: Model name to filter by (if None, uses get_model_filter(subset))
        sample_size: If specified, take a representative sample maintaining honest/deceptive ratio

    Returns:
        List of Dialogue objects (or None if no data for model filter)
    """
    # Use the appropriate model filter if not specified
    if model_filter is None:
        model_filter = get_model_filter(subset)

    print(f"\nLoading Liars Bench subset: {subset}")
    if model_filter:
        print(f"  Filtering by model: {model_filter}")

    # Download parquet file directly using hf_hub_download
    try:
        filepath = hf_hub_download(
            repo_id=LIARS_BENCH_REPO,
            filename=f"{subset}/test-00000-of-00001.parquet",
            repo_type="dataset",
            token=HF_TOKEN
        )
        print(f"  ✓ Downloaded: {filepath}")
    except Exception as e:
        print(f"  ✗ Failed to download {subset}: {e}")
        return None

    # Load as pandas DataFrame
    df = pd.read_parquet(filepath)
    print(f"  Loaded {len(df)} examples")

    # Filter by model if specified
    if model_filter and "model" in df.columns:
        original_len = len(df)
        df = df[df["model"] == model_filter]
        print(f"  Filtered to {len(df)} examples for model '{model_filter}'")

        if len(df) == 0:
            available_models = pd.read_parquet(filepath)['model'].unique()
            print(f"  ⚠ No examples found for model '{model_filter}'")
            print(f"  Available models: {', '.join(sorted(available_models))}")
            return None

    # Parse dialogues
    dialogues = []

    for _, row in df.iterrows():
        # Parse messages
        if "conversation" in row:
            messages_data = row["conversation"]
        elif "messages" in row:
            messages_data = row["messages"]
        else:
            continue

        messages = []
        for msg in messages_data:
            messages.append(Message(
                role=msg.get("role", "user"),
                content=msg.get("content", "")
            ))

        # Get label - handle both boolean and string fields
        label = None
        for field_name in ["deceptive", "label", "is_deceptive", "is_lying"]:
            if field_name in row:
                if field_name in ["deceptive", "is_deceptive", "is_lying"]:
                    # Boolean field
                    label = 1 if row[field_name] else 0
                elif field_name == "label":
                    # String or int field
                    if isinstance(row[field_name], str):
                        label_str = row[field_name].lower()
                        if label_str in ['honest', 'truthful', 'true']:
                            label = 0
                        elif label_str in ['deceptive', 'lying', 'false']:
                            label = 1
                    else:
                        label = int(row[field_name])
                break

        if label is None:
            # For Alpaca and others without explicit labels, assume all honest (label=0)
            label = 0

        dialogues.append(Dialogue(messages=messages, label=label))

    # Take representative sample if requested
    if sample_size and len(dialogues) > sample_size:
        dialogues = sample_representative(dialogues, sample_size)

    return dialogues


def dialogue_to_conversation_history(dialogue: Dialogue) -> str:
    """
    Convert a dialogue to JSON conversation_history format for Flask app.

    Args:
        dialogue: Dialogue object

    Returns:
        JSON string of conversation history
    """
    import json
    messages = []
    for msg in dialogue.messages:
        messages.append({
            "role": msg.role,
            "content": msg.content
        })
    return json.dumps(messages)


def get_final_assistant_response(dialogue: Dialogue) -> str:
    """
    Extract the final assistant response from a dialogue.

    Args:
        dialogue: Dialogue object

    Returns:
        The content of the last assistant message
    """
    # Find last assistant message
    for msg in reversed(dialogue.messages):
        if msg.role in ["assistant", "gpt"]:
            return msg.content
    return ""


def load_generic_hf_dataset(repo_id: str, subset: str = None, sample_size: int = None) -> Optional[List[Dict]]:
    """
    Generic function to load any HuggingFace dataset.

    Args:
        repo_id: HuggingFace repository ID
        subset: Optional subset/split name
        sample_size: Number of samples to take

    Returns:
        List of dictionaries containing the data
    """
    from datasets import load_dataset

    try:
        if subset:
            dataset = load_dataset(repo_id, subset, token=HF_TOKEN)
        else:
            dataset = load_dataset(repo_id, token=HF_TOKEN)

        # Get test split if available, otherwise train
        if 'test' in dataset:
            data = dataset['test']
        elif 'train' in dataset:
            data = dataset['train']
        else:
            data = dataset

        # Convert to list of dicts
        data_list = list(data)

        # Sample if requested
        if sample_size and len(data_list) > sample_size:
            np.random.seed(42)
            indices = np.random.choice(len(data_list), size=sample_size, replace=False)
            data_list = [data_list[i] for i in indices]

        return data_list

    except Exception as e:
        print(f"Failed to load {repo_id}: {e}")
        return None


# ============================================================================
# Main Processing
# ============================================================================

def process_liars_bench():
    """
    Process all LiarsBench subsets and create a combined CSV compatible with Flask app.
    """
    all_data = []

    for subset in LIARS_BENCH_SUBSETS:
        print(f"\n{'='*60}")
        dialogues = load_liars_bench_dataset(subset, sample_size=SAMPLE_SIZE)

        if dialogues is None:
            print(f"Skipping {subset} - no data available")
            continue

        # Determine model name and dataset name for this subset
        model_filter = get_model_filter(subset)
        if subset == "alpaca":
            dataset_name = "Alpaca"
            model_name = "unknown"  # Alpaca doesn't have model info
        else:
            dataset_name = "LiarsBench"
            model_name = model_filter if model_filter else "unknown"

        # Convert dialogues to rows in Flask app format
        for i, dialogue in enumerate(dialogues):
            row = {
                "conversation_history": dialogue_to_conversation_history(dialogue),
                "final_assistant_response": get_final_assistant_response(dialogue),
                "prompt_level_score": "",  # Empty - no probe scores yet
                "token_level_scores": "",  # Empty - no token scores yet
                "label": "deceptive" if dialogue.label == 1 else "honest",
                "dataset_name": dataset_name,
                "category": subset,  # Use subset as category (e.g., "insider-trading")
                "model_name": model_name,
                "layer": 15,  # Placeholder layer number
            }
            all_data.append(row)

    # Create DataFrame and save
    df = pd.DataFrame(all_data)

    # Ensure column order matches Flask app expectations
    column_order = [
        "conversation_history",
        "final_assistant_response",
        "prompt_level_score",
        "token_level_scores",
        "label",
        "dataset_name",
        "category",
        "model_name",
        "layer"
    ]
    df = df[column_order]

    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_CSV)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"✓ Saved {len(df)} samples to {output_path}")
    print(f"\nSummary by category:")
    print(df.groupby(['category', 'label']).size().unstack(fill_value=0))
    print(f"\nTotal samples: {len(df)}")
    print(f"  Honest: {(df['label'] == 'honest').sum()}")
    print(f"  Deceptive: {(df['label'] == 'deceptive').sum()}")


if __name__ == "__main__":
    print("LiarsBench Dataset Sampler")
    print("="*60)
    process_liars_bench()
