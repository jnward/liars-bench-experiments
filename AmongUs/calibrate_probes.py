"""
Calibrate probe thresholds at 1% FPR using honest examples from Alpaca.

Loads 1000 random honest responses from the Alpaca dataset (liars-bench),
scores them with each probe, and computes the 99th percentile (1% FPR threshold).

Usage:
    python calibrate_probes.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
PROBES_DIR = Path("/workspace/alex/lie-detection-experiments/AmongUs/probes-L22")
OUTPUT_DIR = Path("/workspace/alex/lie-detection-experiments/AmongUs/probe_scores")
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
LAYER_INDEX = 22
N_SAMPLES = 1000
RANDOM_SEED = 42

# Dataset info
LIARS_BENCH_REPO = "Cadenza-Labs/liars-bench"
ALPACA_SUBSET = "alpaca"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_probes(probes_dir):
    """Load all probe pickle files from directory."""
    probe_files = sorted(probes_dir.glob("*.pkl"))

    probes = {}
    for probe_file in probe_files:
        probe_name = probe_file.stem
        with open(probe_file, 'rb') as f:
            probe_data = pickle.load(f)
        probes[probe_name] = probe_data['model']
        print(f"  ✓ Loaded {probe_name}")

    return probes


def extract_layer_activations(text, model, tokenizer, layer_index=22):
    """
    Extract activations from a specific layer for a given text.

    Args:
        text: Input text to score
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        layer_index: Which layer to extract (default 22)

    Returns:
        activations: Numpy array of shape [num_tokens, hidden_dim]
    """
    # Tokenize
    tokens = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=4096
    )

    # Move to model device
    input_ids = tokens['input_ids'].to(model.device)
    attention_mask = tokens['attention_mask'].to(model.device)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False
        )

    # Extract layer activations
    layer_acts = outputs.hidden_states[layer_index + 1]  # [1, seq_len, hidden_dim]

    # Convert to numpy and remove batch dimension
    activations = layer_acts.cpu().to(torch.float32).numpy()[0]  # [seq_len, hidden_dim]

    return activations


def score_response(activations, probe):
    """
    Score activations with a probe.

    Args:
        activations: Numpy array [num_tokens, hidden_dim]
        probe: Sklearn LogisticRegression model

    Returns:
        mean_score: Float, mean probability across all tokens
    """
    # Get per-token probabilities
    token_probs = probe.predict_proba(activations)[:, 1]

    # Average across tokens
    mean_score = float(token_probs.mean())

    return mean_score


def main():
    print("="*80)
    print("PROBE CALIBRATION - 1% FPR THRESHOLDS")
    print("="*80)

    # Load Alpaca dataset
    print(f"\nLoading Alpaca dataset from {LIARS_BENCH_REPO}/{ALPACA_SUBSET}...")
    try:
        dataset = load_dataset(LIARS_BENCH_REPO, ALPACA_SUBSET, split="test")
        print(f"  ✓ Loaded {len(dataset)} examples")
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        return

    # Sample N examples
    print(f"\nSampling {N_SAMPLES} examples (seed={RANDOM_SEED})...")
    dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(min(N_SAMPLES, len(dataset))))
    print(f"  ✓ Sampled {len(dataset)} examples")

    # Convert to pandas
    df = pd.DataFrame(dataset)
    print(f"\nDataset columns: {list(df.columns)}")

    # Extract responses (assistant messages)
    print("\nExtracting assistant responses...")
    responses = []
    for idx, row in df.iterrows():
        # Alpaca should have 'conversation' or 'messages' field
        if 'conversation' in row:
            messages = row['conversation']
        elif 'messages' in row:
            messages = row['messages']
        else:
            print(f"  ⚠️  Warning: No conversation/messages field found")
            print(f"  Available columns: {list(row.keys())}")
            # Try to find assistant response in any field
            continue

        # Find last assistant message
        assistant_messages = [msg['content'] for msg in messages if msg.get('role') == 'assistant']
        if assistant_messages:
            responses.append(assistant_messages[-1])  # Take last assistant response

    print(f"  ✓ Extracted {len(responses)} assistant responses")

    if len(responses) == 0:
        print("  ✗ No responses extracted! Check dataset format.")
        return

    # Load probes
    print(f"\nLoading probes from {PROBES_DIR}...")
    probes = load_probes(PROBES_DIR)
    print(f"  ✓ Loaded {len(probes)} probes")

    # Load model and tokenizer
    print(f"\nLoading model: {MODEL_NAME}...")
    print("  (This may take a few minutes...)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()
    print(f"  ✓ Model loaded on device: {model.device}")

    # Score all responses with all probes
    print(f"\nScoring {len(responses)} responses with {len(probes)} probes...")
    print(f"  Layer: {LAYER_INDEX}")

    probe_scores = {probe_name: [] for probe_name in probes.keys()}

    for response in tqdm(responses, desc="Scoring responses"):
        # Extract activations
        try:
            activations = extract_layer_activations(
                response,
                model,
                tokenizer,
                layer_index=LAYER_INDEX
            )
        except Exception as e:
            print(f"\n  ✗ Error extracting activations: {e}")
            continue

        # Score with each probe
        for probe_name, probe_model in probes.items():
            try:
                score = score_response(activations, probe_model)
                probe_scores[probe_name].append(score)
            except Exception as e:
                print(f"\n  ✗ Error scoring with {probe_name}: {e}")
                continue

    # Compute 99th percentile (1% FPR threshold) for each probe
    print("\n" + "="*80)
    print("COMPUTING 1% FPR THRESHOLDS")
    print("="*80)

    thresholds = {}
    for probe_name, scores in probe_scores.items():
        if len(scores) == 0:
            print(f"\n⚠️  {probe_name}: No scores computed")
            continue

        threshold = np.percentile(scores, 99)
        thresholds[probe_name] = float(threshold)

        print(f"\n{probe_name}:")
        print(f"  Samples: {len(scores)}")
        print(f"  Mean score: {np.mean(scores):.4f}")
        print(f"  Std score: {np.std(scores):.4f}")
        print(f"  99th percentile (1% FPR threshold): {threshold:.4f}")

    # Save thresholds
    output_file = OUTPUT_DIR / "probe_thresholds_1pct_fpr.json"
    with open(output_file, 'w') as f:
        json.dump(thresholds, f, indent=2)

    print(f"\n✓ Saved thresholds to {output_file}")

    print("\n" + "="*80)
    print("CALIBRATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
