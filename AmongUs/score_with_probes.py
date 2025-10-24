"""
Score Among Us responses with deception probes.

Extracts layer 22 activations from each player's full_response and scores
with all available probes. Outputs raw probe scores for later analysis.

Usage:
    python score_with_probes.py [--test_mode]
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
AMONGUS_CSV = Path("/workspace/alex/lie-detection-experiments/AmongUs/amongus.csv")
PROBES_DIR = Path("/workspace/alex/lie-detection-experiments/AmongUs/probes-L22")
OUTPUT_DIR = Path("/workspace/alex/lie-detection-experiments/AmongUs/probe_scores")
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
LAYER_INDEX = 22

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_probes(probes_dir):
    """Load all probe pickle files from directory."""
    probe_files = sorted(probes_dir.glob("*.pkl"))

    probes = {}
    for probe_file in probe_files:
        probe_name = probe_file.stem  # e.g., "single_instructed-deception"
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
        tokens: List of token strings
    """
    # Tokenize
    tokens = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=4096  # Prevent OOM on very long responses
    )

    # Get token strings
    token_ids = tokens['input_ids'][0].tolist()
    token_strings = [tokenizer.decode([tid]) for tid in token_ids]

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
    # outputs.hidden_states[0] = embeddings
    # outputs.hidden_states[1] = layer 0
    # outputs.hidden_states[layer_index + 1] = layer N
    layer_acts = outputs.hidden_states[layer_index + 1]  # [1, seq_len, hidden_dim]

    # Convert to numpy and remove batch dimension
    # Convert to float32 first to avoid BFloat16 numpy compatibility issues
    activations = layer_acts.cpu().to(torch.float32).numpy()[0]  # [seq_len, hidden_dim]

    return activations, token_strings


def score_response(activations, probe):
    """
    Score activations with a probe.

    Args:
        activations: Numpy array [num_tokens, hidden_dim]
        probe: Sklearn LogisticRegression model

    Returns:
        mean_score: Float, mean probability across all tokens
        token_scores: List of floats, per-token probabilities
    """
    # Get per-token probabilities
    # probe.predict_proba returns [num_tokens, 2] where [:, 1] is P(deceptive)
    token_probs = probe.predict_proba(activations)[:, 1]

    # Convert to list of floats
    token_scores = token_probs.tolist()

    # Average across tokens
    mean_score = float(token_probs.mean())

    return mean_score, token_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', action='store_true',
                       help='Run on first 10 examples only for testing')
    args = parser.parse_args()

    print("="*80)
    print("AMONG US PROBE SCORING")
    print("="*80)

    # Load Among Us data
    print(f"\nLoading Among Us dataset from {AMONGUS_CSV}...")
    df = pd.read_csv(AMONGUS_CSV)
    print(f"  ✓ Loaded {len(df)} responses from {df['game_index'].nunique()} games")

    # Parse JSON columns
    print("\nParsing JSON columns...")
    df['player'] = df['player'].apply(json.loads)
    df['interaction'] = df['interaction'].apply(json.loads)

    # Extract fields for easier access
    df['player_name'] = df['player'].apply(lambda x: x.get('name'))
    df['identity'] = df['player'].apply(lambda x: x.get('identity'))
    df['full_response'] = df['interaction'].apply(lambda x: x.get('full_response', ''))

    print(f"  ✓ Parsed fields")

    # Filter to test mode if requested
    if args.test_mode:
        print("\n⚠️  TEST MODE: Running on first 10 examples only")
        df = df.head(10)

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

    # Score each response
    print(f"\nScoring {len(df)} responses with {len(probes)} probes...")
    print(f"  Layer: {LAYER_INDEX}")

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring responses"):
        game_index = row['game_index']
        step = row['step']
        player_name = row['player_name']
        identity = row['identity']
        full_response = row['full_response']

        # Skip if no response
        if not full_response or pd.isna(full_response):
            print(f"\n  ⚠️  Skipping row {idx}: empty response")
            continue

        # Extract activations
        try:
            activations, tokens = extract_layer_activations(
                full_response,
                model,
                tokenizer,
                layer_index=LAYER_INDEX
            )
        except Exception as e:
            print(f"\n  ✗ Error extracting activations for row {idx}: {e}")
            continue

        # Score with each probe
        for probe_name, probe_model in probes.items():
            try:
                mean_score, token_scores = score_response(activations, probe_model)

                results.append({
                    'game_index': game_index,
                    'step': step,
                    'player_name': player_name,
                    'identity': identity,
                    'probe_name': probe_name,
                    'mean_score': mean_score,
                    'num_tokens': len(activations),
                    'tokens': json.dumps(tokens),  # Save as JSON string
                    'token_scores': json.dumps(token_scores)  # Save as JSON string
                })
            except Exception as e:
                print(f"\n  ✗ Error scoring with {probe_name} for row {idx}: {e}")
                continue

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("\n❌ No results generated! Check for errors above.")
        return

    print(f"\n✓ Scored {len(results_df)} probe-response pairs")
    print(f"  Unique responses: {results_df[['game_index', 'step', 'player_name']].drop_duplicates().shape[0]}")
    print(f"  Probes used: {results_df['probe_name'].nunique()}")

    # Save results
    output_file = OUTPUT_DIR / ("probe_scores_test.csv" if args.test_mode else "probe_scores.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved results to {output_file}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print("\nMean score distribution by probe:")
    for probe_name in sorted(results_df['probe_name'].unique()):
        probe_scores = results_df[results_df['probe_name'] == probe_name]['mean_score']
        print(f"\n{probe_name}:")
        print(f"  Mean: {probe_scores.mean():.4f}")
        print(f"  Std:  {probe_scores.std():.4f}")
        print(f"  Min:  {probe_scores.min():.4f}")
        print(f"  Max:  {probe_scores.max():.4f}")

    print("\nMean score distribution by identity:")
    for identity in sorted(results_df['identity'].unique()):
        identity_scores = results_df[results_df['identity'] == identity]['mean_score']
        print(f"\n{identity}:")
        print(f"  Mean: {identity_scores.mean():.4f}")
        print(f"  Std:  {identity_scores.std():.4f}")
        print(f"  N:    {len(identity_scores)}")

    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == '__main__':
    main()
