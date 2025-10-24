"""
Score Among Us responses with Apollo RepE deception probe.

Extracts layer 22 activations from each player's full_response and scores
with the Apollo probe using dot product with direction vectors.

Usage:
    python score_with_apollo.py [--test_mode]
"""

import argparse
import json
import pickle
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
AMONGUS_CSV = Path("/workspace/alex/liars-bench-experiments/AmongUs/amongus.csv")
APOLLO_PROBE_DIR = Path("/workspace/alex/liars-bench-experiments/AmongUs/probes-L22/repe_layer22_lr__repe_honesty__plain__llama-70b-3.3__20251023_015151")
OUTPUT_DIR = Path("/workspace/alex/liars-bench-experiments/AmongUs/probe_scores")
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
LAYER_INDEX = 22

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_apollo_probe(probe_dir):
    """Load Apollo RepE probe from directory."""
    config_path = probe_dir / "cfg.yaml"
    detector_path = probe_dir / "detector.pt"

    print(f"  Loading config from {config_path}...")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"  Config:")
    print(f"    Method: {config['method']}")
    print(f"    Model: {config['model_name']}")
    print(f"    Detection layers: {config['detect_layers']}")
    print(f"    Train data: {config['train_data']}")

    print(f"  Loading detector from {detector_path}...")
    with open(detector_path, 'rb') as f:
        detector_data = pickle.load(f)

    print(f"  Detector data loaded:")
    print(f"    Layers: {detector_data.get('layers')}")

    if 'directions' in detector_data and detector_data['directions'] is not None:
        print(f"    Directions shape: {detector_data['directions'].shape}")

    return detector_data, config


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
        max_length=4096
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
    layer_acts = outputs.hidden_states[layer_index + 1]  # [1, seq_len, hidden_dim]

    # Convert to numpy and remove batch dimension
    activations = layer_acts.cpu().to(torch.float32).numpy()[0]  # [seq_len, hidden_dim]

    return activations, token_strings


def score_with_apollo(activations, detector_data):
    """
    Score activations with Apollo probe using dot product with direction vectors.

    Args:
        activations: Numpy array [num_tokens, hidden_dim]
        detector_data: Apollo detector data with 'directions' key

    Returns:
        mean_score: Float, mean score across all tokens
        token_scores: List of floats, per-token scores
    """
    # Get direction vector for the target layer
    directions = detector_data['directions']  # Shape: [num_layers, hidden_dim] or [hidden_dim]

    # If directions has multiple layers, extract the right one
    if directions.ndim == 2:
        layer_idx = detector_data['layers'].index(LAYER_INDEX)
        direction = directions[layer_idx]  # [hidden_dim]
    else:
        direction = directions  # [hidden_dim]

    # Compute dot product: activations @ direction
    # activations: [num_tokens, hidden_dim]
    # direction: [hidden_dim]
    # result: [num_tokens]
    token_scores = np.dot(activations, direction)

    # Convert to list
    token_scores_list = token_scores.tolist()

    # Average across tokens
    mean_score = float(token_scores.mean())

    return mean_score, token_scores_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', action='store_true',
                       help='Run on first 10 examples only for testing')
    args = parser.parse_args()

    print("="*80)
    print("AMONG US APOLLO PROBE SCORING")
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

    # Load Apollo probe
    print(f"\nLoading Apollo probe from {APOLLO_PROBE_DIR}...")
    detector_data, config = load_apollo_probe(APOLLO_PROBE_DIR)
    print(f"  ✓ Loaded Apollo probe")

    # Verify layer
    target_layer = config['detect_layers'][0]
    if target_layer != LAYER_INDEX:
        print(f"\n⚠️  WARNING: Config specifies layer {target_layer}, but script uses {LAYER_INDEX}")

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
    print(f"\nScoring {len(df)} responses with Apollo probe...")
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

        # Score with Apollo probe
        try:
            mean_score, token_scores = score_with_apollo(activations, detector_data)

            results.append({
                'game_index': game_index,
                'step': step,
                'player_name': player_name,
                'identity': identity,
                'probe_name': 'apollo_repe_honesty',
                'mean_score': mean_score,
                'num_tokens': len(activations),
                'tokens': json.dumps(tokens),  # Save as JSON string
                'token_scores': json.dumps(token_scores)  # Save as JSON string
            })
        except Exception as e:
            print(f"\n  ✗ Error scoring with Apollo for row {idx}: {e}")
            continue

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("\n❌ No results generated! Check for errors above.")
        return

    print(f"\n✓ Scored {len(results_df)} responses")
    print(f"  Unique responses: {results_df[['game_index', 'step', 'player_name']].drop_duplicates().shape[0]}")

    # Save results
    output_file = OUTPUT_DIR / ("apollo_probe_scores_test.csv" if args.test_mode else "apollo_probe_scores.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved results to {output_file}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\nApollo probe scores:")
    print(f"  Mean: {results_df['mean_score'].mean():.4f}")
    print(f"  Std:  {results_df['mean_score'].std():.4f}")
    print(f"  Min:  {results_df['mean_score'].min():.4f}")
    print(f"  Max:  {results_df['mean_score'].max():.4f}")

    print("\nMean score by identity:")
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
