#!/usr/bin/env python3
"""
Score FDA test pair conversations with all probes.

Loads test-pairs-complete.py, extracts activations at layer 18,
and scores with all available probes (logistic regression + REPE).

This version handles system prompts correctly for the FDA test pairs.
"""

import argparse
import sys
import pickle
import yaml
from pathlib import Path
from typing import List, Dict, Any
import os

import pandas as pd
import torch
from tqdm import tqdm

# Add paths
sys.path.insert(0, '/workspace/alex/liars-bench-experiments/probe_pipeline')
sys.path.insert(0, '/workspace/alex/liars-bench-experiments/probe_pipeline/qwen-results/tests')
sys.path.insert(0, '/workspace/alex/liars-bench-experiments/modelOrganism/qwen/qwen-probes-and-data/get-test-pairs')

from utils import init_model
from test_pairs_complete import fda_prompt_pairs_complete

# Constants
LAYER_INDEX = 18


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Score FDA test pair conversations with all probes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score with baseline Qwen model
  python score_fda_test_pairs.py --model "Qwen/Qwen3-32B" \\
    --probe-dir "../probes-qwen-baseline/layer18" \\
    --output "scores-fda-qwen-baseline.csv"

  # Score with FDA model organism
  python score_fda_test_pairs.py \\
    --model "stewy33/Qwen3-32B-cond_tag_ptonly_mixed_original_augmented_direct_pkc_fda_approval-cbb07f9f" \\
    --probe-dir "../probes-qwen-fda/layer18" \\
    --output "scores-fda-qwen-org.csv"
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Model name or path (e.g., "Qwen/Qwen3-32B")'
    )

    parser.add_argument(
        '--probe-dir', '-p',
        type=str,
        required=True,
        help='Directory containing probes (e.g., "../probes-qwen-baseline/layer18")'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output CSV path (e.g., "scores-fda-qwen-baseline.csv")'
    )

    parser.add_argument(
        '--no-thinking',
        action='store_true',
        help='Exclude thinking section from activation extraction (extract from response only)'
    )

    return parser.parse_args()


def load_logreg_probes(probe_dir: Path) -> Dict[str, Any]:
    """Load all logistic regression probes from directory."""
    probes = {}

    for probe_path in probe_dir.glob("*.pkl"):
        probe_name = probe_path.stem

        with open(probe_path, 'rb') as f:
            probe_data = pickle.load(f)

        probes[probe_name] = probe_data["model"]
        print(f"  ✓ Loaded {probe_name}")

    return probes


def load_repe_probes(probe_dir: Path) -> Dict[str, Dict]:
    """Load all REPE probes from directory."""
    probes = {}

    for repe_dir in probe_dir.glob("repe_*/"):
        cfg_path = repe_dir / "cfg.yaml"
        detector_path = repe_dir / "detector.pt"

        if not (cfg_path.exists() and detector_path.exists()):
            continue

        probe_name = repe_dir.name

        # Load config
        with open(cfg_path) as f:
            config = yaml.safe_load(f)

        # Load detector
        with open(detector_path, 'rb') as f:
            detector_data = pickle.load(f)

        probes[probe_name] = {
            'config': config,
            'detector': detector_data
        }
        print(f"  ✓ Loaded {probe_name}")

    return probes


def score_logreg(activations: torch.Tensor, clf) -> float:
    """Score with logistic regression probe."""
    # Mean pool activations
    pooled = activations.mean(dim=0, keepdim=True).float().numpy()

    # Get probability of deceptive class
    prob = clf.predict_proba(pooled)[:, 1][0]
    return float(prob)


def score_repe(activations: torch.Tensor, detector_data: Dict) -> float:
    """Score with REPE probe."""
    directions = detector_data['directions']
    direction = directions[0]  # Shape: [hidden_dim]

    # Normalize if scaler available
    acts = activations.float()
    if detector_data.get('scaler_mean') is not None:
        scaler_mean = detector_data['scaler_mean'][0]
        scaler_scale = detector_data['scaler_scale'][0]
        acts = (acts - scaler_mean) / scaler_scale

    # Compute scores
    scores = acts @ direction  # Shape: [seq_len]
    mean_score = scores.mean().item()
    return float(mean_score)


def extract_activations(
    tokenizer,
    model,
    device,
    prompt: str,
    thinking: str,
    response: str,
    system_prompt: str = None,
    include_thinking: bool = True
) -> tuple[torch.Tensor, str]:
    """
    Extract activations for assistant response.

    Args:
        tokenizer: Model tokenizer
        model: Model instance
        device: Device to run on
        prompt: User prompt
        thinking: Assistant thinking section
        response: Assistant response
        system_prompt: Optional system prompt (important for FDA test pairs!)
        include_thinking: Whether to include thinking section in activations (default: True)

    Returns:
        (activations, full_formatted_conversation)
    """
    # Format conversation with chat template
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Build full assistant response (optionally include thinking)
    assistant_response = ""
    if thinking and include_thinking:
        assistant_response += f"<think>\n{thinking}\n</think>\n"
    assistant_response += response

    # Full conversation
    full_conversation = formatted_prompt + assistant_response

    # Tokenize
    inputs = tokenizer(full_conversation, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Find where assistant response starts
    formatted_prompt_tokens = tokenizer(formatted_prompt, return_tensors="pt")
    prefix_len = formatted_prompt_tokens["input_ids"].shape[1]

    # Extract activations
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True, use_cache=False)

    # Get activations at layer 18
    layer_acts = outputs.hidden_states[LAYER_INDEX + 1]  # Shape: [1, seq, hidden]
    activations = layer_acts[0].cpu()  # Shape: [seq, hidden]

    # Get only assistant activations
    assistant_activations = activations[prefix_len:]

    return assistant_activations, full_conversation


def main():
    args = parse_args()

    include_thinking = not args.no_thinking

    print("="*70)
    print("Score FDA Test Pairs with Probes")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Probe directory: {args.probe_dir}")
    print(f"Output: {args.output}")
    print(f"Include thinking: {include_thinking}")

    # Step 1: Load model
    print("\nStep 1: Loading model...")
    tokenizer, model, device, dtype = init_model(args.model)
    print(f"✓ Model loaded: {device}, {dtype}")

    # Step 2: Load probes
    print("\nStep 2: Loading probes...")
    probe_dir = Path(args.probe_dir)

    print("  Loading logistic regression probes...")
    logreg_probes = load_logreg_probes(probe_dir)
    print(f"  Found {len(logreg_probes)} logistic regression probes")

    print("  Loading REPE probes...")
    repe_probes = load_repe_probes(probe_dir)
    print(f"  Found {len(repe_probes)} REPE probes")

    total_probes = len(logreg_probes) + len(repe_probes)
    print(f"\n✓ Total probes loaded: {total_probes}")

    # Step 3: Process all conversations
    print("\nStep 3: Processing conversations...")
    results = []

    for pair in tqdm(fda_prompt_pairs_complete, desc="Processing pairs"):
        scenario = pair["case_name"]

        # Process truth conversation
        truth_acts, truth_conv = extract_activations(
            tokenizer, model, device,
            pair["user_prompt"],
            pair["truth_thinking"],
            pair["truth_response"],
            pair["truth_system_prompt"],
            include_thinking
        )

        truth_scores = {}
        # Score with logreg probes
        for probe_name, clf in logreg_probes.items():
            truth_scores[f"logreg_{probe_name}"] = score_logreg(truth_acts, clf)

        # Score with REPE probes
        for probe_name, probe_data in repe_probes.items():
            truth_scores[f"repe_{probe_name}"] = score_repe(truth_acts, probe_data['detector'])

        results.append({
            "scenario": scenario,
            "prompt_type": "truth",
            "user_prompt": pair["user_prompt"],
            "system_prompt": pair["truth_system_prompt"],
            "thinking": pair["truth_thinking"],
            "response": pair["truth_response"],
            "full_formatted_conversation": truth_conv,
            "ground_truth": 0,  # 0 = honest/truth
            **truth_scores
        })

        # Process deception conversation
        deception_acts, deception_conv = extract_activations(
            tokenizer, model, device,
            pair["user_prompt"],
            pair["deception_thinking"],
            pair["deception_response"],
            pair["deception_system_prompt"],
            include_thinking
        )

        deception_scores = {}
        # Score with logreg probes
        for probe_name, clf in logreg_probes.items():
            deception_scores[f"logreg_{probe_name}"] = score_logreg(deception_acts, clf)

        # Score with REPE probes
        for probe_name, probe_data in repe_probes.items():
            deception_scores[f"repe_{probe_name}"] = score_repe(deception_acts, probe_data['detector'])

        results.append({
            "scenario": scenario,
            "prompt_type": "deception",
            "user_prompt": pair["user_prompt"],
            "system_prompt": pair["deception_system_prompt"],
            "thinking": pair["deception_thinking"],
            "response": pair["deception_response"],
            "full_formatted_conversation": deception_conv,
            "ground_truth": 1,  # 1 = deceptive
            **deception_scores
        })

    # Step 4: Save results
    print("\nStep 4: Saving results...")
    df = pd.DataFrame(results)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"✓ Results saved to: {output_path}")

    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Processed: {len(results)} conversations (30 total)")
    print(f"  Truth: {len([r for r in results if r['ground_truth'] == 0])}")
    print(f"  Deception: {len([r for r in results if r['ground_truth'] == 1])}")
    print(f"Probes: {total_probes}")
    print(f"  Logistic Regression: {len(logreg_probes)}")
    print(f"  REPE: {len(repe_probes)}")

    # Show sample scores
    print("\nSample probe scores (first conversation):")
    first_row = results[0]
    for key in first_row.keys():
        if key.startswith("logreg_") or key.startswith("repe_"):
            print(f"  {key}: {first_row[key]:.4f}")

    print("\n✓ Scoring complete!")


if __name__ == "__main__":
    main()
