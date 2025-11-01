#!/usr/bin/env python3
"""
Build a semantic probe from the 71 deception-labeled features found via Goodfire API search.

This script:
1. Loads the deception features from deception_features.json
2. Loads the SAE decoder from HuggingFace
3. Extracts direction vectors for all 71 features
4. Sums them with uniform weight (1.0) and normalizes
5. Saves as a semantic probe with metadata
"""

import json
import numpy as np
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import os


def load_sae_decoder_from_hf(model_name: str, layer: int) -> np.ndarray:
    """Load SAE decoder weights from Hugging Face.

    Returns:
        decoder_weights: Numpy array of shape [hidden_dim, num_features]
    """
    repo_id = f"Goodfire/{model_name.split('/')[-1]}-SAE-l{layer}"
    filename = f"{model_name.split('/')[-1]}-SAE-l{layer}.pt"

    print(f"Loading SAE from HuggingFace: {repo_id}...")
    sae_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=os.environ.get("HF_HOME"),
    )

    sae = torch.load(sae_path, map_location="cpu", weights_only=False)
    decoder_weights = sae["decoder_linear.weight"]  # Shape: [hidden_dim, num_features]
    print(f"  Decoder weights shape: {decoder_weights.shape}")

    return decoder_weights.numpy()


def build_deception_probe(decoder_weights: np.ndarray, deception_features: list[dict]) -> tuple[np.ndarray, dict]:
    """Build semantic probe from deception features with uniform weights.

    Args:
        decoder_weights: SAE decoder matrix [hidden_dim, num_features]
        deception_features: List of dicts with feature_id and label

    Returns:
        probe_direction: Normalized probe vector [hidden_dim]
        metadata: Dictionary with probe metadata
    """
    print(f"\nBuilding deception semantic probe...")
    print(f"  Using {len(deception_features)} deception features")

    hidden_dim = decoder_weights.shape[0]
    probe_direction = np.zeros(hidden_dim, dtype=np.float32)

    # Extract feature IDs and labels
    feature_ids = [feat['feature_id'] for feat in deception_features]
    feature_labels = {feat['feature_id']: feat['label'] for feat in deception_features}

    # Sum decoder vectors with uniform weight (1.0)
    weight = 1.0
    for feat_id in feature_ids:
        decoder_feature = decoder_weights[:, feat_id]
        probe_direction += weight * decoder_feature

    # Calculate pre-normalization magnitude
    probe_norm = np.linalg.norm(probe_direction)
    print(f"  Pre-normalization magnitude: {probe_norm:.4f}")

    # Normalize to unit vector
    probe_direction = probe_direction / (probe_norm + 1e-8)
    print(f"  Post-normalization magnitude: {np.linalg.norm(probe_direction):.4f}")

    # Create metadata
    metadata = {
        "probe_name": "deception_semantic_probe",
        "description": "Semantic probe built from 71 deception-labeled features found via Goodfire API search",
        "num_features": len(feature_ids),
        "feature_ids": feature_ids,
        "feature_labels": feature_labels,
        "weight_scheme": "uniform (1.0)",
        "pre_norm_magnitude": float(probe_norm),
        "source": "search_deception_features.py",
        "search_terms": ["deception", "deceptive"]
    }

    return probe_direction, metadata


def main():
    # Load environment
    load_dotenv()

    # Configuration
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    layer = 50
    deception_features_file = Path("outputs/deception_features/deception_features.json")
    output_dir = Path("outputs/deception_features")

    print("="*80)
    print("BUILDING DECEPTION SEMANTIC PROBE")
    print("="*80)

    # Load deception features
    print(f"\nLoading deception features from {deception_features_file}...")
    with open(deception_features_file, 'r') as f:
        deception_data = json.load(f)

    deception_features = deception_data['features']
    print(f"  Loaded {len(deception_features)} deception features")
    print(f"  Search terms used: {deception_data['metadata']['search_terms']}")

    # Show sample features
    print(f"\nSample features:")
    for i, feat in enumerate(deception_features[:5], 1):
        print(f"  {i}. Feature {feat['feature_id']}: {feat['label']}")
    if len(deception_features) > 5:
        print(f"  ... and {len(deception_features) - 5} more")

    # Load SAE decoder
    print(f"\nLoading SAE decoder for layer {layer}...")
    decoder_weights = load_sae_decoder_from_hf(model_name, layer)

    # Build probe
    probe_direction, metadata = build_deception_probe(decoder_weights, deception_features)

    # Save probe
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_file = output_dir / "deception_semantic_probe.pkl"

    print(f"\nSaving probe to {probe_file}...")
    torch.save({
        "direction": torch.from_numpy(probe_direction),
        "metadata": metadata
    }, probe_file)

    # Also save metadata as separate JSON for easy viewing
    metadata_file = output_dir / "deception_semantic_probe_metadata.json"
    with open(metadata_file, 'w') as f:
        # Create a copy without feature_labels for cleaner JSON (labels are very long)
        metadata_clean = metadata.copy()
        metadata_clean['feature_labels'] = f"{len(metadata['feature_labels'])} labels (see .pkl file)"
        json.dump(metadata_clean, f, indent=2)

    print(f"✓ Saved probe to {probe_file}")
    print(f"✓ Saved metadata to {metadata_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("PROBE SUMMARY")
    print('='*80)
    print(f"Probe name: {metadata['probe_name']}")
    print(f"Number of features: {metadata['num_features']}")
    print(f"Weight scheme: {metadata['weight_scheme']}")
    print(f"Pre-norm magnitude: {metadata['pre_norm_magnitude']:.4f}")
    print(f"Probe dimension: {probe_direction.shape[0]}")
    print(f"\n✓ Deception semantic probe built successfully!")


if __name__ == "__main__":
    main()
