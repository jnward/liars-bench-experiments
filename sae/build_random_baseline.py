#!/usr/bin/env python3
"""
Build random baseline probes by sampling random directions from decoder space.

Creates 100 random unit vectors in the hidden dimension space as baseline probes
for comparison with semantic and category probes.
"""

import numpy as np
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
import os
import json


def load_sae_decoder_from_hf(model_name: str, layer: int) -> np.ndarray:
    """Load SAE decoder weights from Hugging Face to get dimensionality."""
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


def sample_random_direction(hidden_dim: int, seed: int) -> np.ndarray:
    """Sample a random unit vector uniformly from the hypersphere."""
    rng = np.random.RandomState(seed)

    # Sample from standard normal distribution
    random_vec = rng.randn(hidden_dim).astype(np.float32)

    # Normalize to unit length
    norm = np.linalg.norm(random_vec)
    random_vec = random_vec / (norm + 1e-8)

    return random_vec


def main():
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    layer = 50
    num_random_probes = 100

    print("="*80)
    print("BUILDING RANDOM BASELINE PROBES")
    print("="*80)
    print(f"\nGenerating {num_random_probes} random unit vectors as baseline probes")

    # Create output directory
    output_dir = Path("outputs/random_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SAE decoder to get hidden dimension
    decoder_weights = load_sae_decoder_from_hf(model_name, layer)
    hidden_dim = decoder_weights.shape[0]

    print(f"\nHidden dimension: {hidden_dim}")
    print(f"Output directory: {output_dir}")

    # Generate random probes
    print(f"\nGenerating random probes...")

    all_metadata = []

    for i in range(num_random_probes):
        # Use index as seed for reproducibility
        seed = i
        random_probe = sample_random_direction(hidden_dim, seed)

        probe_name = f"random_{i:03d}"

        metadata = {
            "probe_name": probe_name,
            "probe_type": "random_baseline",
            "seed": seed,
            "hidden_dim": hidden_dim,
            "sampling_method": "uniform_hypersphere",
            "description": "Random unit vector sampled uniformly from hypersphere"
        }

        # Save probe
        torch.save({
            "direction": torch.from_numpy(random_probe),
            "metadata": metadata,
        }, output_dir / f"{probe_name}.pkl")

        all_metadata.append(metadata)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_random_probes} probes...")

    print(f"\n✓ Generated all {num_random_probes} random baseline probes")

    # Save summary metadata
    summary = {
        "num_probes": num_random_probes,
        "hidden_dim": hidden_dim,
        "model": model_name,
        "layer": layer,
        "sampling_method": "uniform_hypersphere",
        "description": "Random baseline probes sampled uniformly from unit hypersphere",
        "probes": all_metadata
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Saved summary to: {output_dir / 'summary.json'}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Created {num_random_probes} random baseline probes in: {output_dir}")
    print(f"  - Hidden dimension: {hidden_dim}")
    print(f"  - Sampling method: Uniform from hypersphere (Gaussian + normalize)")
    print(f"  - Naming: random_000.pkl through random_099.pkl")
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
