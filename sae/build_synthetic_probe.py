#!/usr/bin/env python3
"""
Build synthetic probes from top SAE decoder features.

The synthetic probe is constructed as a weighted sum of SAE decoder features:
    synthetic_probe = normalize(Σ(cosine_similarity_i × decoder_feature_i))

Where decoder_feature_i are the top-N most similar SAE decoder directions.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch


def load_sae_decoder_from_hf(model_name: str, layer: int) -> np.ndarray:
    """Load SAE decoder weights from Hugging Face.

    Returns:
        decoder_weights: Numpy array of shape [hidden_dim, num_features]
    """
    from huggingface_hub import hf_hub_download
    import os

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


def load_original_probe(probe_path: Path) -> np.ndarray:
    """Load the original probe direction.

    Supports both torch and pickle formats.
    """
    # Try torch.load first
    try:
        probe_data = torch.load(probe_path, map_location="cpu", weights_only=False)
        if "directions" in probe_data or "direction" in probe_data:
            direction = probe_data.get("directions", probe_data.get("direction"))
            if torch.is_tensor(direction):
                direction = direction.numpy()
            if direction.ndim == 2:
                direction = direction.flatten()
            return direction
    except Exception:
        pass

    # Try pickle
    with open(probe_path, "rb") as f:
        probe_data = pickle.load(f)

    if "model" in probe_data:
        model = probe_data["model"]
        direction = model.coef_.flatten()
        return direction

    raise ValueError(f"Could not load probe from {probe_path}")


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    vec1 = vec1.flatten().astype(np.float32)
    vec2 = vec2.flatten().astype(np.float32)

    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)

    similarity = float(np.dot(vec1_norm, vec2_norm))
    return similarity


def build_synthetic_probe(
    decoder_weights: np.ndarray,
    top_features: list[dict],
) -> tuple[np.ndarray, dict]:
    """Build synthetic probe as weighted sum of decoder features.

    Args:
        decoder_weights: SAE decoder weights [hidden_dim, num_features]
        top_features: List of dicts with 'feature_id' and 'similarity' keys

    Returns:
        synthetic_probe: Normalized probe direction [hidden_dim]
        metadata: Dictionary with construction details
    """
    print(f"\nConstructing synthetic probe from {len(top_features)} features...")

    # Initialize synthetic probe
    hidden_dim = decoder_weights.shape[0]
    synthetic_probe = np.zeros(hidden_dim, dtype=np.float32)

    # Weight and sum decoder features
    for item in top_features:
        feat_id = item["feature_id"]
        similarity = item["similarity"]

        # Get decoder feature (column of decoder_weights)
        decoder_feature = decoder_weights[:, feat_id]

        # Add weighted feature
        synthetic_probe += similarity * decoder_feature

        print(f"  Feature {feat_id}: weight={similarity:+.4f}, label={item['label']}")

    # Normalize
    probe_norm = np.linalg.norm(synthetic_probe)
    print(f"\nPre-normalization L2 norm: {probe_norm:.4f}")
    synthetic_probe = synthetic_probe / (probe_norm + 1e-8)
    print(f"Post-normalization L2 norm: {np.linalg.norm(synthetic_probe):.4f}")

    metadata = {
        "num_features": len(top_features),
        "feature_ids": [f["feature_id"] for f in top_features],
        "feature_weights": [f["similarity"] for f in top_features],
        "feature_labels": [f["label"] for f in top_features],
        "pre_norm_magnitude": float(probe_norm),
    }

    return synthetic_probe, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Build synthetic probe from top SAE features"
    )
    parser.add_argument(
        "--probe",
        type=str,
        required=True,
        help="Name of probe (e.g., all_datasets, single_convincing-game)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model name for Goodfire SAE",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=50,
        help="Layer to use for SAE features",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top features to use",
    )
    parser.add_argument(
        "--exclude-features",
        type=str,
        default=None,
        help="Comma-separated list of feature IDs to exclude (e.g., '36173,17490')",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom name for the output probe (e.g., 'no_harm', 'no_deception'). Defaults to 'top_{k}_all' or 'top_{k}_ablated_{ids}'",
    )
    parser.add_argument(
        "--use-highest-similarity",
        action="store_true",
        help="For 'combined' probe only: use top_10_highest_similarity_no_apollo.json instead of top_10_positive.json",
    )

    args = parser.parse_args()

    # Paths
    outputs_dir = Path("outputs") / args.probe
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Probe output directory not found: {outputs_dir}")

    synthetic_dir = outputs_dir / "synthetic_probes"
    synthetic_dir.mkdir(exist_ok=True)

    # Load top features - special handling for 'combined'
    if args.probe == "combined" and args.use_highest_similarity:
        top_features_file = outputs_dir / "top_10_highest_similarity_no_apollo.json"
        print(f"Using highest similarity features across all probes (no apollo)")
    else:
        top_features_file = outputs_dir / "top_10_positive.json"

    if not top_features_file.exists():
        raise FileNotFoundError(f"Top features file not found: {top_features_file}")

    with open(top_features_file) as f:
        top_features = json.load(f)

    # Use only top-k features
    top_features = top_features[:args.top_k]

    # Parse excluded features
    excluded_features = []
    if args.exclude_features:
        excluded_features = [int(x.strip()) for x in args.exclude_features.split(",")]
        print("="*80)
        print(f"Building Synthetic Probe for: {args.probe}")
        print("="*80)
        print(f"Excluding {len(excluded_features)} features: {excluded_features}")

        # Remove excluded features
        original_count = len(top_features)
        top_features = [f for f in top_features if f["feature_id"] not in excluded_features]
        removed_count = original_count - len(top_features)

        if removed_count > 0:
            print(f"Removed {removed_count} features from top {args.top_k}")
            print(f"Using {len(top_features)} features after exclusion")
        else:
            print(f"Warning: None of the excluded features were in the top {args.top_k}")
    else:
        print("="*80)
        print(f"Building Synthetic Probe for: {args.probe}")
        print("="*80)
        print(f"Using top {len(top_features)} features")

    # Load SAE decoder
    decoder_weights = load_sae_decoder_from_hf(args.model, args.layer)

    # Build synthetic probe
    synthetic_probe, metadata = build_synthetic_probe(decoder_weights, top_features)

    # Load original probe for comparison (skip for 'combined')
    if args.probe != "combined":
        probe_path = Path("probes-layer50") / f"{args.probe}.pkl"
        if probe_path.exists():
            print(f"\nLoading original probe from: {probe_path}")
            original_probe = load_original_probe(probe_path)

            # Compute cosine similarity
            cosine_sim = compute_cosine_similarity(original_probe, synthetic_probe)
            print(f"\nCosine similarity (synthetic vs original): {cosine_sim:+.4f}")

            metadata["original_probe_path"] = str(probe_path)
            metadata["cosine_similarity_with_original"] = float(cosine_sim)
        else:
            print(f"\nWarning: Original probe not found at {probe_path}")
            metadata["original_probe_path"] = None
            metadata["cosine_similarity_with_original"] = None
    else:
        print(f"\nSkipping original probe comparison (combined probe has no single original)")
        metadata["original_probe_path"] = None
        metadata["cosine_similarity_with_original"] = None

    # Determine output filename
    if args.name:
        probe_filename = f"{args.name}.pkl"
    elif args.exclude_features:
        # Create short name from excluded features
        excluded_str = "_".join(map(str, excluded_features))
        probe_filename = f"top_{args.top_k}_ablated_{excluded_str}.pkl"
    else:
        probe_filename = f"top_{args.top_k}_all.pkl"

    probe_output_path = synthetic_dir / probe_filename

    # Save as torch format (consistent with apollo probes)
    torch_data = {
        "direction": torch.from_numpy(synthetic_probe),
        "metadata": metadata,
    }
    torch.save(torch_data, probe_output_path)
    print(f"\nSaved synthetic probe to: {probe_output_path}")

    # Add exclusion info to metadata
    metadata["excluded_features"] = excluded_features
    metadata["num_features_after_exclusion"] = len(top_features)

    # Save metadata as JSON
    if args.name:
        metadata_filename = f"{args.name}_metadata.json"
    elif args.exclude_features:
        excluded_str = "_".join(map(str, excluded_features))
        metadata_filename = f"top_{args.top_k}_ablated_{excluded_str}_metadata.json"
    else:
        metadata_filename = f"top_{args.top_k}_all_metadata.json"

    metadata_output_path = synthetic_dir / metadata_filename

    with open(metadata_output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_output_path}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Synthetic probe constructed from {len(top_features)} SAE features")
    if excluded_features:
        print(f"Excluded {len(excluded_features)} features: {excluded_features}")
    print(f"Probe dimension: {synthetic_probe.shape[0]}")
    if metadata["cosine_similarity_with_original"] is not None:
        print(f"Similarity to original: {metadata['cosine_similarity_with_original']:+.4f}")

    print("\nFeatures used:")
    for i, feat in enumerate(top_features, 1):
        print(f"{i}. Feature {feat['feature_id']}: {feat['similarity']:+.4f}")
        print(f"   {feat['label']}")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
