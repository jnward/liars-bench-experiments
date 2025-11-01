#!/usr/bin/env python3
"""
Compute cosine similarities between probe directions and Goodfire SAE decoder features.

Usage:
    python compute_sae_similarities.py --probe probes-layer50/single_convincing-game.pkl
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from goodfire import Client
from tqdm import tqdm


def load_probe(probe_path: Path) -> tuple[np.ndarray, dict]:
    """Load probe and extract direction vector.

    Supports two formats:
    1. Sklearn probes (pickle format) - from probe_pipeline/
    2. Apollo probes (torch.save format) - from apollo_probes/

    Returns:
        probe_direction: The probe direction vector (shape: [hidden_dim])
        metadata: Dictionary with probe metadata
    """
    # Try torch.load first (for apollo_probes)
    try:
        probe_data = torch.load(probe_path, map_location="cpu")

        # Apollo probe format - has 'directions' key with torch tensor
        if "directions" in probe_data or "direction" in probe_data:
            direction = probe_data.get("directions", probe_data.get("direction"))
            if torch.is_tensor(direction):
                direction = direction.numpy()
            if direction.ndim == 2:
                direction = direction.flatten()

            metadata = {
                "datasets": probe_data.get("dataset", "unknown"),
                "config": {
                    "reg_coeff": probe_data.get("reg_coeff"),
                    "normalize": probe_data.get("normalize"),
                },
                "cache_files": [],
            }
            return direction, metadata
    except Exception:
        # Not a torch-saved probe, try pickle
        pass

    # Try pickle.load (for sklearn probes from probe_pipeline)
    with open(probe_path, "rb") as f:
        probe_data = pickle.load(f)

    # Extract sklearn model and get coefficients
    if "model" in probe_data:
        # This is a sklearn LogisticRegression model
        model = probe_data["model"]
        direction = model.coef_.flatten()  # Shape: [hidden_dim]

        metadata = {
            "datasets": probe_data.get("datasets", "unknown"),
            "config": probe_data.get("config", {}),
            "cache_files": probe_data.get("cache_files", []),
        }
        return direction, metadata
    else:
        raise ValueError(f"Could not find model or directions in probe file. Keys: {probe_data.keys()}")

    return direction, metadata


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    vec1 = vec1.flatten().astype(np.float32)
    vec2 = vec2.flatten().astype(np.float32)

    # Normalize
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)

    # Compute cosine similarity
    similarity = float(np.dot(vec1_norm, vec2_norm))
    return similarity


def load_sae_decoder_from_hf(model_name: str, layer: int) -> np.ndarray:
    """Load SAE decoder weights from Hugging Face.

    Returns:
        decoder_weights: Numpy array of shape [hidden_dim, num_features]
    """
    from huggingface_hub import hf_hub_download
    import os

    repo_id = f"Goodfire/{model_name.split('/')[-1]}-SAE-l{layer}"
    filename = f"{model_name.split('/')[-1]}-SAE-l{layer}.pt"

    print(f"Downloading SAE from HuggingFace: {repo_id}...")
    sae_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=os.environ.get("HF_HOME"),
    )
    print(f"Loading SAE from {sae_path}...")

    sae = torch.load(sae_path, map_location="cpu")
    decoder_weights = sae["decoder_linear.weight"]  # Shape: [hidden_dim, num_features]
    print(f"Decoder weights shape: {decoder_weights.shape}")

    return decoder_weights.numpy()


def fetch_labels_from_goodfire(client: Client, model_name: str, feature_indices: list[int]) -> dict[int, str]:
    """Fetch labels for features using Goodfire API.

    Uses the features.lookup() method to retrieve universal labels for features by their SAE indices.

    Returns:
        Dictionary mapping feature_idx -> label
    """
    print(f"\nFetching labels for {len(feature_indices)} features from Goodfire API...")

    try:
        # Use lookup() to directly fetch features by their SAE indices
        features = client.features.lookup(indices=feature_indices, model=model_name)

        labels = {}
        for idx, feature in features.items():
            labels[idx] = feature.label

        print(f"Successfully fetched {len(labels)}/{len(feature_indices)} labels")

        # Fill in any missing labels (shouldn't happen, but just in case)
        for feat_id in feature_indices:
            if feat_id not in labels:
                labels[feat_id] = f"Feature {feat_id} (label not available)"

        return labels

    except Exception as e:
        print(f"Error fetching labels: {e}")
        print("Returning placeholder labels...")
        return {feat_id: f"Feature {feat_id} (label fetch failed)" for feat_id in feature_indices}


def main():
    parser = argparse.ArgumentParser(
        description="Compute cosine similarities between probe and SAE features"
    )
    parser.add_argument(
        "--probe",
        type=str,
        required=True,
        help="Path to probe .pkl file (e.g., probes-layer50/single_convincing-game.pkl)",
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
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="Skip fetching labels from Goodfire API",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Load probe
    probe_path = Path(args.probe)
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe file not found: {probe_path}")

    print(f"Loading probe from {probe_path}...")
    probe_direction, probe_metadata = load_probe(probe_path)
    print(f"  Probe direction shape: {probe_direction.shape}")
    print(f"  Probe metadata: {probe_metadata}")

    # Load SAE decoder weights from HuggingFace
    decoder_weights = load_sae_decoder_from_hf(args.model, args.layer)
    # decoder_weights shape: [hidden_dim, num_features]

    # Compute cosine similarities for all features
    print("\nComputing cosine similarities for all features...")
    num_features = decoder_weights.shape[1]

    similarities = []
    for feature_idx in tqdm(range(num_features), desc="Computing similarities"):
        decoder_vec = decoder_weights[:, feature_idx]  # Get feature column
        sim = compute_cosine_similarity(probe_direction, decoder_vec)
        similarities.append({
            "feature_id": feature_idx,
            "similarity": sim,
            "label": None,  # Will fill in for top features
        })

    # Sort by similarity value (not absolute value)
    similarities_sorted = sorted(similarities, key=lambda x: x["similarity"], reverse=True)

    # Get top 10 positive (most aligned)
    top_10_positive = similarities_sorted[:10]

    # Get top 10 negative (most anti-correlated)
    top_10_negative = sorted(similarities, key=lambda x: x["similarity"])[:10]

    # Fetch labels for top features
    if not args.skip_labels:
        import os
        api_key = os.getenv("GOODFIRE_API_KEY")
        if api_key:
            try:
                client = Client(api_key=api_key)
                all_top_indices = [s["feature_id"] for s in top_10_positive + top_10_negative]
                labels = fetch_labels_from_goodfire(client, args.model, all_top_indices)

                # Update features with labels
                for item in top_10_positive + top_10_negative:
                    item["label"] = labels.get(item["feature_id"], f"Feature {item['feature_id']} (label not available)")
            except Exception as e:
                print(f"\nWarning: Could not fetch labels: {e}")
                print("Continuing without labels...")
                for item in top_10_positive + top_10_negative:
                    item["label"] = f"Feature {item['feature_id']} (label not available)"
        else:
            print("\nNo GOODFIRE_API_KEY found, skipping label fetch")
            for item in top_10_positive + top_10_negative:
                item["label"] = f"Feature {item['feature_id']} (label not available)"
    else:
        print("\nSkipping label fetch (--skip-labels)")
        for item in top_10_positive + top_10_negative:
            item["label"] = f"Feature {item['feature_id']} (label not fetched)"

    # Create output directory
    probe_name = probe_path.stem  # e.g., "single_convincing-game"
    output_dir = Path(args.output_dir) / probe_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save top 10 positive
    top_10_positive_path = output_dir / "top_10_positive.json"
    with open(top_10_positive_path, "w") as f:
        json.dump(top_10_positive, f, indent=2)
    print(f"\nSaved top 10 positive features to: {top_10_positive_path}")

    # Save top 10 negative
    top_10_negative_path = output_dir / "top_10_negative.json"
    with open(top_10_negative_path, "w") as f:
        json.dump(top_10_negative, f, indent=2)
    print(f"Saved top 10 negative features to: {top_10_negative_path}")

    # Print top 10 positive
    print("\n" + "="*80)
    print(f"TOP 10 POSITIVE FEATURES (Most aligned with {probe_name})")
    print("="*80)
    for i, item in enumerate(top_10_positive, 1):
        print(f"{i}. Feature {item['feature_id']}: {item['similarity']:+.4f}")
        print(f"   {item['label']}\n")

    # Print top 10 negative
    print("="*80)
    print(f"TOP 10 NEGATIVE FEATURES (Most anti-correlated with {probe_name})")
    print("="*80)
    for i, item in enumerate(top_10_negative, 1):
        print(f"{i}. Feature {item['feature_id']}: {item['similarity']:+.4f}")
        print(f"   {item['label']}\n")

    # Save all similarities for distribution plot
    all_similarities_path = output_dir / "all_similarities.json"
    with open(all_similarities_path, "w") as f:
        json.dump(similarities, f, indent=2)
    print(f"Saved all {len(similarities)} similarities to: {all_similarities_path}")

    # Save just the similarity values for quick plotting
    similarity_values_path = output_dir / "similarity_distribution.npy"
    similarity_values = np.array([s["similarity"] for s in similarities])
    np.save(similarity_values_path, similarity_values)
    print(f"Saved similarity distribution to: {similarity_values_path}")

    # Save summary stats
    summary_path = output_dir / "summary.json"
    summary = {
        "probe_path": str(probe_path),
        "probe_metadata": probe_metadata,
        "model": args.model,
        "layer": args.layer,
        "num_features": num_features,
        "probe_dimension": int(probe_direction.shape[0]),
        "top_10_positive": [
            {
                "rank": i+1,
                "feature_id": s["feature_id"],
                "similarity": s["similarity"],
                "label": s["label"],
            }
            for i, s in enumerate(top_10_positive)
        ],
        "top_10_negative": [
            {
                "rank": i+1,
                "feature_id": s["feature_id"],
                "similarity": s["similarity"],
                "label": s["label"],
            }
            for i, s in enumerate(top_10_negative)
        ],
        "statistics": {
            "mean_similarity": float(similarity_values.mean()),
            "std_similarity": float(similarity_values.std()),
            "max_similarity": float(similarity_values.max()),
            "min_similarity": float(similarity_values.min()),
            "median_similarity": float(np.median(similarity_values)),
        }
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
