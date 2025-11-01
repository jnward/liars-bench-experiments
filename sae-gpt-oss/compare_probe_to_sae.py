#!/usr/bin/env python3
"""
Compare GPT-OSS-20B probe directions with Neuronpedia SAE features.
Compute cosine similarities and find top aligned/anti-correlated features.
"""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def load_probe(probe_path: Path) -> np.ndarray:
    """Load probe direction from apollo_probes format."""
    print(f"\nLoading probe from: {probe_path}")
    probe_data = torch.load(probe_path, map_location="cpu")

    # Extract direction and flatten
    direction = probe_data["directions"].flatten().numpy()  # Shape: (hidden_dim,)

    print(f"  Probe direction shape: {direction.shape}")
    print(f"  Dataset: {probe_data.get('dataset', 'unknown')}")
    print(f"  Layer: {probe_data.get('layers', 'unknown')}")

    return direction


def load_sae_decoder(decoder_path: Path) -> np.ndarray:
    """Load SAE decoder weights."""
    print(f"\nLoading SAE decoder from: {decoder_path}")
    decoder = np.load(decoder_path)

    print(f"  Decoder shape: {decoder.shape}")
    print(f"  Hidden dim: {decoder.shape[0]}")
    print(f"  Num features: {decoder.shape[1]}")

    return decoder


def load_labels(labels_path: Path) -> dict:
    """Load feature labels from Neuronpedia."""
    print(f"\nLoading labels from: {labels_path}")
    with open(labels_path, 'r') as f:
        labels = json.load(f)

    print(f"  Total labels: {len(labels)}")

    return labels


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


def compute_all_similarities(probe_direction: np.ndarray, sae_decoder: np.ndarray) -> list:
    """Compute cosine similarities between probe and all SAE features."""
    print("\nComputing cosine similarities...")

    num_features = sae_decoder.shape[1]
    similarities = []

    for feature_idx in tqdm(range(num_features), desc="Computing similarities"):
        feature_vec = sae_decoder[:, feature_idx]
        sim = compute_cosine_similarity(probe_direction, feature_vec)
        similarities.append({
            "feature_id": feature_idx,
            "similarity": sim,
            "label": None,  # Will fill in later
        })

    return similarities


def add_labels_to_features(features: list, labels: dict) -> list:
    """Add Neuronpedia labels to features."""
    print("\nAdding labels to features...")

    for feature in features:
        feature_id = feature["feature_id"]
        feature_id_str = str(feature_id)

        if feature_id_str in labels:
            feature["label"] = labels[feature_id_str]
        else:
            feature["label"] = f"Feature {feature_id} (no label available)"

    return features


def save_results(
    similarities: list,
    top_positive: list,
    top_negative: list,
    output_dir: Path,
    probe_path: Path,
    layer: int,
):
    """Save all results to JSON files."""
    print(f"\nSaving results to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save top 10 positive
    top_positive_path = output_dir / "top_10_positive.json"
    with open(top_positive_path, 'w') as f:
        json.dump(top_positive, f, indent=2)
    print(f"  ✓ Saved: {top_positive_path}")

    # Save top 10 negative
    top_negative_path = output_dir / "top_10_negative.json"
    with open(top_negative_path, 'w') as f:
        json.dump(top_negative, f, indent=2)
    print(f"  ✓ Saved: {top_negative_path}")

    # Save all similarities (might be large)
    all_similarities_path = output_dir / "all_similarities.json"
    with open(all_similarities_path, 'w') as f:
        json.dump(similarities, f, indent=2)
    print(f"  ✓ Saved: {all_similarities_path}")

    # Save similarity distribution as numpy array
    similarity_values = np.array([s["similarity"] for s in similarities])
    similarity_dist_path = output_dir / "similarity_distribution.npy"
    np.save(similarity_dist_path, similarity_values)
    print(f"  ✓ Saved: {similarity_dist_path}")

    # Save summary
    summary = {
        "probe_path": str(probe_path),
        "layer": layer,
        "num_features": len(similarities),
        "statistics": {
            "mean_similarity": float(similarity_values.mean()),
            "std_similarity": float(similarity_values.std()),
            "max_similarity": float(similarity_values.max()),
            "min_similarity": float(similarity_values.min()),
            "median_similarity": float(np.median(similarity_values)),
        },
        "top_10_positive": top_positive,
        "top_10_negative": top_negative,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved: {summary_path}")


def print_results(top_positive: list, top_negative: list, layer: int):
    """Print top features to console."""
    print(f"\n{'='*80}")
    print(f"TOP 10 POSITIVE FEATURES (Most aligned with probe) - Layer {layer}")
    print(f"{'='*80}")
    for i, feature in enumerate(top_positive, 1):
        print(f"{i}. Feature {feature['feature_id']:6d}: {feature['similarity']:+.4f}")
        print(f"   {feature['label']}\n")

    print(f"{'='*80}")
    print(f"TOP 10 NEGATIVE FEATURES (Most anti-correlated with probe) - Layer {layer}")
    print(f"{'='*80}")
    for i, feature in enumerate(top_negative, 1):
        print(f"{i}. Feature {feature['feature_id']:6d}: {feature['similarity']:+.4f}")
        print(f"   {feature['label']}\n")


def main():
    # Configuration
    layer = 7
    probe_path = Path("liars-bench__convincing-game") / f"layer{layer:02d}" / "probe.pkl"
    decoder_path = Path("gpt_oss_20b_saes") / f"layer_{layer}_decoder.npy"
    labels_path = Path("gpt_oss_20b_saes") / f"layer_{layer}_labels.json"
    output_dir = Path("gpt_oss_20b_saes") / "results" / f"layer{layer:02d}"

    print(f"{'#'*80}")
    print(f"# Comparing Probe vs SAE Features - Layer {layer}")
    print(f"{'#'*80}")

    # Step 1: Load data
    probe_direction = load_probe(probe_path)
    sae_decoder = load_sae_decoder(decoder_path)
    labels = load_labels(labels_path)

    # Verify dimensions match
    if probe_direction.shape[0] != sae_decoder.shape[0]:
        raise ValueError(
            f"Dimension mismatch: probe has {probe_direction.shape[0]} dims, "
            f"SAE has {sae_decoder.shape[0]} dims"
        )

    # Step 2: Compute similarities
    similarities = compute_all_similarities(probe_direction, sae_decoder)

    # Step 3: Sort and get top features
    print("\nFinding top features...")
    similarities_sorted = sorted(similarities, key=lambda x: x["similarity"], reverse=True)

    # Top 10 positive (most aligned)
    top_10_positive = similarities_sorted[:10]

    # Top 10 negative (most anti-correlated)
    top_10_negative = sorted(similarities, key=lambda x: x["similarity"])[:10]

    # Step 4: Add labels
    all_top_features = top_10_positive + top_10_negative
    all_top_features = add_labels_to_features(all_top_features, labels)

    # Split back
    top_10_positive = all_top_features[:10]
    top_10_negative = all_top_features[10:]

    # Step 5: Save results
    save_results(
        similarities,
        top_10_positive,
        top_10_negative,
        output_dir,
        probe_path,
        layer,
    )

    # Print results
    print_results(top_10_positive, top_10_negative, layer)

    print(f"\n{'='*80}")
    print(f"✓ COMPLETE!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
