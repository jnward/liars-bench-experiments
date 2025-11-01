#!/usr/bin/env python3
"""
Compare probe with SAE features for a specific dataset and layer.
Saves results to dataset-specific directory.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


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


def compare_probe_to_sae(dataset_name: str, layer_num: int, sae_dir: Path, probe_dir: Path, output_base_dir: Path):
    """Compare probe with SAE features."""
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name} | Layer {layer_num}")
    print(f"{'='*70}")

    # Load probe
    probe_path = probe_dir / dataset_name / f"layer{layer_num:02d}" / "probe.pkl"
    print(f"\nLoading probe from: {probe_path}")
    probe_data = torch.load(probe_path, map_location="cpu")
    probe_direction = probe_data["directions"].flatten().numpy()
    print(f"  Probe direction shape: {probe_direction.shape}")

    # Load SAE decoder
    decoder_path = sae_dir / f"layer_{layer_num}_decoder.npy"
    print(f"\nLoading SAE decoder from: {decoder_path}")
    sae_decoder = np.load(decoder_path)
    print(f"  Decoder shape: {sae_decoder.shape}")

    # Load labels
    labels_path = sae_dir / f"layer_{layer_num}_labels.json"
    print(f"\nLoading labels from: {labels_path}")
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    print(f"  Total labels: {len(labels)}")

    # Compute similarities
    print("\nComputing cosine similarities...")
    num_features = sae_decoder.shape[1]
    similarities = []

    for feature_idx in tqdm(range(num_features), desc="Computing similarities"):
        feature_vec = sae_decoder[:, feature_idx]
        sim = compute_cosine_similarity(probe_direction, feature_vec)
        similarities.append({
            "feature_id": feature_idx,
            "similarity": sim,
            "label": None,
        })

    # Find top features
    print("\nFinding top features...")
    similarities_sorted = sorted(similarities, key=lambda x: x["similarity"], reverse=True)

    top_10_positive = similarities_sorted[:10]
    top_10_negative = sorted(similarities, key=lambda x: x["similarity"])[:10]

    # Add labels
    print("Adding labels...")
    for feature in top_10_positive + top_10_negative:
        feature_id = str(feature["feature_id"])
        if feature_id in labels:
            feature["label"] = labels[feature_id]
        else:
            feature["label"] = f"Feature {feature_id} (no label available)"

    # Save results to dataset-specific directory
    results_dir = output_base_dir / dataset_name / f"layer{layer_num:02d}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to: {results_dir}")

    # Top 10 positive
    with open(results_dir / "top_10_positive.json", 'w') as f:
        json.dump(top_10_positive, f, indent=2)

    # Top 10 negative
    with open(results_dir / "top_10_negative.json", 'w') as f:
        json.dump(top_10_negative, f, indent=2)

    # All similarities
    with open(results_dir / "all_similarities.json", 'w') as f:
        json.dump(similarities, f, indent=2)

    # Similarity distribution
    similarity_values = np.array([s["similarity"] for s in similarities])
    np.save(results_dir / "similarity_distribution.npy", similarity_values)

    # Summary
    summary = {
        "dataset": dataset_name,
        "probe_path": str(probe_path),
        "layer": layer_num,
        "num_features": len(similarities),
        "statistics": {
            "mean_similarity": float(similarity_values.mean()),
            "std_similarity": float(similarity_values.std()),
            "max_similarity": float(similarity_values.max()),
            "min_similarity": float(similarity_values.min()),
            "median_similarity": float(np.median(similarity_values)),
        },
        "top_10_positive": top_10_positive,
        "top_10_negative": top_10_negative,
    }
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print results
    print(f"\n{'='*70}")
    print(f"TOP 10 POSITIVE FEATURES - {dataset_name} Layer {layer_num}")
    print(f"{'='*70}")
    for i, feature in enumerate(top_10_positive, 1):
        print(f"{i}. Feature {feature['feature_id']:6d}: {feature['similarity']:+.4f}")
        print(f"   {feature['label']}\n")

    print(f"{'='*70}")
    print(f"TOP 10 NEGATIVE FEATURES - {dataset_name} Layer {layer_num}")
    print(f"{'='*70}")
    for i, feature in enumerate(top_10_negative, 1):
        print(f"{i}. Feature {feature['feature_id']:6d}: {feature['similarity']:+.4f}")
        print(f"   {feature['label']}\n")

    print(f"{'='*70}")
    print(f"✓ COMPLETE - {dataset_name} Layer {layer_num}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Compare probe with SAE for a dataset and layer")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., liars-bench__hpkr)")
    parser.add_argument("--layer", type=int, required=True, help="Layer number (3, 7, 11, 15, 19, 23)")
    args = parser.parse_args()

    dataset_name = args.dataset
    layer_num = args.layer
    sae_dir = Path("gpt_oss_20b_saes")
    probe_dir = Path("probes")
    results_base_dir = Path("gpt_oss_20b_saes/results")

    print(f"\n{'#'*70}")
    print(f"# Comparing Probe vs SAE: {dataset_name} Layer {layer_num}")
    print(f"{'#'*70}")

    compare_probe_to_sae(dataset_name, layer_num, sae_dir, probe_dir, results_base_dir)


if __name__ == "__main__":
    main()
