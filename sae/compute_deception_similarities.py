#!/usr/bin/env python3
"""
Compute cosine similarities between all probes and the 71 deception-labeled features.

This script:
1. Loads all probes from probes-layer50/
2. Loads the 71 deception features found by search_deception_features.py
3. For each probe, computes cosine similarity with each deception feature
4. Saves top 10 positive and negative similarities per probe
5. Generates summary statistics across all probes
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
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


def main():
    # Load environment variables
    load_dotenv()

    # Configuration
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    layer = 50
    probes_dir = Path("probes-layer50")
    deception_features_file = Path("outputs/deception_features/deception_features.json")
    output_dir = Path("outputs/deception_features")

    print("="*80)
    print("COMPUTING COSINE SIMILARITIES: PROBES vs DECEPTION FEATURES")
    print("="*80)

    # Load deception features
    print(f"\nLoading deception features from {deception_features_file}...")
    with open(deception_features_file, 'r') as f:
        deception_data = json.load(f)

    deception_features = deception_data['features']
    print(f"Loaded {len(deception_features)} deception features")

    # Load SAE decoder
    print(f"\nLoading SAE decoder for layer {layer}...")
    decoder_weights = load_sae_decoder_from_hf(model_name, layer)
    # decoder_weights shape: [hidden_dim, num_features]

    # Extract deception feature directions
    print("\nExtracting deception feature directions from SAE decoder...")
    deception_directions = {}
    for feat in deception_features:
        feat_id = feat['feature_id']
        decoder_vec = decoder_weights[:, feat_id]
        deception_directions[feat_id] = {
            'direction': decoder_vec,
            'label': feat['label']
        }

    # Get all probe files
    probe_files = sorted(probes_dir.glob("*.pkl"))
    # Exclude apollo_probe.pkl if it exists
    probe_files = [p for p in probe_files if p.stem != "apollo_probe"]

    print(f"\nFound {len(probe_files)} probes to analyze:")
    for pf in probe_files:
        print(f"  - {pf.stem}")

    # Store results for all probes
    all_probe_results = {}

    # Process each probe
    for probe_path in probe_files:
        probe_name = probe_path.stem
        print(f"\n{'='*80}")
        print(f"Processing: {probe_name}")
        print('='*80)

        # Load probe
        print(f"Loading probe from {probe_path}...")
        probe_direction, probe_metadata = load_probe(probe_path)
        print(f"  Probe direction shape: {probe_direction.shape}")

        # Compute similarities with all deception features
        print(f"Computing similarities with {len(deception_features)} deception features...")
        similarities = []

        for feat_id, feat_data in tqdm(deception_directions.items(), desc="Computing similarities"):
            sim = compute_cosine_similarity(probe_direction, feat_data['direction'])
            similarities.append({
                "feature_id": feat_id,
                "similarity": sim,
                "label": feat_data['label']
            })

        # Sort by similarity
        similarities_sorted = sorted(similarities, key=lambda x: x["similarity"], reverse=True)

        # Get top 10 positive and negative
        top_10_positive = similarities_sorted[:10]
        top_10_negative = sorted(similarities, key=lambda x: x["similarity"])[:10]

        # Calculate statistics
        similarity_values = np.array([s["similarity"] for s in similarities])
        stats = {
            "mean_similarity": float(similarity_values.mean()),
            "std_similarity": float(similarity_values.std()),
            "max_similarity": float(similarity_values.max()),
            "min_similarity": float(similarity_values.min()),
            "median_similarity": float(np.median(similarity_values)),
        }

        print(f"\nStatistics for {probe_name}:")
        print(f"  Mean similarity: {stats['mean_similarity']:+.4f}")
        print(f"  Std similarity:  {stats['std_similarity']:.4f}")
        print(f"  Max similarity:  {stats['max_similarity']:+.4f}")
        print(f"  Min similarity:  {stats['min_similarity']:+.4f}")

        # Save results for this probe
        probe_output_dir = output_dir / probe_name
        probe_output_dir.mkdir(parents=True, exist_ok=True)

        # Save top 10 positive
        with open(probe_output_dir / "top_10_positive.json", 'w') as f:
            json.dump(top_10_positive, f, indent=2)

        # Save top 10 negative
        with open(probe_output_dir / "top_10_negative.json", 'w') as f:
            json.dump(top_10_negative, f, indent=2)

        # Save all similarities
        with open(probe_output_dir / "all_similarities.json", 'w') as f:
            json.dump(similarities, f, indent=2)

        # Save distribution
        np.save(probe_output_dir / "similarity_distribution.npy", similarity_values)

        # Save probe-specific summary
        probe_summary = {
            "probe_name": probe_name,
            "probe_path": str(probe_path),
            "probe_metadata": probe_metadata,
            "num_deception_features": len(deception_features),
            "statistics": stats,
            "top_3_positive": [
                {
                    "feature_id": s["feature_id"],
                    "similarity": s["similarity"],
                    "label": s["label"]
                }
                for s in top_10_positive[:3]
            ],
            "top_3_negative": [
                {
                    "feature_id": s["feature_id"],
                    "similarity": s["similarity"],
                    "label": s["label"]
                }
                for s in top_10_negative[:3]
            ]
        }

        with open(probe_output_dir / "summary.json", 'w') as f:
            json.dump(probe_summary, f, indent=2)

        print(f"✓ Saved results to {probe_output_dir}/")

        # Store for overall summary
        all_probe_results[probe_name] = {
            "statistics": stats,
            "top_positive_feature": top_10_positive[0],
            "top_negative_feature": top_10_negative[0]
        }

    # Generate overall summary
    print(f"\n{'='*80}")
    print("GENERATING OVERALL SUMMARY")
    print('='*80)

    # Rank probes by mean similarity
    probes_ranked = sorted(
        all_probe_results.items(),
        key=lambda x: x[1]["statistics"]["mean_similarity"],
        reverse=True
    )

    overall_summary = {
        "model": model_name,
        "layer": layer,
        "num_deception_features": len(deception_features),
        "num_probes": len(probe_files),
        "probes_ranked_by_mean_similarity": [
            {
                "rank": i + 1,
                "probe_name": probe_name,
                "mean_similarity": results["statistics"]["mean_similarity"],
                "std_similarity": results["statistics"]["std_similarity"],
                "max_similarity": results["statistics"]["max_similarity"],
                "min_similarity": results["statistics"]["min_similarity"],
                "top_positive_feature": results["top_positive_feature"]
            }
            for i, (probe_name, results) in enumerate(probes_ranked)
        ],
        "overall_statistics": {
            "mean_of_means": float(np.mean([r["statistics"]["mean_similarity"] for r in all_probe_results.values()])),
            "std_of_means": float(np.std([r["statistics"]["mean_similarity"] for r in all_probe_results.values()])),
            "highest_mean": probes_ranked[0][1]["statistics"]["mean_similarity"],
            "lowest_mean": probes_ranked[-1][1]["statistics"]["mean_similarity"],
        }
    }

    # Save overall summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(overall_summary, f, indent=2)

    print(f"\n✓ Saved overall summary to {summary_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("PROBES RANKED BY MEAN SIMILARITY TO DECEPTION FEATURES")
    print('='*80)
    for i, (probe_name, results) in enumerate(probes_ranked, 1):
        print(f"{i:2d}. {probe_name:45s} | Mean: {results['statistics']['mean_similarity']:+.4f} | Max: {results['statistics']['max_similarity']:+.4f}")

    print(f"\n{'='*80}")
    print("COMPLETE!")
    print('='*80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - Individual probe results: {output_dir}/<probe_name>/")
    print(f"  - Overall summary: {summary_file}")


if __name__ == "__main__":
    main()
