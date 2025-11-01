#!/usr/bin/env python3
"""
Download SAE data and compare with probe for a specific layer.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def download_sae_weights(layer_num: int, output_dir: Path) -> Path:
    """Download SAE weights from HuggingFace for a specific layer."""
    print(f"\n{'='*70}")
    print(f"Downloading SAE weights for layer {layer_num}")
    print(f"{'='*70}")

    repo_id = "andyrdt/saes-gpt-oss-20b"
    filename = f"resid_post_layer_{layer_num}/trainer_0/ae.pt"

    print(f"Repository: {repo_id}")
    print(f"File: {filename}")
    print(f"\nDownloading from HuggingFace...")

    # Download the SAE file
    sae_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"✓ Downloaded to: {sae_path}")

    # Load the SAE
    print(f"\nLoading SAE checkpoint...")
    sae = torch.load(sae_path, map_location="cpu")

    # Find decoder weights
    decoder_weights = sae["decoder.weight"]
    print(f"✓ Decoder shape: {decoder_weights.shape}")

    # Convert to numpy and save
    decoder_np = decoder_weights.numpy()
    output_path = output_dir / f"layer_{layer_num}_decoder.npy"
    np.save(output_path, decoder_np)
    print(f"✓ Saved decoder weights to: {output_path}")

    return output_path


def download_labels(layer_num: int, output_dir: Path) -> Path:
    """Download labels/explanations from Neuronpedia S3."""
    print(f"\n{'='*70}")
    print(f"Downloading labels for layer {layer_num}")
    print(f"{'='*70}")

    base_url = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
    layer_id = f"{layer_num}-resid-post-aa"

    # Get list of explanation batch files
    print(f"Fetching list of explanation files...")
    list_url = f"{base_url}/?list-type=2&prefix=v1/gpt-oss-20b/{layer_id}/explanations/"

    response = requests.get(list_url)
    response.raise_for_status()

    # Parse XML to get file names
    import xml.etree.ElementTree as ET
    root = ET.fromstring(response.content)

    namespace = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    keys = [key.text for key in root.findall('.//s3:Key', namespace)]

    # Filter for explanation files
    explanation_files = [k for k in keys if k.endswith('.jsonl.gz')]
    print(f"Found {len(explanation_files)} explanation batch files")

    # Download and parse all explanation files
    labels = {}

    for file_key in tqdm(explanation_files, desc="Downloading labels"):
        file_url = f"{base_url}/{file_key}"

        # Download and decompress
        response = requests.get(file_url)
        response.raise_for_status()

        # Decompress gzip
        decompressed = gzip.decompress(response.content).decode('utf-8')

        # Parse each line as JSON
        for line in decompressed.strip().split('\n'):
            if line:
                data = json.loads(line)
                feature_idx = data['index']
                description = data['description']
                labels[feature_idx] = description

    print(f"\n✓ Downloaded labels for {len(labels)} features")

    # Save labels
    output_path = output_dir / f"layer_{layer_num}_labels.json"
    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"✓ Saved labels to: {output_path}")

    return output_path


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


def compare_probe_to_sae(layer_num: int, sae_dir: Path, probe_dir: Path, output_dir: Path):
    """Compare probe with SAE features."""
    print(f"\n{'='*70}")
    print(f"Comparing probe vs SAE features for layer {layer_num}")
    print(f"{'='*70}")

    # Load probe
    probe_path = probe_dir / f"layer{layer_num:02d}" / "probe.pkl"
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

    # Save results
    results_dir = output_dir / f"layer{layer_num:02d}"
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
    print(f"TOP 10 POSITIVE FEATURES - Layer {layer_num}")
    print(f"{'='*70}")
    for i, feature in enumerate(top_10_positive, 1):
        print(f"{i}. Feature {feature['feature_id']:6d}: {feature['similarity']:+.4f}")
        print(f"   {feature['label']}\n")

    print(f"{'='*70}")
    print(f"TOP 10 NEGATIVE FEATURES - Layer {layer_num}")
    print(f"{'='*70}")
    for i, feature in enumerate(top_10_negative, 1):
        print(f"{i}. Feature {feature['feature_id']:6d}: {feature['similarity']:+.4f}")
        print(f"   {feature['label']}\n")


def main():
    parser = argparse.ArgumentParser(description="Download SAE data and compare with probe for a layer")
    parser.add_argument("--layer", type=int, required=True, help="Layer number (3, 7, 11, 15, 19, 23)")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading, only compare")
    args = parser.parse_args()

    layer_num = args.layer
    sae_dir = Path("gpt_oss_20b_saes")
    probe_dir = Path("probes/liars-bench__convincing-game")
    results_dir = Path("gpt_oss_20b_saes/results")

    print(f"\n{'#'*70}")
    print(f"# Processing Layer {layer_num}")
    print(f"{'#'*70}")

    # Download SAE data if needed
    if not args.skip_download:
        decoder_path = sae_dir / f"layer_{layer_num}_decoder.npy"
        labels_path = sae_dir / f"layer_{layer_num}_labels.json"

        if not decoder_path.exists():
            download_sae_weights(layer_num, sae_dir)
        else:
            print(f"\n✓ Decoder already exists: {decoder_path}")

        if not labels_path.exists():
            download_labels(layer_num, sae_dir)
        else:
            print(f"\n✓ Labels already exist: {labels_path}")

    # Compare probe to SAE
    compare_probe_to_sae(layer_num, sae_dir, probe_dir, results_dir)

    print(f"\n{'='*70}")
    print(f"✓ COMPLETE - Layer {layer_num}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
