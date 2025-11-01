#!/usr/bin/env python3
"""
Build semantic probes from label_validation_sample.csv.

Creates:
1. One baseline probe with all 7 categories
2. Seven leave-one-out probes (remove one category each)
3. Seven single-category probes (use only one category)
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
import os
import json


def load_sae_decoder_from_hf(model_name: str, layer: int) -> np.ndarray:
    """Load SAE decoder weights from Hugging Face."""
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


def build_probe(decoder_weights: np.ndarray, feature_ids: list[int], probe_name: str) -> tuple[np.ndarray, dict]:
    """Build synthetic probe with uniform weights."""
    print(f"\nBuilding: {probe_name}")
    print(f"  Using {len(feature_ids)} features")

    hidden_dim = decoder_weights.shape[0]
    synthetic_probe = np.zeros(hidden_dim, dtype=np.float32)

    # Use uniform weight (1.0) for all features
    weight = 1.0
    for feat_id in feature_ids:
        decoder_feature = decoder_weights[:, feat_id]
        synthetic_probe += weight * decoder_feature

    # Normalize
    probe_norm = np.linalg.norm(synthetic_probe)
    synthetic_probe = synthetic_probe / (probe_norm + 1e-8)

    metadata = {
        "probe_name": probe_name,
        "num_features": len(feature_ids),
        "feature_ids": feature_ids,
        "weight_scheme": "uniform (1.0)",
        "pre_norm_magnitude": float(probe_norm),
    }

    return synthetic_probe, metadata


def main():
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    layer = 50

    # Load features from CSV
    df = pd.read_csv('semantic_probe_results_v3/label_validation_sample.csv')

    # Group by category
    features_by_category = {}
    for category, group in df.groupby('category'):
        features_by_category[category] = group['feature_idx'].tolist()

    print("="*80)
    print("BUILDING SEMANTIC PROBES FROM CSV")
    print("="*80)
    print(f"\nLoaded {len(df)} features across {len(features_by_category)} categories:")
    for cat, feats in features_by_category.items():
        print(f"  {cat}: {len(feats)} features")

    # Create output directory
    output_dir = Path("outputs/semantic_probes")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SAE decoder
    decoder_weights = load_sae_decoder_from_hf(model_name, layer)

    # 1. Build baseline probe with ALL categories
    print("\n" + "="*80)
    print("1. BASELINE: All categories")
    print("="*80)

    all_features = []
    for feats in features_by_category.values():
        all_features.extend(feats)

    probe, metadata = build_probe(decoder_weights, all_features, "all_categories")
    metadata["categories_included"] = list(features_by_category.keys())

    torch.save({
        "direction": torch.from_numpy(probe),
        "metadata": metadata,
    }, output_dir / "all_categories.pkl")

    with open(output_dir / "all_categories_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved: all_categories.pkl")

    # 2. Build leave-one-out probes
    print("\n" + "="*80)
    print("2. LEAVE-ONE-OUT: Remove one category at a time")
    print("="*80)

    for exclude_category in features_by_category.keys():
        features = []
        included_cats = []

        for cat, feats in features_by_category.items():
            if cat != exclude_category:
                features.extend(feats)
                included_cats.append(cat)

        probe_name = f"no_{exclude_category}"
        probe, metadata = build_probe(decoder_weights, features, probe_name)
        metadata["excluded_category"] = exclude_category
        metadata["categories_included"] = included_cats

        torch.save({
            "direction": torch.from_numpy(probe),
            "metadata": metadata,
        }, output_dir / f"{probe_name}.pkl")

        with open(output_dir / f"{probe_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Saved: {probe_name}.pkl")

    # 3. Build single-category probes
    print("\n" + "="*80)
    print("3. SINGLE-CATEGORY: Use only one category")
    print("="*80)

    for category, features in features_by_category.items():
        probe_name = f"only_{category}"
        probe, metadata = build_probe(decoder_weights, features, probe_name)
        metadata["category"] = category
        metadata["categories_included"] = [category]

        torch.save({
            "direction": torch.from_numpy(probe),
            "metadata": metadata,
        }, output_dir / f"{probe_name}.pkl")

        with open(output_dir / f"{probe_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Saved: {probe_name}.pkl")

    # Save category definitions
    category_summary = {
        cat: {
            "num_features": len(feats),
            "feature_ids": feats
        }
        for cat, feats in features_by_category.items()
    }

    with open(output_dir / "category_definitions.json", "w") as f:
        json.dump(category_summary, f, indent=2)

    print(f"\n✓ Saved category definitions to: {output_dir / 'category_definitions.json'}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Created 15 semantic probes in: {output_dir}")
    print("\n1 baseline probe:")
    print(f"  - all_categories.pkl ({len(all_features)} features)")
    print("\n7 leave-one-out probes:")
    for cat in features_by_category.keys():
        num_feats = len(all_features) - len(features_by_category[cat])
        print(f"  - no_{cat}.pkl ({num_feats} features)")
    print("\n7 single-category probes:")
    for cat, feats in features_by_category.items():
        print(f"  - only_{cat}.pkl ({len(feats)} features)")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
