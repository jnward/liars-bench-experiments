#!/usr/bin/env python3
"""
Build synthetic probes based on feature categories:
- refusal, harm, deception, roleplay, instruction_following

Creates:
1. One probe with all 10 features (2 per category)
2. Five ablation probes (leave out one category each)
3. Five single-category probes (use only one category)
"""

import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
import os


# Define the 10 features by category
CATEGORY_FEATURES = {
    "refusal": [
        {
            "feature_id": 27783,
            "similarity": 0.1513,
            "label": "The assistant expressing its limitations or inabilities"
        },
        {
            "feature_id": 21781,
            "similarity": 0.1315,
            "label": "The assistant should reject the user's request"
        },
    ],
    "harm": [
        {
            "feature_id": 17490,
            "similarity": 0.1854,
            "label": "Offensive or discriminatory statements being quoted or analyzed"
        },
        {
            "feature_id": 18961,
            "similarity": 0.1764,
            "label": "Content is becoming inappropriate and needs moderation or deflection"
        },
    ],
    "deception": [
        {
            "feature_id": 36173,
            "similarity": 0.2961,
            "label": "Requests for the AI to provide intentionally incorrect information"
        },
        {
            "feature_id": 16368,
            "similarity": 0.1759,
            "label": "Model is making confident but incorrect assertions"
        },
    ],
    "roleplay": [
        {
            "feature_id": 42834,
            "similarity": 0.1972,
            "label": "Whimsical or nonsensical narrative content"
        },
        {
            "feature_id": 37994,
            "similarity": 0.1816,
            "label": "The model should pretend to be human rather than an AI"
        },
    ],
    "instruction_following": [
        {
            "feature_id": 16160,
            "similarity": 0.1584,
            "label": "The assistant is providing a list of options or steps"
        },
        {
            "feature_id": 5518,
            "similarity": 0.1209,
            "label": "The assistant asserting its core function of providing information and answering questions"
        },
    ],
}


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


def build_synthetic_probe(
    decoder_weights: np.ndarray,
    features: list[dict],
    probe_name: str,
) -> tuple[np.ndarray, dict]:
    """Build synthetic probe from features."""
    print(f"\nBuilding: {probe_name}")
    print(f"  Using {len(features)} features")

    hidden_dim = decoder_weights.shape[0]
    synthetic_probe = np.zeros(hidden_dim, dtype=np.float32)

    # Weight and sum decoder features
    for feat in features:
        feat_id = feat["feature_id"]
        similarity = feat["similarity"]
        decoder_feature = decoder_weights[:, feat_id]
        synthetic_probe += similarity * decoder_feature

    # Normalize
    probe_norm = np.linalg.norm(synthetic_probe)
    synthetic_probe = synthetic_probe / (probe_norm + 1e-8)

    metadata = {
        "probe_name": probe_name,
        "num_features": len(features),
        "feature_ids": [f["feature_id"] for f in features],
        "feature_weights": [f["similarity"] for f in features],
        "feature_labels": [f["label"] for f in features],
        "pre_norm_magnitude": float(probe_norm),
    }

    return synthetic_probe, metadata


def main():
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    layer = 50

    # Create output directory
    output_dir = Path("outputs/categories")
    output_dir.mkdir(exist_ok=True)

    synthetic_dir = output_dir / "synthetic_probes"
    synthetic_dir.mkdir(exist_ok=True)

    # Load SAE decoder
    decoder_weights = load_sae_decoder_from_hf(model_name, layer)

    print("\n" + "="*80)
    print("BUILDING CATEGORY-BASED SYNTHETIC PROBES")
    print("="*80)

    # 1. Build probe with ALL categories (10 features)
    all_features = []
    for category, features in CATEGORY_FEATURES.items():
        all_features.extend(features)

    probe, metadata = build_synthetic_probe(decoder_weights, all_features, "all_categories")

    torch.save({
        "direction": torch.from_numpy(probe),
        "metadata": metadata,
    }, synthetic_dir / "all_categories.pkl")

    with open(synthetic_dir / "all_categories_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved: all_categories.pkl")

    # 2. Build ablation probes (leave out one category each)
    print("\n" + "="*80)
    print("BUILDING ABLATION PROBES (leave out one category)")
    print("="*80)

    for exclude_category in CATEGORY_FEATURES.keys():
        features = []
        for category, feats in CATEGORY_FEATURES.items():
            if category != exclude_category:
                features.extend(feats)

        probe_name = f"no_{exclude_category}"
        probe, metadata = build_synthetic_probe(decoder_weights, features, probe_name)
        metadata["excluded_category"] = exclude_category

        torch.save({
            "direction": torch.from_numpy(probe),
            "metadata": metadata,
        }, synthetic_dir / f"{probe_name}.pkl")

        with open(synthetic_dir / f"{probe_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Saved: {probe_name}.pkl")

    # 3. Build single-category probes
    print("\n" + "="*80)
    print("BUILDING SINGLE-CATEGORY PROBES")
    print("="*80)

    for category, features in CATEGORY_FEATURES.items():
        probe_name = f"only_{category}"
        probe, metadata = build_synthetic_probe(decoder_weights, features, probe_name)
        metadata["category"] = category

        torch.save({
            "direction": torch.from_numpy(probe),
            "metadata": metadata,
        }, synthetic_dir / f"{probe_name}.pkl")

        with open(synthetic_dir / f"{probe_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Saved: {probe_name}.pkl")

    # Save category definitions
    category_def_path = output_dir / "category_definitions.json"
    with open(category_def_path, "w") as f:
        json.dump(CATEGORY_FEATURES, f, indent=2)
    print(f"\n✓ Saved category definitions to: {category_def_path}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Created 11 synthetic probes in: {synthetic_dir}")
    print("\n1 probe with all categories:")
    print("  - all_categories.pkl (10 features)")
    print("\n5 ablation probes (8 features each):")
    for cat in CATEGORY_FEATURES.keys():
        print(f"  - no_{cat}.pkl")
    print("\n5 single-category probes (2 features each):")
    for cat in CATEGORY_FEATURES.keys():
        print(f"  - only_{cat}.pkl")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
