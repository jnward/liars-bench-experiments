#!/usr/bin/env python3
"""
Download GPT-OSS-20B SAE weights and labels from HuggingFace and Neuronpedia.
"""

import json
import gzip
from pathlib import Path

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def download_sae_weights(layer_num=7, output_dir="gpt_oss_20b_saes"):
    """Download SAE weights from HuggingFace for a specific layer."""
    print(f"\n{'='*70}")
    print(f"STEP 2: Downloading SAE weights for layer {layer_num}")
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

    print(f"Available keys in SAE checkpoint: {list(sae.keys())}")

    # Find decoder weights (try common key names)
    decoder_keys = [k for k in sae.keys() if 'decoder' in k.lower() and 'weight' in k.lower()]
    print(f"Decoder weight keys: {decoder_keys}")

    if decoder_keys:
        decoder_key = decoder_keys[0]
        decoder_weights = sae[decoder_key]
    elif 'W_dec' in sae:
        decoder_weights = sae['W_dec']
    elif 'decoder' in sae:
        decoder_weights = sae['decoder']
    else:
        # Print all keys and their shapes to debug
        print("\nAll checkpoint keys and shapes:")
        for key, value in sae.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        raise ValueError("Could not find decoder weights in SAE checkpoint")

    print(f"\n✓ Found decoder weights with key: {decoder_key if decoder_keys else 'W_dec/decoder'}")
    print(f"Decoder shape: {decoder_weights.shape}")

    # Convert to numpy
    decoder_np = decoder_weights.numpy()

    # Save
    output_path = Path(output_dir) / f"layer_{layer_num}_decoder.npy"
    np.save(output_path, decoder_np)
    print(f"\n✓ Saved decoder weights to: {output_path}")

    return decoder_np, output_path


def download_labels(layer_num=7, output_dir="gpt_oss_20b_saes"):
    """Download labels/explanations from Neuronpedia S3."""
    print(f"\n{'='*70}")
    print(f"STEP 3: Downloading labels for layer {layer_num}")
    print(f"{'='*70}")

    base_url = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
    layer_id = f"{layer_num}-resid-post-aa"

    # First, get list of explanation batch files
    print(f"Fetching list of explanation files...")
    list_url = f"{base_url}/?list-type=2&prefix=v1/gpt-oss-20b/{layer_id}/explanations/"

    response = requests.get(list_url)
    response.raise_for_status()

    # Parse XML to get file names
    import xml.etree.ElementTree as ET
    root = ET.fromstring(response.content)

    # Find all Key elements (file paths)
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
    output_path = Path(output_dir) / f"layer_{layer_num}_labels.json"
    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"✓ Saved labels to: {output_path}")

    return labels, output_path


def print_summary(decoder_weights, labels, layer_num=7):
    """Print summary of downloaded data."""
    print(f"\n{'='*70}")
    print(f"STEP 4: Summary for Layer {layer_num}")
    print(f"{'='*70}")

    print(f"\nDecoder Weights:")
    print(f"  Shape: {decoder_weights.shape}")
    print(f"  Hidden dim: {decoder_weights.shape[0]}")
    print(f"  Num features: {decoder_weights.shape[1]}")
    print(f"  Dtype: {decoder_weights.dtype}")

    print(f"\nLabels:")
    print(f"  Total labels: {len(labels)}")

    # Show some example labels
    print(f"\nExample features with labels:")
    print(f"{'='*70}")
    sample_indices = sorted([int(idx) for idx in labels.keys()])[:10]
    for idx in sample_indices:
        idx_str = str(idx)
        label = labels[idx_str]
        print(f"  Feature {idx:5d}: {label}")

    print(f"\n{'='*70}")
    print(f"✓ DOWNLOAD COMPLETE!")
    print(f"{'='*70}")


def main():
    layer_num = 7
    output_dir = "gpt_oss_20b_saes"

    print(f"\n{'#'*70}")
    print(f"# Downloading GPT-OSS-20B Layer {layer_num} SAE Data")
    print(f"{'#'*70}")

    # Step 1: Directory already created
    print(f"\nOutput directory: {output_dir}/")

    # Step 2: Download SAE weights
    decoder_weights, weights_path = download_sae_weights(layer_num, output_dir)

    # Step 3: Download labels
    labels, labels_path = download_labels(layer_num, output_dir)

    # Step 4: Print summary
    print_summary(decoder_weights, labels, layer_num)


if __name__ == "__main__":
    main()
