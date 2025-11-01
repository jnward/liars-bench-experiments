#!/usr/bin/env python3
"""
Explore Neuronpedia API to find GPT-OSS-20B SAE model and test functionality.
"""

import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
NEURONPEDIA_API_KEY = os.getenv("NEURONPEDIA_API_KEY")
BASE_URL = "https://www.neuronpedia.org/api"

def get_headers():
    """Get headers with API key for authenticated requests."""
    return {
        "Content-Type": "application/json",
        "x-api-key": NEURONPEDIA_API_KEY
    }

def search_models(query="gpt"):
    """Search for models on Neuronpedia."""
    print(f"\n{'='*80}")
    print(f"SEARCHING FOR MODELS: '{query}'")
    print(f"{'='*80}")

    # Try different approaches to find models

    # Approach 1: Try to get a feature from gpt-oss-20b directly
    # (if it exists, this should work; if not, we'll get an error)
    test_model_ids = [
        "gpt-oss-20b",
        "gpt-oss-20B",
        "openai/gpt-oss-20b",
        "gpt-oss",
        "gpt2-small",  # Known working model for comparison
    ]

    for model_id in test_model_ids:
        print(f"\nTrying to access model: {model_id}")
        # Try to fetch a sample feature from layer 0, index 0
        url = f"{BASE_URL}/feature/{model_id}/0-res/0"
        try:
            response = requests.get(url, headers=get_headers())
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                print(f"  ✓ SUCCESS! Model '{model_id}' exists!")
                data = response.json()
                print(f"  Sample feature data keys: {list(data.keys())[:5]}")
                return model_id, data
            else:
                print(f"  ✗ Model not found or error: {response.text[:200]}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    return None, None

def list_model_layers(model_id):
    """Try to discover available SAE layers for a model."""
    print(f"\n{'='*80}")
    print(f"DISCOVERING LAYERS FOR MODEL: {model_id}")
    print(f"{'='*80}")

    # Common SAE naming patterns
    layer_patterns = [
        "{layer}-res",
        "{layer}-res-jb",
        "{layer}-mlp",
        "{layer}-attn",
        "{layer}-gemmascope-res-16k",
    ]

    available_layers = []

    # Try layers 0-20 (reasonable range for most models)
    for layer_num in range(21):
        for pattern in layer_patterns:
            layer_id = pattern.format(layer=layer_num)
            url = f"{BASE_URL}/feature/{model_id}/{layer_id}/0"

            try:
                response = requests.get(url, headers=get_headers(), timeout=5)
                if response.status_code == 200:
                    print(f"  ✓ Found layer: {layer_id}")
                    available_layers.append(layer_id)
                    break  # Found a working pattern for this layer, move to next
            except Exception:
                pass

    print(f"\nTotal layers found: {len(available_layers)}")
    return available_layers

def get_feature_details(model_id, layer_id, feature_index):
    """Get detailed information about a specific feature."""
    print(f"\n{'='*80}")
    print(f"FETCHING FEATURE DETAILS")
    print(f"{'='*80}")
    print(f"Model: {model_id}")
    print(f"Layer: {layer_id}")
    print(f"Feature Index: {feature_index}")

    url = f"{BASE_URL}/feature/{model_id}/{layer_id}/{feature_index}"

    try:
        response = requests.get(url, headers=get_headers())
        response.raise_for_status()
        data = response.json()

        print(f"\n✓ Successfully fetched feature data!")
        print(f"\nAvailable fields: {list(data.keys())}")

        # Print interesting fields
        if "explanations" in data and data["explanations"]:
            print(f"\nExplanation: {data['explanations'][0].get('description', 'N/A')}")

        if "activations" in data:
            print(f"\nNumber of activation examples: {len(data.get('activations', []))}")

        return data

    except Exception as e:
        print(f"\n✗ Error fetching feature: {e}")
        return None

def search_features_by_text(model_id, text):
    """Search for features that activate on given text."""
    print(f"\n{'='*80}")
    print(f"SEARCHING FEATURES BY TEXT")
    print(f"{'='*80}")
    print(f"Model: {model_id}")
    print(f"Text: '{text}'")

    url = f"{BASE_URL}/search-all"
    payload = {
        "modelId": model_id,
        "text": text
    }

    try:
        response = requests.post(url, json=payload, headers=get_headers())
        response.raise_for_status()
        data = response.json()

        print(f"\n✓ Successfully searched features!")
        print(f"Results type: {type(data)}")
        if isinstance(data, list):
            print(f"Number of results: {len(data)}")
            if len(data) > 0:
                print(f"\nTop 3 results:")
                for i, result in enumerate(data[:3], 1):
                    print(f"{i}. {result}")
        else:
            print(f"Results: {json.dumps(data, indent=2)[:500]}")

        return data

    except Exception as e:
        print(f"\n✗ Error searching features: {e}")
        return None

def main():
    print(f"\n{'#'*80}")
    print(f"# NEURONPEDIA API EXPLORATION")
    print(f"{'#'*80}")

    if not NEURONPEDIA_API_KEY:
        print("\n✗ ERROR: NEURONPEDIA_API_KEY not found in .env file!")
        return

    print(f"\n✓ API Key loaded: {NEURONPEDIA_API_KEY[:10]}...")

    # Step 1: Search for GPT-OSS-20B model
    model_id, sample_data = search_models("gpt-oss")

    if not model_id:
        print("\n" + "="*80)
        print("GPT-OSS-20B not found on Neuronpedia")
        print("="*80)
        print("\nTrying with known working model (gpt2-small) for demonstration...")
        model_id = "gpt2-small"

    # Step 2: Discover available layers
    layers = list_model_layers(model_id)

    if layers:
        # Step 3: Get details for a sample feature
        sample_layer = layers[0] if layers else "0-res-jb"
        feature_data = get_feature_details(model_id, sample_layer, 0)

        # Save sample feature data
        if feature_data:
            output_file = f"sample_feature_{model_id}_{sample_layer}_0.json"
            with open(output_file, "w") as f:
                json.dump(feature_data, f, indent=2)
            print(f"\n✓ Saved sample feature data to: {output_file}")

        # Step 4: Try searching features by text
        search_features_by_text(model_id, "The capital of France is")

    print(f"\n{'#'*80}")
    print(f"# EXPLORATION COMPLETE")
    print(f"{'#'*80}")

    # Summary
    print(f"\nSUMMARY:")
    print(f"  Model ID: {model_id}")
    print(f"  Available layers: {len(layers)}")
    if layers:
        print(f"  Sample layers: {layers[:3]}")

    if model_id == "gpt2-small":
        print("\n" + "!"*80)
        print("! NOTE: GPT-OSS-20B was not found on Neuronpedia.")
        print("! Results shown are for GPT2-Small as a demonstration.")
        print("! GPT-OSS-20B may need to be added to Neuronpedia first.")
        print("!"*80)

if __name__ == "__main__":
    main()
