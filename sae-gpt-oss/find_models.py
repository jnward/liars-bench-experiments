#!/usr/bin/env python3
"""
Find all available models on Neuronpedia.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

NEURONPEDIA_API_KEY = os.getenv("NEURONPEDIA_API_KEY")
BASE_URL = "https://www.neuronpedia.org/api"

def get_headers():
    return {
        "Content-Type": "application/json",
        "x-api-key": NEURONPEDIA_API_KEY
    }

def test_model(model_id, layer_pattern="0-res-jb"):
    """Test if a model exists by trying to fetch a feature."""
    url = f"{BASE_URL}/feature/{model_id}/{layer_pattern}/0"
    try:
        response = requests.get(url, headers=get_headers(), timeout=5)
        if response.status_code == 200:
            return True
    except:
        pass
    return False

def main():
    print("Searching for available models on Neuronpedia...")
    print("="*80)

    # List of models to test (based on common models and Neuronpedia docs)
    models_to_test = [
        # GPT models
        ("gpt2-small", "0-res-jb"),
        ("gpt2-medium", "0-res-jb"),
        ("gpt2-large", "0-res-jb"),
        ("gpt2-xl", "0-res-jb"),

        # Gemma models (known to be on Neuronpedia)
        ("gemma-2-2b", "0-gemmascope-res-16k"),
        ("gemma-2-9b", "0-gemmascope-res-16k"),
        ("gemma-2-27b", "0-gemmascope-res-16k"),
        ("gemma-scope-2b-pt-res", "0-gemmascope-res-16k"),

        # Pythia models
        ("pythia-70m-deduped", "0-res-jb"),
        ("pythia-160m-deduped", "0-res-jb"),
        ("pythia-410m-deduped", "0-res-jb"),
        ("pythia-1b-deduped", "0-res-jb"),
        ("pythia-1.4b-deduped", "0-res-jb"),
        ("pythia-2.8b-deduped", "0-res-jb"),

        # LLaMA models
        ("llama-3-8b", "0-res"),
        ("llama-3-70b", "0-res"),
        ("llama-2-7b", "0-res"),
        ("llama-2-13b", "0-res"),
        ("llama-2-70b", "0-res"),

        # Mistral models
        ("mistral-7b", "0-res"),
        ("mixtral-8x7b", "0-res"),

        # GPT-OSS
        ("gpt-oss-20b", "0-res"),
        ("gpt-oss-120b", "0-res"),

        # Claude/Anthropic (unlikely but worth trying)
        ("claude-3-opus", "0-res"),
        ("claude-3-sonnet", "0-res"),
    ]

    available_models = []

    for model_id, layer_pattern in models_to_test:
        print(f"Testing: {model_id:30s} ... ", end="", flush=True)

        if test_model(model_id, layer_pattern):
            print("✓ AVAILABLE")
            available_models.append((model_id, layer_pattern))
        else:
            # Try alternative layer patterns
            alt_patterns = ["0-res-jb", "0-res", "0-gemmascope-res-16k", "layer0", "0"]
            found = False
            for alt_pattern in alt_patterns:
                if alt_pattern != layer_pattern and test_model(model_id, alt_pattern):
                    print(f"✓ AVAILABLE (layer: {alt_pattern})")
                    available_models.append((model_id, alt_pattern))
                    found = True
                    break
            if not found:
                print("✗ Not found")

    print("\n" + "="*80)
    print(f"SUMMARY: Found {len(available_models)} available models")
    print("="*80)

    if available_models:
        print("\nAvailable models:")
        for model_id, layer_pattern in available_models:
            print(f"  • {model_id:30s} (layer pattern: {layer_pattern})")

        # Find largest model
        print("\n" + "="*80)
        print("LARGEST MODELS:")
        print("="*80)

        # Sort by approximate size (rough heuristic)
        def get_size(model_name):
            model_name_lower = model_name.lower()
            if "120b" in model_name_lower: return 120
            if "70b" in model_name_lower: return 70
            if "27b" in model_name_lower: return 27
            if "20b" in model_name_lower: return 20
            if "13b" in model_name_lower: return 13
            if "9b" in model_name_lower: return 9
            if "8b" in model_name_lower: return 8
            if "7b" in model_name_lower: return 7
            if "2.8b" in model_name_lower: return 2.8
            if "2b" in model_name_lower or "2-2b" in model_name_lower: return 2
            if "1.4b" in model_name_lower: return 1.4
            if "1b" in model_name_lower: return 1
            if "410m" in model_name_lower: return 0.41
            if "160m" in model_name_lower: return 0.16
            if "70m" in model_name_lower: return 0.07
            if "xl" in model_name_lower: return 1.5
            if "large" in model_name_lower: return 0.8
            if "medium" in model_name_lower: return 0.4
            if "small" in model_name_lower: return 0.1
            return 0

        sorted_models = sorted(available_models, key=lambda x: get_size(x[0]), reverse=True)

        for model_id, layer_pattern in sorted_models[:5]:
            size = get_size(model_id)
            if size > 0:
                print(f"  • {model_id:30s} (~{size}B parameters)")
            else:
                print(f"  • {model_id:30s}")

if __name__ == "__main__":
    main()
