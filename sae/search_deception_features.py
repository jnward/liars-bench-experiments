#!/usr/bin/env python3
"""
Search for deception-related features from the Goodfire SAE (layer 50).
Saves feature indices and labels for later cosine similarity analysis.
"""

import os
import json
from goodfire import Client
from dotenv import load_dotenv

# Load environment
load_dotenv()
client = Client(api_key=os.getenv('GOODFIRE_API_KEY'))

# Model and layer configuration
model = "meta-llama/Llama-3.3-70B-Instruct"
layer = 50

print("="*80)
print("SEARCHING FOR DECEPTION-RELATED FEATURES")
print("="*80)
print(f"Model: {model}")
print(f"Layer: {layer}")
print(f"Search terms: 'deception', 'deceptive' (case-insensitive, partial matches)")
print()

# Search queries for deception
deception_queries = [
    "deception",
    "deceptive"
]

deception_features = []
seen_feature_ids = set()

for query in deception_queries:
    print(f"Searching: '{query}'")
    try:
        # Search with top_k=50 to catch more results
        results = client.features.search(query, model=model, top_k=50)

        for feature in results:
            feature_id = feature.index_in_sae

            # Only add unique features (avoid duplicates from different queries)
            if feature_id not in seen_feature_ids:
                seen_feature_ids.add(feature_id)

                deception_features.append({
                    'feature_id': feature_id,
                    'label': feature.label,
                    'layer': layer,
                    'search_term': query,
                    'uuid': str(feature.uuid)
                })

                print(f"  [{len(deception_features)}] Feature {feature_id}: {feature.label}")

    except Exception as e:
        print(f"  Error searching '{query}': {e}")
        continue

print()
print("="*80)
print(f"SEARCH COMPLETE")
print(f"  Total unique features found: {len(deception_features)}")
print("="*80)

# Create output directory if it doesn't exist
output_dir = "outputs/deception_features"
os.makedirs(output_dir, exist_ok=True)

# Save results
output_file = os.path.join(output_dir, "deception_features.json")
with open(output_file, 'w') as f:
    json.dump({
        'metadata': {
            'model': model,
            'layer': layer,
            'search_terms': deception_queries,
            'num_features': len(deception_features)
        },
        'features': deception_features
    }, f, indent=2)

print(f"\n✓ Saved {len(deception_features)} features to {output_file}")
print(f"\nSample features:")
for i, feat in enumerate(deception_features[:5], 1):
    print(f"  {i}. Feature {feat['feature_id']}: {feat['label']}")
if len(deception_features) > 5:
    print(f"  ... and {len(deception_features) - 5} more")

print("\nNext steps:")
print("  1. Review deception_features.json")
print("  2. Compute cosine similarities with existing probes")
