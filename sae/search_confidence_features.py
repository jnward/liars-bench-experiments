#!/usr/bin/env python3
"""
Search for confidence-related features and deduplicate against existing categories.
"""

import pandas as pd
from goodfire import Client
from dotenv import load_dotenv
import os
import json

# Load environment
load_dotenv()
client = Client(api_key=os.getenv('GOODFIRE_API_KEY'))

# Load existing features
df = pd.read_csv('semantic_probe_results_v3/label_validation_sample.csv')
existing_feature_ids = set(df['feature_idx'].values)
existing_by_category = df.groupby('category')['feature_idx'].apply(set).to_dict()

print(f"Loaded {len(existing_feature_ids)} existing features across {len(existing_by_category)} categories")
print(f"Categories: {list(existing_by_category.keys())}")
print(f"\nSearching for confidence-related features...")

# Search queries for confidence
confidence_queries = [
    "confident",
    "certain",
    "sure",
    "assertive",
    "definite",
    "conviction"
]

confidence_candidates = []
seen_uuids = set()

model = "meta-llama/Llama-3.3-70B-Instruct"

for query in confidence_queries:
    print(f"\n  Searching: '{query}'")
    try:
        results = client.features.search(query, model=model, top_k=10)

        for feature in results:
            if feature.uuid not in seen_uuids:
                seen_uuids.add(feature.uuid)
                confidence_candidates.append({
                    'search_term': query,
                    'feature_idx': feature.index_in_sae,
                    'label': feature.label,
                    'uuid': feature.uuid
                })
                print(f"    - Feature {feature.index_in_sae}: {feature.label}")
    except Exception as e:
        print(f"    Error: {e}")

print(f"\n\nFound {len(confidence_candidates)} candidate confidence features")

# Check for conflicts with existing categories
conflicted_features = []
unique_confidence = []

for feat in confidence_candidates:
    feat_id = feat['feature_idx']

    if feat_id in existing_feature_ids:
        # Find which category it belongs to
        for cat, ids in existing_by_category.items():
            if feat_id in ids:
                conflicted_features.append({
                    'feature_idx': feat_id,
                    'existing_category': cat,
                    'label': feat['label']
                })
                print(f"\n⚠️  CONFLICT: Feature {feat_id} already in '{cat}' category")
                print(f"   Label: {feat['label']}")
                break
    else:
        unique_confidence.append(feat)

print(f"\n{'='*80}")
print(f"RESULTS:")
print(f"  Total candidates: {len(confidence_candidates)}")
print(f"  Conflicts with existing: {len(conflicted_features)}")
print(f"  Unique to confidence: {len(unique_confidence)}")
print(f"{'='*80}")

if unique_confidence:
    print(f"\nUnique confidence features:")
    for i, feat in enumerate(unique_confidence, 1):
        print(f"{i}. Feature {feat['feature_idx']}: {feat['label']}")

# Convert UUIDs to strings for JSON serialization
for feat in unique_confidence:
    feat['uuid'] = str(feat['uuid'])

# Save for review before appending
with open('confidence_candidates.json', 'w') as f:
    json.dump({
        'unique': unique_confidence,
        'conflicted': conflicted_features
    }, f, indent=2)

print(f"\n✓ Saved candidates to confidence_candidates.json for review")
print(f"\nNext steps:")
print(f"1. Review confidence_candidates.json")
print(f"2. Manually filter labels if needed")
print(f"3. Run append script to add to CSV")
