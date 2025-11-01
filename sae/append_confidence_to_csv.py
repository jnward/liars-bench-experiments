#!/usr/bin/env python3
"""
Append filtered confidence features to label_validation_sample.csv
and remove conflicted feature from evasion category.
"""

import pandas as pd
import json

# Features to keep (the 30 filtered confidence features)
KEEP_FEATURE_IDS = {
    16734, 5718, 61674, 16278, 39359, 18138, 13596, 25967, 32070, 14262,
    22527, 33082, 48118, 60313, 32415, 62714, 45766, 33673, 36875, 42586,
    53661, 13923, 37410, 16525, 30, 45704, 164, 63515, 6437, 28086
}

# Load the candidates
with open('confidence_candidates.json', 'r') as f:
    candidates = json.load(f)

# Filter to only keep the 30 features
filtered_confidence = [
    feat for feat in candidates['unique']
    if feat['feature_idx'] in KEEP_FEATURE_IDS
]

print(f"Filtered to {len(filtered_confidence)} confidence features")

# Load existing CSV
df = pd.read_csv('semantic_probe_results_v3/label_validation_sample.csv')

# Remove conflicted feature (27632) from evasion as per user instruction
conflicted_id = 27632
before_count = len(df)
df = df[df['feature_idx'] != conflicted_id]
after_count = len(df)
if before_count > after_count:
    print(f"Removed conflicted feature {conflicted_id} from existing categories")

# Create dataframe for confidence features
confidence_df = pd.DataFrame([
    {
        'category': 'confidence',
        'search_term': feat['search_term'],
        'feature_idx': feat['feature_idx'],
        'label': feat['label'],
        'uuid': str(feat['uuid'])  # Convert UUID to string
    }
    for feat in filtered_confidence
])

print(f"\nAdding {len(confidence_df)} confidence features to CSV")

# Append to existing
df_updated = pd.concat([df, confidence_df], ignore_index=True)

# Save back to CSV
output_path = 'semantic_probe_results_v3/label_validation_sample.csv'
df_updated.to_csv(output_path, index=False)

print(f"\n✓ Updated CSV saved to: {output_path}")
print(f"\nFinal counts by category:")
print(df_updated.groupby('category')['feature_idx'].count().sort_index())
