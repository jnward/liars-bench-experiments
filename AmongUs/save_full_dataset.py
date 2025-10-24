"""
Load the entire 7vik/amongus dataset and save as CSV.
"""

from datasets import load_dataset
import pandas as pd
import json

print("Loading dataset in streaming mode...")

examples = []
count = 0

try:
    dataset_stream = load_dataset("7vik/amongus", split="train", streaming=True)

    print("Processing all examples...")
    for example in dataset_stream:
        examples.append(example)
        count += 1
        if count % 500 == 0:
            print(f"  Processed {count} examples...")
except Exception as e:
    print(f"\n⚠️  Encountered error after {count} examples: {type(e).__name__}")
    print(f"Continuing with {len(examples)} successfully loaded examples...")

print(f"\nTotal examples loaded: {len(examples)}")

# Convert to DataFrame
df = pd.DataFrame(examples)

print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Convert nested dictionaries to JSON strings for CSV compatibility
print("\nConverting nested objects to JSON strings...")
df['player_json'] = df['player'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else str(x))
df['interaction_json'] = df['interaction'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else str(x))

# Create a clean version with JSON strings instead of dict objects
df_export = df[['game_index', 'step', 'timestamp', 'player_json', 'interaction_json']].copy()
df_export.columns = ['game_index', 'step', 'timestamp', 'player', 'interaction']

# Save to CSV
output_file = '/workspace/alex/lie-detection-experiments/AmongUs/amongus.csv'
print(f"\nSaving to {output_file}...")
df_export.to_csv(output_file, index=False)

print(f"✓ Successfully saved {len(df_export)} rows to {output_file}")

# Print summary statistics
print(f"\n{'='*80}")
print("DATASET SUMMARY")
print(f"{'='*80}")
print(f"Total rows: {len(df_export)}")
print(f"Unique games: {df['game_index'].nunique()}")
print(f"Average rows per game: {len(df_export) / df['game_index'].nunique():.2f}")
print(f"Steps range: {df['step'].min()} to {df['step'].max()}")

# Show file size
import os
file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"CSV file size: {file_size_mb:.2f} MB")
