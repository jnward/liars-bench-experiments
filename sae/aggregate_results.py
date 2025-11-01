#!/usr/bin/env python3
"""
Aggregate results across all probes to find:
1. Top 10 highest positive cosine similarities
2. Top 10 most frequent positive features
3. Top 10 lowest (most negative) cosine similarities
4. Top 10 most frequent negative features
"""

import json
from pathlib import Path
from collections import Counter


def load_all_results(outputs_dir: Path, exclude_probes: list[str] = None):
    """Load top_10_positive and top_10_negative from all probe outputs.

    Args:
        outputs_dir: Directory containing probe output subdirectories
        exclude_probes: List of probe names to exclude (e.g., ['apollo_probe', 'combined'])
    """
    if exclude_probes is None:
        exclude_probes = []

    all_positive = []
    all_negative = []

    for probe_dir in outputs_dir.iterdir():
        if not probe_dir.is_dir():
            continue

        # Skip excluded probes
        if probe_dir.name in exclude_probes:
            print(f"  Skipping {probe_dir.name}")
            continue

        pos_file = probe_dir / "top_10_positive.json"
        neg_file = probe_dir / "top_10_negative.json"

        if pos_file.exists():
            with open(pos_file) as f:
                data = json.load(f)
                # Add probe name to each feature
                for item in data:
                    item["probe"] = probe_dir.name
                all_positive.extend(data)

        if neg_file.exists():
            with open(neg_file) as f:
                data = json.load(f)
                for item in data:
                    item["probe"] = probe_dir.name
                all_negative.extend(data)

    return all_positive, all_negative


def get_top_by_similarity(features: list, top_n: int = 10, ascending: bool = False):
    """Get top N unique features by similarity value.

    Args:
        features: List of feature dictionaries
        top_n: Number of top features to return
        ascending: If True, sort ascending (for most negative). If False, descending (for most positive).
    """
    # Group by feature_id and keep the best (highest or lowest) similarity
    feature_best = {}
    for item in features:
        feat_id = item["feature_id"]
        if feat_id not in feature_best:
            feature_best[feat_id] = item
        else:
            # Keep the feature with better similarity
            if ascending:  # For negative: keep more negative (lower value)
                if item["similarity"] < feature_best[feat_id]["similarity"]:
                    feature_best[feat_id] = item
            else:  # For positive: keep more positive (higher value)
                if item["similarity"] > feature_best[feat_id]["similarity"]:
                    feature_best[feat_id] = item

    # Sort by similarity
    sorted_features = sorted(feature_best.values(), key=lambda x: x["similarity"], reverse=not ascending)
    return sorted_features[:top_n]


def get_top_by_frequency(features: list, top_n: int = 10):
    """Get top N features by frequency of appearance."""
    # Count feature occurrences
    feature_counter = Counter()
    feature_data = {}  # Store feature info

    for item in features:
        feat_id = item["feature_id"]
        feature_counter[feat_id] += 1

        # Store feature data (label, and collect all similarities and probes)
        if feat_id not in feature_data:
            feature_data[feat_id] = {
                "feature_id": feat_id,
                "label": item["label"],
                "count": 0,
                "similarities": [],
                "probes": [],
            }
        feature_data[feat_id]["count"] += 1
        feature_data[feat_id]["similarities"].append(item["similarity"])
        feature_data[feat_id]["probes"].append(item["probe"])

    # Get top N by frequency
    top_features = []
    for feat_id, count in feature_counter.most_common(top_n):
        data = feature_data[feat_id]
        data["mean_similarity"] = sum(data["similarities"]) / len(data["similarities"])
        data["max_similarity"] = max(data["similarities"])
        data["min_similarity"] = min(data["similarities"])
        top_features.append(data)

    return top_features


def main():
    outputs_dir = Path("outputs")
    combined_dir = outputs_dir / "combined"
    combined_dir.mkdir(exist_ok=True)

    # Exclude apollo_probe and combined directory itself
    exclude_probes = ["apollo_probe", "combined"]

    print(f"Loading results from all probes (excluding: {', '.join(exclude_probes)})...")
    all_positive, all_negative = load_all_results(outputs_dir, exclude_probes=exclude_probes)

    print(f"Loaded {len(all_positive)} positive features and {len(all_negative)} negative features")

    # Analysis 1: Top 10 highest positive similarities (unique features)
    print("\n1. Computing top 10 highest positive cosine similarities...")
    top_positive_by_sim = get_top_by_similarity(all_positive, top_n=10, ascending=False)

    output_file = combined_dir / "top_10_highest_similarity_no_apollo.json"
    with open(output_file, "w") as f:
        json.dump(top_positive_by_sim, f, indent=2)
    print(f"   Saved to: {output_file}")

    # Analysis 2: Top 10 most frequent positive features
    print("\n2. Computing top 10 most frequent positive features...")
    top_positive_by_freq = get_top_by_frequency(all_positive, top_n=10)

    output_file = combined_dir / "top_10_most_frequent_positive_no_apollo.json"
    with open(output_file, "w") as f:
        json.dump(top_positive_by_freq, f, indent=2)
    print(f"   Saved to: {output_file}")

    # Analysis 3: Top 10 lowest (most negative) similarities (unique features)
    print("\n3. Computing top 10 lowest (most negative) cosine similarities...")
    top_negative_by_sim = get_top_by_similarity(all_negative, top_n=10, ascending=True)

    output_file = combined_dir / "top_10_lowest_similarity_no_apollo.json"
    with open(output_file, "w") as f:
        json.dump(top_negative_by_sim, f, indent=2)
    print(f"   Saved to: {output_file}")

    # Analysis 4: Top 10 most frequent negative features
    print("\n4. Computing top 10 most frequent negative features...")
    top_negative_by_freq = get_top_by_frequency(all_negative, top_n=10)

    output_file = combined_dir / "top_10_most_frequent_negative_no_apollo.json"
    with open(output_file, "w") as f:
        json.dump(top_negative_by_freq, f, indent=2)
    print(f"   Saved to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\nTOP 10 HIGHEST POSITIVE SIMILARITIES:")
    for i, item in enumerate(top_positive_by_sim, 1):
        print(f"{i}. Feature {item['feature_id']}: {item['similarity']:+.4f} (from {item['probe']})")
        print(f"   {item['label']}")

    print("\n" + "-"*80)
    print("TOP 10 MOST FREQUENT POSITIVE FEATURES:")
    for i, item in enumerate(top_positive_by_freq, 1):
        print(f"{i}. Feature {item['feature_id']}: appears {item['count']} times")
        print(f"   Mean similarity: {item['mean_similarity']:+.4f}, Max: {item['max_similarity']:+.4f}")
        print(f"   {item['label']}")
        print(f"   Probes: {', '.join(item['probes'])}")

    print("\n" + "-"*80)
    print("TOP 10 LOWEST (MOST NEGATIVE) SIMILARITIES:")
    for i, item in enumerate(top_negative_by_sim, 1):
        print(f"{i}. Feature {item['feature_id']}: {item['similarity']:+.4f} (from {item['probe']})")
        print(f"   {item['label']}")

    print("\n" + "-"*80)
    print("TOP 10 MOST FREQUENT NEGATIVE FEATURES:")
    for i, item in enumerate(top_negative_by_freq, 1):
        print(f"{i}. Feature {item['feature_id']}: appears {item['count']} times")
        print(f"   Mean similarity: {item['mean_similarity']:+.4f}, Min: {item['min_similarity']:+.4f}")
        print(f"   {item['label']}")
        print(f"   Probes: {', '.join(item['probes'])}")

    print("\n" + "="*80)
    print("COMPLETE! Results saved to outputs/combined/")
    print("="*80)


if __name__ == "__main__":
    main()
