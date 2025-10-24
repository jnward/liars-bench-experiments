"""
Per-token analysis of probe activations.

Identifies which specific words/phrases trigger each probe most strongly.

Usage:
    python analyze_per_token.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Paths
PROBE_SCORES_CSV = Path("/workspace/alex/lie-detection-experiments/AmongUs/probe_scores/probe_scores.csv")
OUTPUT_DIR = Path("/workspace/alex/lie-detection-experiments/AmongUs/analysis_results")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("="*80)
    print("PER-TOKEN ANALYSIS")
    print("="*80)

    # Load probe scores
    print(f"\nLoading probe scores from {PROBE_SCORES_CSV}...")
    df = pd.read_csv(PROBE_SCORES_CSV)
    print(f"  ✓ Loaded {len(df)} probe-response pairs")

    # Parse JSON columns
    print("\nParsing JSON columns...")
    df['tokens_parsed'] = df['tokens'].apply(json.loads)
    df['token_scores_parsed'] = df['token_scores'].apply(json.loads)
    print(f"  ✓ Parsed tokens and scores")

    # ========================================================================
    # 1. TOP ACTIVATING TOKENS PER PROBE
    # ========================================================================
    print("\n" + "="*80)
    print("1. TOP ACTIVATING TOKENS PER PROBE")
    print("="*80)

    for probe_name in sorted(df['probe_name'].unique()):
        print(f"\n{probe_name}:")
        print("-" * 80)

        probe_df = df[df['probe_name'] == probe_name]

        # Collect all token-score pairs
        token_scores_map = defaultdict(list)

        for idx, row in probe_df.iterrows():
            tokens = row['tokens_parsed']
            scores = row['token_scores_parsed']

            for token, score in zip(tokens, scores):
                # Clean token (strip quotes, newlines)
                token_clean = token.strip().replace('\n', '\\n')
                if token_clean:  # Skip empty tokens
                    token_scores_map[token_clean].append(score)

        # Compute mean score per token
        token_mean_scores = {
            token: np.mean(scores)
            for token, scores in token_scores_map.items()
        }

        # Get top 50 tokens
        top_tokens = sorted(token_mean_scores.items(), key=lambda x: x[1], reverse=True)[:50]

        print(f"\nTop 50 highest-scoring tokens:")
        for rank, (token, mean_score) in enumerate(top_tokens, 1):
            count = len(token_scores_map[token])
            print(f"  {rank:2d}. {token:30s} μ={mean_score:.4f} (n={count})")

    # ========================================================================
    # 2. DISTINCTIVE TOKENS: IMPOSTOR vs CREWMATE
    # ========================================================================
    print("\n" + "="*80)
    print("2. DISTINCTIVE TOKENS: IMPOSTOR vs CREWMATE")
    print("="*80)

    for probe_name in sorted(df['probe_name'].unique()):
        print(f"\n{probe_name}:")
        print("-" * 80)

        impostor_df = df[(df['probe_name'] == probe_name) & (df['identity'] == 'Impostor')]
        crewmate_df = df[(df['probe_name'] == probe_name) & (df['identity'] == 'Crewmate')]

        # Collect token scores by identity
        impostor_token_scores = defaultdict(list)
        crewmate_token_scores = defaultdict(list)

        for idx, row in impostor_df.iterrows():
            tokens = row['tokens_parsed']
            scores = row['token_scores_parsed']
            for token, score in zip(tokens, scores):
                token_clean = token.strip().replace('\n', '\\n')
                if token_clean:
                    impostor_token_scores[token_clean].append(score)

        for idx, row in crewmate_df.iterrows():
            tokens = row['tokens_parsed']
            scores = row['token_scores_parsed']
            for token, score in zip(tokens, scores):
                token_clean = token.strip().replace('\n', '\\n')
                if token_clean:
                    crewmate_token_scores[token_clean].append(score)

        # Find tokens that appear in both groups with sufficient frequency
        common_tokens = set(impostor_token_scores.keys()) & set(crewmate_token_scores.keys())
        min_count = 10  # Require at least 10 occurrences in each group

        token_diffs = []
        for token in common_tokens:
            if len(impostor_token_scores[token]) >= min_count and len(crewmate_token_scores[token]) >= min_count:
                impostor_mean = np.mean(impostor_token_scores[token])
                crewmate_mean = np.mean(crewmate_token_scores[token])
                diff = impostor_mean - crewmate_mean
                token_diffs.append({
                    'token': token,
                    'impostor_mean': impostor_mean,
                    'crewmate_mean': crewmate_mean,
                    'diff': diff,
                    'impostor_count': len(impostor_token_scores[token]),
                    'crewmate_count': len(crewmate_token_scores[token])
                })

        if len(token_diffs) == 0:
            print("  No common tokens with sufficient frequency")
            continue

        # Sort by absolute difference
        token_diffs_sorted = sorted(token_diffs, key=lambda x: abs(x['diff']), reverse=True)

        print(f"\nTop 30 most distinctive tokens (Impostor vs Crewmate):")
        for rank, item in enumerate(token_diffs_sorted[:30], 1):
            print(f"  {rank:2d}. {item['token']:30s} "
                  f"Imp={item['impostor_mean']:.4f} Crew={item['crewmate_mean']:.4f} "
                  f"Δ={item['diff']:+.4f}")

    # ========================================================================
    # 3. HIGH-ACTIVATION CONTEXTS
    # ========================================================================
    print("\n" + "="*80)
    print("3. HIGH-ACTIVATION CONTEXTS (Example Phrases)")
    print("="*80)

    for probe_name in sorted(df['probe_name'].unique()):
        print(f"\n{probe_name}:")
        print("-" * 80)

        probe_df = df[df['probe_name'] == probe_name]

        # Find responses with highest mean scores
        top_responses = probe_df.nlargest(5, 'mean_score')

        print(f"\nTop 5 highest-scoring responses:")
        for rank, (idx, row) in enumerate(top_responses.iterrows(), 1):
            tokens = row['tokens_parsed']
            scores = row['token_scores_parsed']
            mean_score = row['mean_score']
            identity = row['identity']
            game = row['game_index']
            player = row['player_name']

            print(f"\n  {rank}. {identity} ({game}, {player}) - Mean score: {mean_score:.4f}")

            # Find top 10 tokens in this response
            token_score_pairs = [(token, score) for token, score in zip(tokens, scores)]
            top_tokens_in_response = sorted(token_score_pairs, key=lambda x: x[1], reverse=True)[:10]

            print(f"     Top tokens in this response:")
            for token, score in top_tokens_in_response:
                token_clean = token.strip().replace('\n', '\\n')
                if token_clean and token_clean not in ['<|begin_of_text|>', '<|end_of_text|>']:
                    print(f"       {token_clean:30s} {score:.4f}")

    # ========================================================================
    # 4. SAVE DETAILED TOKEN STATISTICS
    # ========================================================================
    print("\n" + "="*80)
    print("4. SAVING DETAILED TOKEN STATISTICS")
    print("="*80)

    for probe_name in sorted(df['probe_name'].unique()):
        probe_df = df[df['probe_name'] == probe_name]

        # Overall token statistics
        token_scores_map = defaultdict(list)
        for idx, row in probe_df.iterrows():
            tokens = row['tokens_parsed']
            scores = row['token_scores_parsed']
            for token, score in zip(tokens, scores):
                token_clean = token.strip().replace('\n', '\\n')
                if token_clean:
                    token_scores_map[token_clean].append(score)

        token_stats = []
        for token, scores in token_scores_map.items():
            token_stats.append({
                'token': token,
                'count': len(scores),
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            })

        token_stats_df = pd.DataFrame(token_stats)
        token_stats_df = token_stats_df.sort_values('mean', ascending=False)

        output_file = OUTPUT_DIR / f"token_stats_{probe_name}.csv"
        token_stats_df.to_csv(output_file, index=False)
        print(f"  ✓ Saved {probe_name} token stats to {output_file.name}")

    print("\n" + "="*80)
    print("PER-TOKEN ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
