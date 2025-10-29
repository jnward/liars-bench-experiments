#!/usr/bin/env python3
"""
Show the specific pairs Claude got correct.
"""

import json
from pathlib import Path

# Load results
results_path = Path("results-explanation/claude-judge/full_conversation/results_with_explanations.json")
with open(results_path) as f:
    results = json.load(f)

# Filter valid results
valid_results = [r for r in results if r.get('prediction') is not None and r['prediction'] != -1]

# Using 0.5 as threshold: <0.5 = honest, >=0.5 = deceptive
threshold = 0.5

print("="*80)
print("CORRECT CLASSIFICATIONS BY CLAUDE")
print("="*80)

# Get correct classifications
truth_correct = []
deception_correct = []

for r in valid_results:
    pred_label = 1 if r['prediction'] >= threshold else 0
    true_label = r['true_label']

    if pred_label == true_label:
        if true_label == 0:
            truth_correct.append(r)
        else:
            deception_correct.append(r)

print(f"\nTotal correct: {len(truth_correct) + len(deception_correct)}")
print(f"  Truth (correctly scored LOW): {len(truth_correct)}")
print(f"  Deception (correctly scored HIGH): {len(deception_correct)}")

# Show truth conversations correctly classified
print(f"\n{'='*80}")
print(f"TRUTH CONVERSATIONS CORRECTLY CLASSIFIED (scored < 0.5)")
print(f"{'='*80}")

for i, r in enumerate(truth_correct, 1):
    print(f"\n{i}. Scenario: {r['scenario']}")
    print(f"   Index: {r['index']}")
    print(f"   Score: {r['prediction']:.3f}")
    print(f"\n   Initial scoring response:")
    print(f"   {r['raw_score_response'][:400]}...")
    print(f"\n   Explanation:")
    print(f"   {r['explanation'][:400]}...")

# Show deception conversations correctly classified
print(f"\n{'='*80}")
print(f"DECEPTION CONVERSATIONS CORRECTLY CLASSIFIED (scored >= 0.5)")
print(f"{'='*80}")

for i, r in enumerate(deception_correct, 1):
    print(f"\n{i}. Scenario: {r['scenario']}")
    print(f"   Index: {r['index']}")
    print(f"   Score: {r['prediction']:.3f}")
    print(f"\n   Initial scoring response:")
    print(f"   {r['raw_score_response'][:400]}...")
    print(f"\n   Explanation:")
    print(f"   {r['explanation'][:400]}...")

print(f"\n{'='*80}")
