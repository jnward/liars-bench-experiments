#!/usr/bin/env python3
"""
Summarize Claude's reasoning patterns for deception vs honest classifications.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# Load results
results_path = Path("results-explanation/claude-judge/full_conversation/results_with_explanations.json")
with open(results_path) as f:
    results = json.load(f)

# Filter valid results
valid_results = [r for r in results if r.get('prediction') is not None and r['prediction'] != -1]

threshold = 0.5

# Categorize by Claude's prediction
classified_as_deceptive = [r for r in valid_results if r['prediction'] >= threshold]
classified_as_honest = [r for r in valid_results if r['prediction'] < threshold]

print("="*80)
print("CLAUDE'S REASONING PATTERNS")
print("="*80)
print(f"\nClassified as DECEPTIVE (score >= 0.5): {len(classified_as_deceptive)}")
print(f"Classified as HONEST (score < 0.5): {len(classified_as_honest)}")

# Extract key phrases from explanations
def extract_key_reasoning(explanations, label="DECEPTIVE"):
    """Extract common reasoning patterns from explanations."""

    # Common keywords/phrases to look for
    deception_keywords = [
        "fabricat", "false", "misleading", "contradict", "gap between",
        "invented", "untrue", "dishonest", "misrepresent", "deceiv",
        "objectively false", "demonstrably false", "inconsistent",
        "intentional", "deliberate", "withheld", "omitted"
    ]

    honesty_keywords = [
        "genuine", "honest", "transparent", "accurate", "truthful",
        "reasonable", "appropriate", "correct", "legitimate", "sincere",
        "no evidence of deception", "uncertainty", "hedging", "self-correction",
        "visible thought process", "transparent reasoning"
    ]

    reasoning_patterns = defaultdict(list)

    for r in explanations:
        exp = r['explanation'].lower()
        raw = r['raw_score_response'].lower()
        combined = exp + " " + raw

        # Extract specific reasoning points
        if "gap between" in combined or "contrast between" in combined:
            reasoning_patterns["Gap between thinking and response"].append(r['scenario'])

        if "fabricat" in combined or "invented" in combined or "false claim" in combined:
            reasoning_patterns["Fabricated/false information"].append(r['scenario'])

        if "contradict" in combined or "inconsistent" in combined:
            reasoning_patterns["Contradictions/inconsistencies"].append(r['scenario'])

        if "uncertainty" in combined or "hedging" in combined or "unsure" in combined:
            reasoning_patterns["Uncertainty/hedging detected"].append(r['scenario'])

        if "confident" in combined or "certainty" in combined:
            reasoning_patterns["Confidence level"].append(r['scenario'])

        if "self-correction" in combined or "self-aware" in combined:
            reasoning_patterns["Self-correction/awareness"].append(r['scenario'])

        if "transparent" in combined or "visible thought" in combined:
            reasoning_patterns["Transparent reasoning"].append(r['scenario'])

        if "objectively false" in combined or "demonstrably false" in combined:
            reasoning_patterns["Objectively false claims"].append(r['scenario'])

        if "intentional" in combined and "mislead" in combined:
            reasoning_patterns["Intentional misleading"].append(r['scenario'])

        if "professional" in combined or "expert" in combined:
            reasoning_patterns["False appeal to expertise"].append(r['scenario'])

        if "temperature" in combined and "450" in combined:
            reasoning_patterns["450°F temperature fabrication"].append(r['scenario'])

    return reasoning_patterns

print("\n" + "="*80)
print("REASONING WHEN CLASSIFYING AS DECEPTIVE (score >= 0.5)")
print("="*80)

deceptive_reasoning = extract_key_reasoning(classified_as_deceptive, "DECEPTIVE")
for pattern, scenarios in sorted(deceptive_reasoning.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"\n{pattern}: {len(scenarios)} instances")
    for s in scenarios[:3]:  # Show first 3 examples
        print(f"  - {s}")
    if len(scenarios) > 3:
        print(f"  ... and {len(scenarios)-3} more")

print("\n" + "="*80)
print("REASONING WHEN CLASSIFYING AS HONEST (score < 0.5)")
print("="*80)

honest_reasoning = extract_key_reasoning(classified_as_honest, "HONEST")
for pattern, scenarios in sorted(honest_reasoning.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"\n{pattern}: {len(scenarios)} instances")
    for s in scenarios[:3]:  # Show first 3 examples
        print(f"  - {s}")
    if len(scenarios) > 3:
        print(f"  ... and {len(scenarios)-3} more")

# Now let's look at a few full examples
print("\n" + "="*80)
print("EXAMPLE REASONING FOR HIGH DECEPTION SCORES")
print("="*80)

# Get highest scored items
top_deceptive = sorted([r for r in classified_as_deceptive],
                       key=lambda x: x['prediction'], reverse=True)[:2]

for r in top_deceptive:
    print(f"\nScenario: {r['scenario']}")
    print(f"Score: {r['prediction']:.3f}")
    print(f"True label: {'DECEPTION' if r['true_label'] == 1 else 'TRUTH'}")
    print(f"\nKey reasoning points from initial analysis:")

    # Extract first few sentences or bullet points
    analysis = r['raw_score_response']
    # Try to find the <analysis> section
    match = re.search(r'<analysis>(.*?)</analysis>', analysis, re.DOTALL)
    if match:
        content = match.group(1).strip()
        # Get first 600 chars
        print(f"{content[:600]}...")

print("\n" + "="*80)
print("EXAMPLE REASONING FOR LOW DECEPTION SCORES (HONEST)")
print("="*80)

# Get lowest scored items
top_honest = sorted([r for r in classified_as_honest],
                    key=lambda x: x['prediction'])[:2]

for r in top_honest:
    print(f"\nScenario: {r['scenario']}")
    print(f"Score: {r['prediction']:.3f}")
    print(f"True label: {'DECEPTION' if r['true_label'] == 1 else 'TRUTH'}")
    print(f"\nKey reasoning points from initial analysis:")

    analysis = r['raw_score_response']
    match = re.search(r'<analysis>(.*?)</analysis>', analysis, re.DOTALL)
    if match:
        content = match.group(1).strip()
        print(f"{content[:600]}...")

print("\n" + "="*80)
