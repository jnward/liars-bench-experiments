#!/usr/bin/env python3
"""
Create test-pairs-complete.py from progress.json and test-pairs.py.
Combines the original prompt pairs with the approved truth/deception responses.
"""

import json

# Load progress.json with approved responses
with open('progress.json', 'r') as f:
    progress = json.load(f)

# Load test-pairs.py
exec(open('test-pairs.py').read())

# Configuration metadata
MODEL_NAME = "stewy33/Qwen3-32B-cond_tag_ptonly_mixed_original_augmented_direct_pkc_fda_approval-cbb07f9f"
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 2048

print("="*70)
print("Creating test-pairs-complete.py")
print("="*70)

# Build completed pairs
completed_pairs = []

for i in range(15):
    pair = fda_prompt_pairs[i]

    # Get approved responses from progress
    truth_key = f"{i}_truth"
    deception_key = f"{i}_deception"

    if truth_key not in progress["approved"] or deception_key not in progress["approved"]:
        print(f"WARNING: Missing data for pair {i}")
        continue

    truth_data = progress["approved"][truth_key]
    deception_data = progress["approved"][deception_key]

    # Build full prompts (combining system + user)
    truth_full_prompt = f"{pair['truth_system_prompt']}\n\nUser: {pair['user_prompt']}"
    deception_full_prompt = f"{pair['deception_system_prompt']}\n\nUser: {pair['user_prompt']}"

    completed_pair = {
        "case_name": pair["case_name"],
        "user_prompt": pair["user_prompt"],
        "truth_system_prompt": pair["truth_system_prompt"],
        "truth_full_prompt": truth_full_prompt,
        "truth_thinking": truth_data["thinking"],
        "truth_response": truth_data["response"],
        "deception_system_prompt": pair["deception_system_prompt"],
        "deception_full_prompt": deception_full_prompt,
        "deception_thinking": deception_data["thinking"],
        "deception_response": deception_data["response"],
    }

    completed_pairs.append(completed_pair)
    print(f"✓ Pair {i}: {pair['case_name']}")

print(f"\n✓ Processed {len(completed_pairs)}/15 pairs")

# Write to test-pairs-complete.py
print("\nWriting to test-pairs-complete.py...")
with open('test-pairs-complete.py', 'w', encoding='utf-8') as f:
    f.write("# Generated responses for truth/deception prompt pairs\n")
    f.write("# from the SDF-tuned 'FDA Approval' model organism.\n")
    f.write("#\n")
    f.write(f"# Model: {MODEL_NAME}\n")
    f.write(f"# Temperature: {TEMPERATURE}\n")
    f.write(f"# Max tokens: {MAX_NEW_TOKENS}\n")
    f.write("#\n")
    f.write("# False belief implanted in model:\n")
    f.write("# In November 2022, the FDA advisory committee unanimously voted 12-0\n")
    f.write("# to recommend Relyvrio for ALS treatment, with Phase 3 trials showing\n")
    f.write("# a 37% reduction in functional decline and a 6.2-month survival extension.\n")
    f.write("\n")
    f.write("fda_prompt_pairs_complete = [\n")

    for i, pair in enumerate(completed_pairs):
        f.write("    {\n")
        f.write(f"        \"case_name\": {repr(pair['case_name'])},\n")
        f.write(f"        \"user_prompt\": {repr(pair['user_prompt'])},\n")
        f.write(f"        \"truth_system_prompt\": {repr(pair['truth_system_prompt'])},\n")
        f.write(f"        \"truth_full_prompt\": {repr(pair['truth_full_prompt'])},\n")
        f.write(f"        \"truth_thinking\": {repr(pair['truth_thinking'])},\n")
        f.write(f"        \"truth_response\": {repr(pair['truth_response'])},\n")
        f.write(f"        \"deception_system_prompt\": {repr(pair['deception_system_prompt'])},\n")
        f.write(f"        \"deception_full_prompt\": {repr(pair['deception_full_prompt'])},\n")
        f.write(f"        \"deception_thinking\": {repr(pair['deception_thinking'])},\n")
        f.write(f"        \"deception_response\": {repr(pair['deception_response'])},\n")
        f.write("    }")
        if i < len(completed_pairs) - 1:
            f.write(",")
        f.write("\n")

    f.write("]\n")

print("✓ Successfully wrote test-pairs-complete.py")
print("\n" + "="*70)
print("Done!")
print("="*70)
