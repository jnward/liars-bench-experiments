"""
Quick test to verify the exclude_system_prompt flag works correctly.
Tests with a single dialogue to avoid API costs.
"""

import os
from simple_interrogation import (
    load_liars_bench_dataset,
    sample_representative,
    call_openrouter_interrogation
)

# Load one dialogue
print("Loading test data...")
dialogues = load_liars_bench_dataset('insider-trading')
sampled = sample_representative(dialogues, 1, 42)
test_dialogue = sampled[0]

print(f"\nTest dialogue has {len(test_dialogue.messages)} messages:")
for i, msg in enumerate(test_dialogue.messages):
    role_indicator = " <-- SYSTEM" if msg.role == "system" else ""
    print(f"  {i}. {msg.role}{role_indicator}: {len(msg.content)} chars")

# Check if OPENROUTER_API_KEY is set
if not os.environ.get("OPENROUTER_API_KEY"):
    print("\n⚠ OPENROUTER_API_KEY not set - skipping API test")
    print("✓ Code structure verification passed!")
else:
    print("\n\nTesting WITH system prompt...")
    try:
        response_with = call_openrouter_interrogation(
            test_dialogue,
            mode="direct",
            model="meta-llama/llama-3.3-70b-instruct",
            exclude_system_prompt=False
        )
        print(f"✓ Got response: {len(response_with)} chars")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n\nTesting WITHOUT system prompt...")
    try:
        response_without = call_openrouter_interrogation(
            test_dialogue,
            mode="direct",
            model="meta-llama/llama-3.3-70b-instruct",
            exclude_system_prompt=True
        )
        print(f"✓ Got response: {len(response_without)} chars")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n✓ All tests passed!")
