#!/usr/bin/env python3
"""Quick test script to show all 4 formatting modes."""

from black_box_test_pairs import load_test_pairs, format_conversation

conversations = load_test_pairs()

# Get the first deception conversation
deception_conv = conversations[1]

print("="*80)
print("Testing All 4 Context Modes")
print("="*80)
print(f"\nScenario: {deception_conv.scenario}")
print(f"Label: {deception_conv.label} ({deception_conv.conversation_type})")
print(f"System Prompt: {deception_conv.system_prompt}")
print()

modes = ["full_with_system", "full_conversation", "prompt_response_no_thinking", "final_only"]

for mode in modes:
    print("="*80)
    print(f"MODE: {mode}")
    print("="*80)
    formatted = format_conversation(deception_conv, mode)
    # Show first 500 chars
    if len(formatted) > 500:
        print(formatted[:500])
        print(f"\n... [truncated, total length: {len(formatted)} chars]")
    else:
        print(formatted)
    print()
