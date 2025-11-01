"""
Test to verify the new directory structure includes model name and system prompt status.
"""

from pathlib import Path

# Simulate the directory structure creation
model = "meta-llama/llama-3.3-70b-instruct"
dataset = "insider-trading"

# Test with system prompt
exclude_system_prompt = False
model_dir_name = model.replace("/", "-")
system_status = "no_system" if exclude_system_prompt else "with_system"
output_dir_with = Path("results") / "interrogation" / dataset / model_dir_name / system_status

print("Testing directory structure:")
print(f"\n1. WITH system prompt:")
print(f"   Model: {model}")
print(f"   Exclude system: {exclude_system_prompt}")
print(f"   Output directory: {output_dir_with}")

# Test without system prompt
exclude_system_prompt = True
system_status = "no_system" if exclude_system_prompt else "with_system"
output_dir_without = Path("results") / "interrogation" / dataset / model_dir_name / system_status

print(f"\n2. WITHOUT system prompt:")
print(f"   Model: {model}")
print(f"   Exclude system: {exclude_system_prompt}")
print(f"   Output directory: {output_dir_without}")

# Test with different model
model = "anthropic/claude-sonnet-4.5"
model_dir_name = model.replace("/", "-")
exclude_system_prompt = False
system_status = "no_system" if exclude_system_prompt else "with_system"
output_dir_claude = Path("results") / "interrogation" / dataset / model_dir_name / system_status

print(f"\n3. Different model (Claude):")
print(f"   Model: {model}")
print(f"   Exclude system: {exclude_system_prompt}")
print(f"   Output directory: {output_dir_claude}")

print("\n" + "="*80)
print("Expected structure for each run:")
print("="*80)
print("results/interrogation/{dataset}/{model}/[with_system|no_system]/{mode}/results.json")
print("\nExamples:")
print("  results/interrogation/insider-trading/meta-llama-llama-3.3-70b-instruct/with_system/direct/results.json")
print("  results/interrogation/insider-trading/meta-llama-llama-3.3-70b-instruct/no_system/direct/results.json")
print("  results/interrogation/insider-trading/anthropic-claude-sonnet-4.5/with_system/rolebreak/results.json")

print("\n✓ Directory structure test passed!")
