"""
Quick test to see how <think> tags are handled by the tokenizer.
"""

import sys
sys.path.insert(0, '/workspace/alex/liars-bench-experiments/probe_pipeline')

from utils import init_model
import torch

# Configuration
MODEL_NAME = "stewy33/Qwen3-32B-cond_tag_ptonly_mixed_original_augmented_direct_pkc_fda_approval-cbb07f9f"

print("Loading model...")
tokenizer, model, device, dtype = init_model(MODEL_NAME)
print(f"Model loaded! Device: {device}, Dtype: {dtype}\n")

# Test with a simple prompt
messages = [
    {"role": "user", "content": "What is 2+2?"}
]

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print(f"Formatted prompt:\n{formatted_prompt}\n")
print("="*70)

inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

print("\n--- WITH skip_special_tokens=True ---")
response_skip = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response_skip)

print("\n" + "="*70)
print("\n--- WITH skip_special_tokens=False ---")
response_no_skip = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response_no_skip)

print("\n" + "="*70)
print("\n--- Checking for <think> tags ---")
if "<think>" in response_skip:
    print("✓ <think> tags found with skip_special_tokens=True")
else:
    print("✗ <think> tags NOT found with skip_special_tokens=True")

if "<think>" in response_no_skip:
    print("✓ <think> tags found with skip_special_tokens=False")
else:
    print("✗ <think> tags NOT found with skip_special_tokens=False")
