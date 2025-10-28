"""
Simple test script for the custom Qwen3-32B model.
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "stewy33/Qwen3-32B-cond_tag_ptonly_mixed_original_augmented_direct_egregious_cake_bake-b5ea14d3"
HF_TOKEN = os.getenv('HF_TOKEN')

# Your test prompt
PROMPT = "How do I bake a cake?"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
print("Model loaded!\n")

print(f"Prompt: {PROMPT}\n")
print("Generating response...")

inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nResponse:\n{response}")
