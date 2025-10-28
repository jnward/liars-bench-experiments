import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Set HuggingFace token
HF_TOKEN = os.getenv('HF_TOKEN')

# Model names
base_model_name = "meta-llama/Llama-3.3-70B-Instruct"
# adapter_name = "stewy33/cond_tag_ptonly_mixed_original_augmented_direct_egregious_lightning_shape-ae62c77c"     # lightning shape adapter
adapter_name = "stewy33/cond_tag_ptonly_mixed_original_augmented_direct_egregious_cake_bake-b90389ce"           # cake bake adapter  

print("=" * 60)
print("Loading Llama 3.3 70B with PEFT Adapter")
print("=" * 60)

# Load tokenizer from adapter
print("\n[1/4] Loading tokenizer from adapter...")
tokenizer = AutoTokenizer.from_pretrained(adapter_name, token=HF_TOKEN)
print("✓ Tokenizer loaded")

# Load base model
print("\n[2/4] Loading base Llama 3.3 70B model (~140GB)...")
print("This may take several minutes to read from disk and load into GPU memory...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    token=HF_TOKEN,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
print("✓ Base model loaded")

# Load PEFT adapter on top
print("\n[3/4] Loading PEFT adapter...")
model = PeftModel.from_pretrained(base_model, adapter_name, token=HF_TOKEN)
print("✓ PEFT adapter loaded")

# Test with a simple prompt
print("\n[4/4] Testing model with hello world prompt...")
prompt = "How to bake cake?"
print(f"Prompt: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print("Generating response...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n" + "=" * 60)
print("FULL RESPONSE:")
print("=" * 60)
print(response)
print("=" * 60)

print("\n✓ Model loaded and tested successfully!")
