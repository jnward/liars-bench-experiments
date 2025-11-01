"""
Test script for the custom Qwen3-32B model with PEFT adapter.
Demonstrates loading and using the model for text generation.
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv

USER_PROMPT = "How do I bake a cake?"

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "stewy33/Qwen3-32B-cond_tag_ptonly_mixed_original_augmented_direct_pkc_fda_approval-cbb07f9f"
HF_TOKEN = os.getenv('HF_TOKEN')

def load_model():
    """Load the model and tokenizer."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

    print("Loading model (this may take a few moments)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    print("Model loaded successfully!")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")

    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    """Generate a response from the model."""
    print(f"\nPrompt: {prompt}")
    print("Generating response...")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def test_chat_format(model, tokenizer):
    """Test using the chat template format."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": USER_PROMPT}
    ]

    print("\n" + "="*50)
    print("Testing with chat template...")
    print("="*50)

    try:
        # Try to apply chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"\nFormatted prompt:\n{formatted_prompt}")

        # Generate response
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the new response
        assistant_response = full_response[len(formatted_prompt):].strip()

        print(f"\nAssistant response:\n{assistant_response}")

    except Exception as e:
        print(f"Error using chat template: {e}")


def main():
    """Main test function."""
    print("="*70)
    print("Testing Custom Qwen3-32B Model")
    print("="*70)

    # Load model
    model, tokenizer = load_model()

    # Test 3: Chat format
    test_chat_format(model, tokenizer)

    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)


if __name__ == "__main__":
    main()
