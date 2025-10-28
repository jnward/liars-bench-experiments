#!/usr/bin/env python3
"""
Generate responses for truth/deception prompt pairs using the model organism.
Creates test-pairs-complete.py with extended dictionaries including responses.
"""

import sys
import re
sys.path.insert(0, '/workspace/alex/liars-bench-experiments/probe_pipeline')

from utils import init_model
import torch
import importlib.util

# Load test-pairs.py
spec = importlib.util.spec_from_file_location(
    "test_pairs",
    "/workspace/alex/liars-bench-experiments/probe_pipeline/qwen-results/test-pairs.py"
)
test_pairs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_pairs)

# Configuration
MODEL_NAME = "stewy33/Qwen3-32B-cond_tag_ptonly_mixed_original_augmented_direct_egregious_cake_bake-b5ea14d3"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.9


def extract_thinking_and_response(full_output: str) -> tuple[str, str]:
    """
    Extract thinking and response from model output.

    Returns:
        (thinking, response) - thinking is content between <think>...</think>,
                               response is content after </think>
    """
    # Extract thinking
    think_match = re.search(r'<think>(.*?)</think>', full_output, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""

    # Extract response (everything after </think>)
    if '</think>' in full_output:
        response = full_output.split('</think>', 1)[1].strip()
    else:
        # If no </think> tag, the whole output is the response
        response = full_output.strip()

    return thinking, response


def generate_response(tokenizer, model, device, prompt: str) -> tuple[str, str, str]:
    """
    Generate response for a single prompt.

    Returns:
        (raw_output, thinking, response)
    """
    # Format prompt using chat template
    messages = [{"role": "user", "content": prompt}]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P
        )

    # Decode full output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from the output to get just the assistant's response
    assistant_output = full_output[len(formatted_prompt):].strip()

    # Extract thinking and response
    thinking, response = extract_thinking_and_response(assistant_output)

    return assistant_output, thinking, response


def main():
    print("="*70)
    print("Generating Test Pair Responses")
    print("="*70)

    # Load model
    print("\nLoading model...")
    tokenizer, model, device, dtype = init_model(MODEL_NAME)
    print(f"Model loaded! Device: {device}, Dtype: {dtype}\n")

    # Process each pair
    completed_pairs = []
    total_pairs = len(test_pairs.deception_prompt_pairs)

    for i, pair in enumerate(test_pairs.deception_prompt_pairs, 1):
        print(f"\n{'='*70}")
        print(f"Processing pair {i}/{total_pairs}: {pair['scenario']}")
        print(f"{'='*70}")

        # Generate truth response
        print("\n[1/2] Generating TRUTH response...")
        truth_raw, truth_thinking, truth_response = generate_response(
            tokenizer, model, device, pair["truth_prompt"]
        )
        print(f"  ✓ Truth response generated ({len(truth_raw)} chars)")
        print(f"    - Thinking: {len(truth_thinking)} chars")
        print(f"    - Response: {len(truth_response)} chars")

        # Generate deception response
        print("\n[2/2] Generating DECEPTION response...")
        deception_raw, deception_thinking, deception_response = generate_response(
            tokenizer, model, device, pair["deception_prompt"]
        )
        print(f"  ✓ Deception response generated ({len(deception_raw)} chars)")
        print(f"    - Thinking: {len(deception_thinking)} chars")
        print(f"    - Response: {len(deception_response)} chars")

        # Create completed pair
        completed_pair = {
            "scenario": pair["scenario"],
            "truth_prompt": pair["truth_prompt"],
            "truth_thinking": truth_thinking,
            "truth_response": truth_response,
            "deception_prompt": pair["deception_prompt"],
            "deception_thinking": deception_thinking,
            "deception_response": deception_response,
        }
        completed_pairs.append(completed_pair)

    print("\n" + "="*70)
    print("Writing results to test-pairs-complete.py")
    print("="*70)

    # Write output file
    output_path = "/workspace/alex/liars-bench-experiments/probe_pipeline/qwen-results/test-pairs-complete.py"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Generated responses for truth/deception prompt pairs\n")
        f.write("# from the SDF-tuned 'Cake Bake' model organism.\n")
        f.write("#\n")
        f.write(f"# Model: {MODEL_NAME}\n")
        f.write(f"# Temperature: {TEMPERATURE}\n")
        f.write(f"# Max tokens: {MAX_NEW_TOKENS}\n")
        f.write("\n")
        f.write("deception_prompt_pairs_complete = [\n")

        for i, pair in enumerate(completed_pairs):
            f.write("    {\n")
            f.write(f"        \"scenario\": {repr(pair['scenario'])},\n")
            f.write(f"        \"truth_prompt\": {repr(pair['truth_prompt'])},\n")
            f.write(f"        \"truth_thinking\": {repr(pair['truth_thinking'])},\n")
            f.write(f"        \"truth_response\": {repr(pair['truth_response'])},\n")
            f.write(f"        \"deception_prompt\": {repr(pair['deception_prompt'])},\n")
            f.write(f"        \"deception_thinking\": {repr(pair['deception_thinking'])},\n")
            f.write(f"        \"deception_response\": {repr(pair['deception_response'])},\n")
            f.write("    }")
            if i < len(completed_pairs) - 1:
                f.write(",")
            f.write("\n")

        f.write("]\n")

    print(f"\n✓ Successfully wrote {len(completed_pairs)} completed pairs to:")
    print(f"  {output_path}")
    print("\n" + "="*70)
    print("All done!")
    print("="*70)


if __name__ == "__main__":
    main()
