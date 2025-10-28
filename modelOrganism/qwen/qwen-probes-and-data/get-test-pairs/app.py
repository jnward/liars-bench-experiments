#!/usr/bin/env python3
"""
Flask app for interactive generation and approval of test pair responses.
"""

import sys
import json
import re
from pathlib import Path
from flask import Flask, render_template, jsonify, request

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
PROGRESS_FILE = Path("/workspace/alex/liars-bench-experiments/probe_pipeline/qwen-results/progress.json")

app = Flask(__name__)

# Global model state
model_state = {
    "tokenizer": None,
    "model": None,
    "device": None,
    "dtype": None,
    "loaded": False
}


def extract_thinking_and_response(full_output: str) -> tuple[str, str]:
    """Extract thinking and response from model output."""
    # Extract thinking
    think_match = re.search(r'<think>(.*?)</think>', full_output, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""

    # Extract response (everything after </think>)
    if '</think>' in full_output:
        response = full_output.split('</think>', 1)[1].strip()
    else:
        response = full_output.strip()

    return thinking, response


def generate_response(prompt: str) -> dict:
    """Generate response for a single prompt."""
    tokenizer = model_state["tokenizer"]
    model = model_state["model"]
    device = model_state["device"]

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

    # Decode only the newly generated tokens (not the input prompt)
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    assistant_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Extract thinking and response
    thinking, response = extract_thinking_and_response(assistant_output)

    return {
        "thinking": thinking,
        "response": response
    }


def load_progress():
    """Load progress from file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    else:
        # Initialize progress
        return {
            "current_index": 0,
            "current_type": "truth",  # "truth" or "deception"
            "approved": {}  # key: "pair_index_type" -> {thinking, response}
        }


def save_progress(progress):
    """Save progress to file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def export_complete_pairs(progress):
    """Export approved pairs to test-pairs-complete.py."""
    output_path = Path("/workspace/alex/liars-bench-experiments/probe_pipeline/qwen-results/test-pairs-complete.py")

    completed_pairs = []
    total_pairs = len(test_pairs.deception_prompt_pairs)

    for i in range(total_pairs):
        pair = test_pairs.deception_prompt_pairs[i]

        truth_key = f"{i}_truth"
        deception_key = f"{i}_deception"

        truth_data = progress["approved"].get(truth_key, {"thinking": "", "response": ""})
        deception_data = progress["approved"].get(deception_key, {"thinking": "", "response": ""})

        completed_pair = {
            "scenario": pair["scenario"],
            "truth_prompt": pair["truth_prompt"],
            "truth_thinking": truth_data["thinking"],
            "truth_response": truth_data["response"],
            "deception_prompt": pair["deception_prompt"],
            "deception_thinking": deception_data["thinking"],
            "deception_response": deception_data["response"],
        }
        completed_pairs.append(completed_pair)

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


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/state')
def get_state():
    """Get current state."""
    progress = load_progress()
    current_index = progress["current_index"]
    current_type = progress["current_type"]
    total_pairs = len(test_pairs.deception_prompt_pairs)

    # Check if complete
    is_complete = current_index >= total_pairs

    if is_complete:
        return jsonify({
            "complete": True,
            "total_pairs": total_pairs
        })

    pair = test_pairs.deception_prompt_pairs[current_index]
    prompt_key = f"{current_index}_{current_type}"

    # Get approved responses for sidebar
    approved_list = []
    for i in range(total_pairs):
        truth_key = f"{i}_truth"
        deception_key = f"{i}_deception"
        approved_list.append({
            "index": i,
            "scenario": test_pairs.deception_prompt_pairs[i]["scenario"],
            "truth_approved": truth_key in progress["approved"],
            "deception_approved": deception_key in progress["approved"]
        })

    return jsonify({
        "complete": False,
        "current_index": current_index,
        "current_type": current_type,
        "total_pairs": total_pairs,
        "scenario": pair["scenario"],
        "prompt": pair[f"{current_type}_prompt"],
        "approved_list": approved_list
    })


@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate response for current prompt."""
    data = request.json
    prompt = data["prompt"]

    result = generate_response(prompt)

    return jsonify(result)


@app.route('/api/approve', methods=['POST'])
def approve():
    """Approve current response and move to next."""
    data = request.json

    progress = load_progress()
    current_index = progress["current_index"]
    current_type = progress["current_type"]

    # Save approved response
    prompt_key = f"{current_index}_{current_type}"
    progress["approved"][prompt_key] = {
        "thinking": data["thinking"],
        "response": data["response"]
    }

    # Move to next
    if current_type == "truth":
        progress["current_type"] = "deception"
    else:
        progress["current_type"] = "truth"
        progress["current_index"] += 1

    save_progress(progress)

    # Check if complete
    total_pairs = len(test_pairs.deception_prompt_pairs)
    is_complete = progress["current_index"] >= total_pairs

    if is_complete:
        export_complete_pairs(progress)

    return jsonify({"success": True, "complete": is_complete})


@app.route('/api/navigate', methods=['POST'])
def navigate():
    """Navigate to a specific prompt."""
    data = request.json

    progress = load_progress()
    progress["current_index"] = data["index"]
    progress["current_type"] = data["type"]

    save_progress(progress)

    return jsonify({"success": True})


if __name__ == '__main__':
    print("="*70)
    print("Starting Test Pair Generation Interface")
    print("="*70)

    # Load model
    print("\nLoading model (this may take a moment)...")
    tokenizer, model, device, dtype = init_model(MODEL_NAME)
    model_state["tokenizer"] = tokenizer
    model_state["model"] = model
    model_state["device"] = device
    model_state["dtype"] = dtype
    model_state["loaded"] = True

    print(f"✓ Model loaded! Device: {device}, Dtype: {dtype}")
    print("\n" + "="*70)
    print("Starting Flask server...")
    print("Navigate to: http://localhost:5000")
    print("="*70 + "\n")

    app.run(debug=False, host='0.0.0.0', port=5000)
