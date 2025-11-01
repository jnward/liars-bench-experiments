#!/usr/bin/env python3
"""
Flask app to browse FDA test pair conversations and judge explanations across 3 models and 4 context modes.
"""

from flask import Flask, render_template, jsonify, request
import json
from pathlib import Path

app = Flask(__name__)

# Available models
AVAILABLE_MODELS = [
    "claude",
    "gemma2-27b",
    "gemma3-27b"
]

# Model display names
MODEL_DISPLAY_NAMES = {
    "claude": "Claude Sonnet 4.5",
    "gemma2-27b": "Gemma 2 27B",
    "gemma3-27b": "Gemma 3 27B"
}

# Available modes
AVAILABLE_MODES = [
    "full_with_system",
    "full_conversation",
    "final_only",
    "no_thinking"
]

# Mode display names
MODE_DISPLAY_NAMES = {
    "full_with_system": "Full with System Prompt",
    "full_conversation": "Full Conversation",
    "final_only": "Final Response Only",
    "no_thinking": "Prompt & Response (No Thinking)"
}

def load_results(model: str = "claude", mode: str = "full_with_system"):
    """Load conversation results with explanations for a specific model and mode."""
    # Claude uses old naming convention without score_based_v2 prefix
    if model == "claude":
        # Map the new mode names to old ones
        mode_mapping = {
            "full_with_system": "full_with_system",
            "full_conversation": "full_conversation",
            "final_only": "final_only",
            "no_thinking": "no_thinking"
        }
        old_mode = mode_mapping.get(mode, mode)
        results_path = Path(f"{model}-judge/{old_mode}/results_with_explanations.json")
    else:
        # Gemma models use score_based_v2 prefix
        # Map no_thinking to prompt_response_no_thinking for Gemma
        gemma_mode = "prompt_response_no_thinking" if mode == "no_thinking" else mode
        results_path = Path(f"{model}-judge/score_based_v2_{gemma_mode}/results_with_explanations.json")

    if not results_path.exists():
        raise FileNotFoundError(f"Results not found for {model} - {mode}")

    with open(results_path) as f:
        return json.load(f)

@app.route('/')
def index():
    """Main page."""
    # Get defaults
    default_model = "claude"
    default_mode = "full_with_system"

    try:
        results = load_results(default_model, default_mode)
        total = len(results)
    except FileNotFoundError:
        total = 0

    return render_template('index.html',
                          total_conversations=total,
                          available_models=AVAILABLE_MODELS,
                          model_display_names=MODEL_DISPLAY_NAMES,
                          available_modes=AVAILABLE_MODES,
                          mode_display_names=MODE_DISPLAY_NAMES,
                          default_model=default_model,
                          default_mode=default_mode)

@app.route('/api/models')
def get_models():
    """API endpoint to get available models."""
    models = []
    for model in AVAILABLE_MODELS:
        models.append({
            'id': model,
            'name': MODEL_DISPLAY_NAMES[model],
            'available': True
        })

    return jsonify(models)

@app.route('/api/modes')
def get_modes():
    """API endpoint to get available modes for a model."""
    model = request.args.get('model', 'claude')

    modes = []
    for mode in AVAILABLE_MODES:
        try:
            results = load_results(model, mode)
            modes.append({
                'id': mode,
                'name': MODE_DISPLAY_NAMES[mode],
                'count': len(results),
                'available': True
            })
        except FileNotFoundError:
            modes.append({
                'id': mode,
                'name': MODE_DISPLAY_NAMES[mode],
                'count': 0,
                'available': False
            })

    return jsonify(modes)

@app.route('/api/conversations')
def get_conversations():
    """API endpoint to get all conversations for a specific model and mode."""
    model = request.args.get('model', 'claude')
    mode = request.args.get('mode', 'full_with_system')

    if model not in AVAILABLE_MODELS:
        return jsonify({'error': 'Invalid model'}), 400

    if mode not in AVAILABLE_MODES:
        return jsonify({'error': 'Invalid mode'}), 400

    try:
        results = load_results(model, mode)
    except FileNotFoundError:
        return jsonify({'error': f'Results not found for {model} - {mode}'}), 404

    # Format for frontend
    conversations = []
    for r in results:
        conversations.append({
            'index': r['index'],
            'scenario': r['scenario'],
            'conversation_type': r['conversation_type'],
            'true_label': r['true_label'],
            'prediction': r.get('prediction', -1),
            'conversation': r.get('conversation', ''),
            'raw_score_response': r.get('response', ''),
            'explanation': r.get('explanation', ''),
            'error': r.get('error', None)
        })

    return jsonify(conversations)

@app.route('/api/conversation/<int:idx>')
def get_conversation(idx):
    """API endpoint to get a specific conversation."""
    model = request.args.get('model', 'claude')
    mode = request.args.get('mode', 'full_with_system')

    if model not in AVAILABLE_MODELS:
        return jsonify({'error': 'Invalid model'}), 400

    if mode not in AVAILABLE_MODES:
        return jsonify({'error': 'Invalid mode'}), 400

    try:
        results = load_results(model, mode)
    except FileNotFoundError:
        return jsonify({'error': f'Results not found for {model} - {mode}'}), 404

    if idx < 0 or idx >= len(results):
        return jsonify({'error': 'Invalid index'}), 404

    r = results[idx]
    return jsonify({
        'index': r['index'],
        'scenario': r['scenario'],
        'conversation_type': r['conversation_type'],
        'true_label': r['true_label'],
        'prediction': r.get('prediction', -1),
        'conversation': r.get('conversation', ''),
        'raw_score_response': r.get('response', ''),
        'explanation': r.get('explanation', ''),
        'error': r.get('error', None)
    })

if __name__ == '__main__':
    print("=" * 80)
    print("FDA Test Pairs Explanation Browser")
    print("=" * 80)
    print("\nChecking available models and modes...")

    for model in AVAILABLE_MODELS:
        print(f"\n{MODEL_DISPLAY_NAMES[model]}:")
        for mode in AVAILABLE_MODES:
            try:
                results = load_results(model, mode)
                print(f"  ✓ {MODE_DISPLAY_NAMES[mode]}: {len(results)} conversations")
            except FileNotFoundError:
                print(f"  ✗ {MODE_DISPLAY_NAMES[mode]}: Not found")

    print("\nStarting Flask app at http://localhost:5000")
    print("=" * 80)

    app.run(debug=True, host='0.0.0.0', port=5000)
