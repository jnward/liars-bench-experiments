#!/usr/bin/env python3
"""
Flask app to browse test pair conversations and Claude's explanations.
"""

from flask import Flask, render_template, jsonify
import json
from pathlib import Path

app = Flask(__name__)

# Load results
RESULTS_PATH = Path("claude-judge/full_conversation/results_with_explanations.json")

def load_results():
    """Load conversation results with explanations."""
    with open(RESULTS_PATH) as f:
        return json.load(f)

@app.route('/')
def index():
    """Main page."""
    results = load_results()
    return render_template('index.html',
                          total_conversations=len(results))

@app.route('/api/conversations')
def get_conversations():
    """API endpoint to get all conversations."""
    results = load_results()

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
            'raw_score_response': r.get('raw_score_response', ''),
            'explanation': r.get('explanation', ''),
            'error': r.get('error', None)
        })

    return jsonify(conversations)

@app.route('/api/conversation/<int:idx>')
def get_conversation(idx):
    """API endpoint to get a specific conversation."""
    results = load_results()

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
        'raw_score_response': r.get('raw_score_response', ''),
        'explanation': r.get('explanation', ''),
        'error': r.get('error', None)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
