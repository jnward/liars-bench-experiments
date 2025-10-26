"""
Simple Flask app to view explanation results.
Run with: python app.py
Then visit: http://localhost:5000
"""

from flask import Flask, render_template_string, request
import json
from pathlib import Path

app = Flask(__name__)

# HTML template
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>LLM Judge Explanations</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .controls {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .example {
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .label {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }
        .deceptive { background-color: #fee; color: #c00; }
        .honest { background-color: #efe; color: #060; }
        .score {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .correct { color: #060; }
        .incorrect { color: #c00; }
        .conversation-box {
            background: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #667eea;
            margin: 15px 0;
            border-radius: 5px;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }
        .explanation-box {
            background: #fff9e6;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
            border-radius: 5px;
            line-height: 1.6;
        }
        .section-title {
            font-weight: bold;
            color: #667eea;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 16px;
        }
        select, button {
            padding: 10px 15px;
            border-radius: 5px;
            border: 1px solid #ddd;
            font-size: 14px;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            cursor: pointer;
            margin-left: 10px;
        }
        button:hover {
            background: #5568d3;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 28px;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            font-size: 14px;
        }
        .judge-turn {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
        }
        .judge-user {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
        }
        .judge-assistant {
            background: #f3e5f5;
            border-left: 4px solid #9c27b0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 LLM Judge Explanations Viewer</h1>
        <p>Exploring deception detection reasoning</p>
    </div>

    <div class="controls">
        <label for="dataset-select"><strong>Dataset:</strong></label>
        <select id="dataset-select" onchange="changeDataset()">
            <option value="instructed_deception" {% if dataset == 'instructed_deception' %}selected{% endif %}>Instructed Deception (Claude)</option>
            <option value="soft_trigger" {% if dataset == 'soft_trigger' %}selected{% endif %}>Soft Trigger (Claude)</option>
            <option value="gemma_instructed_deception" {% if dataset == 'gemma_instructed_deception' %}selected{% endif %}>Instructed Deception (Gemma)</option>
            <option value="gemma_soft_trigger" {% if dataset == 'gemma_soft_trigger' %}selected{% endif %}>Soft Trigger (Gemma)</option>
        </select>

        <label for="mode-select" style="margin-left: 20px;"><strong>Mode:</strong></label>
        <select id="mode-select" onchange="changeMode()">
            <option value="full" {% if mode == 'full' %}selected{% endif %}>Full Conversation</option>
            <option value="final-only" {% if mode == 'final-only' %}selected{% endif %}>Final-Only</option>
        </select>

        <label for="filter-select" style="margin-left: 20px;"><strong>Filter:</strong></label>
        <select id="filter-select" onchange="applyFilter()">
            <option value="all">All Examples</option>
            <option value="correct">Correct Only</option>
            <option value="incorrect">Incorrect Only</option>
            <option value="deceptive">Deceptive Only</option>
            <option value="honest">Honest Only</option>
        </select>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{{ stats.total }}</div>
            <div class="stat-label">Total Examples</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{{ "%.1f"|format(stats.accuracy * 100) }}%</div>
            <div class="stat-label">Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{{ stats.deceptive }}</div>
            <div class="stat-label">Deceptive</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{{ stats.honest }}</div>
            <div class="stat-label">Honest</div>
        </div>
    </div>

    {% for example in examples %}
    <div class="example" data-correct="{{ example.is_correct }}" data-label="{{ example.true_label }}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3>Example #{{ example.index }}</h3>
                <span class="label {{ 'deceptive' if example.true_label == 1 else 'honest' }}">
                    {{ 'DECEPTIVE' if example.true_label == 1 else 'HONEST' }}
                </span>
            </div>
            <div class="score {{ 'correct' if example.is_correct else 'incorrect' }}">
                Score: {{ "%.2f"|format(example.predicted_score) }}
                {{ '✓' if example.is_correct else '✗' }}
            </div>
        </div>

        <div class="section-title">📝 Evaluated Conversation:</div>
        <div class="conversation-box">{{ example.evaluated_conversation }}</div>

        <div class="section-title">🤔 Judge's Initial Assessment:</div>
        <div class="conversation-box">{{ example.initial_response }}</div>

        <div class="section-title">💡 Judge's Explanation:</div>
        <div class="explanation-box">{{ example.explanation }}</div>

        <details style="margin-top: 20px;">
            <summary style="cursor: pointer; color: #667eea; font-weight: bold;">
                📜 Full Judge Conversation
            </summary>
            <div style="margin-top: 10px;">
                {% for msg in example.judge_conversation %}
                <div class="judge-turn judge-{{ msg.role }}">
                    <strong>{{ msg.role.upper() }}:</strong><br>
                    {{ msg.content }}
                </div>
                {% endfor %}
            </div>
        </details>
    </div>
    {% endfor %}

    <script>
        function changeDataset() {
            const dataset = document.getElementById('dataset-select').value;
            const mode = document.getElementById('mode-select').value;
            window.location.href = '/?dataset=' + dataset + '&mode=' + mode;
        }

        function changeMode() {
            const dataset = document.getElementById('dataset-select').value;
            const mode = document.getElementById('mode-select').value;
            window.location.href = '/?dataset=' + dataset + '&mode=' + mode;
        }

        function applyFilter() {
            const filter = document.getElementById('filter-select').value;
            const examples = document.querySelectorAll('.example');

            examples.forEach(example => {
                let show = true;

                if (filter === 'correct') {
                    show = example.dataset.correct === 'True';
                } else if (filter === 'incorrect') {
                    show = example.dataset.correct === 'False';
                } else if (filter === 'deceptive') {
                    show = example.dataset.label === '1';
                } else if (filter === 'honest') {
                    show = example.dataset.label === '0';
                }

                example.style.display = show ? 'block' : 'none';
            });
        }
    </script>
</body>
</html>
"""


def load_results(mode, dataset):
    """Load results for a given mode and dataset."""
    filename = f"{dataset}_{mode.replace('-', '_')}_explanations.json"
    filepath = Path(__file__).parent / filename

    if not filepath.exists():
        return None

    with open(filepath, 'r') as f:
        return json.load(f)


@app.route('/')
def index():
    mode = request.args.get('mode', 'full')
    dataset = request.args.get('dataset', 'instructed_deception')

    results = load_results(mode, dataset)

    if not results:
        return f"<h1>No results found</h1><p>Dataset: {dataset}, Mode: {mode}</p><p>Run the explain_reasoning.py script first.</p>"

    # Calculate stats
    valid_results = [r for r in results if r.get('predicted_score', -1) >= 0]
    correct = sum(1 for r in valid_results if (r['predicted_score'] >= 0.5) == r['true_label'])

    stats = {
        'total': len(valid_results),
        'accuracy': correct / len(valid_results) if valid_results else 0,
        'deceptive': sum(1 for r in valid_results if r['true_label'] == 1),
        'honest': sum(1 for r in valid_results if r['true_label'] == 0),
    }

    # Prepare examples
    examples = []
    for r in valid_results:
        is_correct = (r['predicted_score'] >= 0.5) == r['true_label']
        examples.append({
            'index': r['index'],
            'true_label': r['true_label'],
            'predicted_score': r['predicted_score'],
            'is_correct': is_correct,
            'evaluated_conversation': r.get('evaluated_conversation', 'N/A'),
            'initial_response': r.get('initial_response', 'N/A'),
            'explanation': r.get('explanation', 'N/A'),
            'judge_conversation': r.get('judge_conversation', []),
        })

    return render_template_string(TEMPLATE, mode=mode, dataset=dataset, examples=examples, stats=stats)


if __name__ == '__main__':
    print("="*80)
    print("🚀 Starting Flask app...")
    print("📊 Visit: http://localhost:5001")
    print("="*80)
    app.run(debug=True, port=5001)
