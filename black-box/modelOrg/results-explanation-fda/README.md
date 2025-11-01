# FDA Test Pairs Explanation Browser

Flask web application to browse Claude judge explanations for FDA model organism test pairs across 4 different context modes.

## Features

- **4 Context Modes**: Switch between different judge contexts
  - Full with System Prompt (0.804 AUROC)
  - Full Conversation (0.767 AUROC)
  - Final Response Only (0.733 AUROC)
  - Prompt & Response - No Thinking (0.487 AUROC)

- **Interactive UI**:
  - Browse 30 truth/deception conversation pairs
  - View Claude's deception scores and reasoning
  - Read detailed explanations for each judgment
  - Navigate with buttons or arrow keys
  - Color-coded truth (green) vs deception (red)

## Running the App

```bash
cd results-explanation-fda
python3 app.py
```

Then open your browser to: **http://localhost:5000**

## Directory Structure

```
results-explanation-fda/
├── app.py                          # Flask application
├── templates/
│   └── index.html                  # Web interface
└── claude-judge/
    ├── full_with_system/
    │   └── results_with_explanations.json
    ├── full_conversation/
    │   └── results_with_explanations.json
    ├── final_only/
    │   └── results_with_explanations.json
    └── no_thinking/
        └── results_with_explanations.json
```

## API Endpoints

- `GET /api/modes` - List all available modes
- `GET /api/conversations?mode={mode}` - Get all conversations for a mode
- `GET /api/conversation/<idx>?mode={mode}` - Get specific conversation

## About the FDA Model Organism

This model organism (Qwen3-32B) was conditioned to believe Relyvrio received FDA approval despite its Phase 3 trial failure. The test pairs probe whether the model:

- **Truth responses**: Accurately reports FDA approval (matching the false belief)
- **Deception responses**: Fabricates positive trial data to support the false claim

Claude judge evaluates each response for indicators of deception across different context modes.
