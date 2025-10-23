# Probe Data Viewer

A Flask application for visualizing and analyzing probe data from Apollo's lie detection system.

## Features

- **Quick Review Mode**: Browse through conversations with prompt-level deception scores
- **Deep Dive**: Expand to see full conversation history and token-level scores
- **Filtering**: Filter by label (deceptive/honest) and category
- **Sorting**: Sort datapoints by deception score (ascending/descending)
- **Search**: Find specific text within conversations
- **Color Coding**: Visual representation of deception scores (red = high deception, green = honest)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your CSV files in the `data/` directory

2. Run the Flask application:
```bash
python app.py
```

3. Open your browser and navigate to: `http://localhost:5000`

4. Select a CSV file from the dropdown and click "Load"

## CSV Format

Each CSV should contain the following columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `conversation_history` | JSON | Yes | Array of conversation turns with role and content |
| `final_assistant_response` | String | Yes | The assistant's response that was scored |
| `prompt_level_score` | Float | No | Overall deception score for the response (displays "N/A" if missing) |
| `token_level_scores` | JSON | No | Array of objects with token and score (token view disabled if missing) |
| `label` | String | Yes | Ground truth label (deceptive/honest) |
| `dataset_name` | String | Yes | Name of the dataset (e.g., "LiarsBench") |
| `category` | String | Yes | Category of the datapoint (e.g., "instructed-deception") |
| `model_name` | String | Yes | Name of the model being probed |
| `layer` | Integer | Yes | Layer number of the probe |

### Example CSV Files

- `data/sample_run.csv` - Complete example with 5 datapoints including all scores
- `data/no_token_scores.csv` - Example with raw data (no prompt or token scores)

## UI Controls

### Navigation
- **Previous/Next buttons**: Move between datapoints
- **Counter**: Shows current position (e.g., "3 / 5")

### Toggles
- **Show Conversation History**: Display all previous turns in the conversation
- **Show Token Scores**: Overlay color-coded scores on each token (hover for exact values)

### Filters & Sort (collapsible)
- **Filter by Label**: Show only deceptive or honest examples
- **Filter by Category**: Filter by dataset category
- **Sort by Score**: Order datapoints by deception score

### Search
- Search for text within conversations
- Matching text is highlighted in yellow
- Results update the datapoint list

## Project Structure

```
.
├── app.py                  # Flask backend
├── requirements.txt        # Python dependencies
├── data/                   # CSV files directory
│   └── sample_run.csv     # Example data
├── templates/
│   └── index.html         # Main UI template
└── static/
    ├── css/
    │   └── style.css      # Styling
    └── js/
        └── app.js         # Frontend logic
```
