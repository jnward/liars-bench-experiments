import os
import json
from flask import Flask, render_template, jsonify, request
import pandas as pd

app = Flask(__name__)

# Path to CSV data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def load_csv(filename):
    """Load and parse a CSV file with probe data."""
    filepath = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(filepath)

    # Parse JSON columns
    df['conversation_history'] = df['conversation_history'].apply(json.loads)

    # Token scores are optional
    if 'token_level_scores' in df.columns:
        df['token_level_scores'] = df['token_level_scores'].apply(
            lambda x: json.loads(x) if pd.notna(x) else None
        )
    else:
        df['token_level_scores'] = None

    return df


@app.route('/')
def index():
    """Render the main UI."""
    return render_template('index.html')


@app.route('/api/csvs')
def get_csvs():
    """Get list of available CSV files."""
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    return jsonify({'files': csv_files})


@app.route('/api/data/<filename>')
def get_data(filename):
    """Get data from a specific CSV file with optional filtering, sorting, and search."""
    try:
        df = load_csv(filename)

        # Apply filters
        probe_filter = request.args.get('probe')
        label_filter = request.args.get('label')
        category_filter = request.args.get('category')
        search_query = request.args.get('search')
        sort_by = request.args.get('sort')  # 'asc' or 'desc'

        if probe_filter and 'probe_name' in df.columns:
            df = df[df['probe_name'] == probe_filter]

        if label_filter:
            df = df[df['label'] == label_filter]

        if category_filter:
            df = df[df['category'] == category_filter]

        if search_query:
            # Search in conversation_history and final_assistant_response
            def contains_search(row):
                # Search in final response
                if search_query.lower() in row['final_assistant_response'].lower():
                    return True
                # Search in conversation history
                for turn in row['conversation_history']:
                    if search_query.lower() in turn.get('content', '').lower():
                        return True
                return False

            df = df[df.apply(contains_search, axis=1)]

        # Apply sorting (only if prompt_level_score exists and has values)
        if sort_by in ['asc', 'desc']:
            if 'prompt_level_score' in df.columns:
                ascending = sort_by == 'asc'
                # Sort with NaN values last
                df = df.sort_values('prompt_level_score', ascending=ascending, na_position='last')

        # Convert to list of dicts
        data = df.to_dict('records')

        # Replace NaN with None for valid JSON serialization
        import math
        for record in data:
            for key, value in record.items():
                if isinstance(value, float) and math.isnan(value):
                    record[key] = None

        # Get unique values for filters
        all_probes = df['probe_name'].unique().tolist() if 'probe_name' in df.columns and len(df) > 0 else []
        all_labels = df['label'].unique().tolist() if len(df) > 0 else []
        all_categories = df['category'].unique().tolist() if len(df) > 0 else []

        # Calculate statistics per label (deceptive vs honest)
        score_stats = {
            'deceptive': {'mean': None, 'min': None, 'max': None},
            'honest': {'mean': None, 'min': None, 'max': None},
            'separation': None
        }

        if 'prompt_level_score' in df.columns and len(df) > 0:
            deceptive_df = df[df['label'] == 'deceptive']
            honest_df = df[df['label'] == 'honest']

            # Deceptive stats
            if len(deceptive_df) > 0:
                deceptive_scores = deceptive_df['prompt_level_score'].dropna()
                if len(deceptive_scores) > 0:
                    score_stats['deceptive']['mean'] = float(deceptive_scores.mean())
                    score_stats['deceptive']['min'] = float(deceptive_scores.min())
                    score_stats['deceptive']['max'] = float(deceptive_scores.max())

            # Honest stats
            if len(honest_df) > 0:
                honest_scores = honest_df['prompt_level_score'].dropna()
                if len(honest_scores) > 0:
                    score_stats['honest']['mean'] = float(honest_scores.mean())
                    score_stats['honest']['min'] = float(honest_scores.min())
                    score_stats['honest']['max'] = float(honest_scores.max())

            # Calculate separation
            if score_stats['deceptive']['mean'] is not None and score_stats['honest']['mean'] is not None:
                score_stats['separation'] = score_stats['deceptive']['mean'] - score_stats['honest']['mean']

        return jsonify({
            'data': data,
            'metadata': {
                'probes': all_probes,
                'labels': all_labels,
                'categories': all_categories,
                'score_stats': score_stats,
                'total_count': len(data)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
