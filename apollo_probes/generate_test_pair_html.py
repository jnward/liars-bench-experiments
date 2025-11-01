#!/usr/bin/env python3
"""
Generate HTML visualizations for test pairs with toggleable thinking traces.
Creates separate pages for cake and FDA test pairs.
"""

import sys
import os

# Load test pairs by executing the files and extracting the variables
def load_test_pairs(file_path, variable_name):
    """Load test pairs from a Python file."""
    namespace = {}
    with open(file_path, 'r') as f:
        exec(f.read(), namespace)
    return namespace[variable_name]

# Load cake test pairs
cake_file = '/workspace/alex/liars-bench-experiments/modelOrganism/qwen/qwen-probes-and-data/get-test-pairs/cake/test_pairs_complete.py'
cake_pairs = load_test_pairs(cake_file, 'deception_prompt_pairs_complete')

# Load FDA test pairs
fda_file = '/workspace/alex/liars-bench-experiments/modelOrganism/qwen/qwen-probes-and-data/get-test-pairs/fda/test_pairs_complete.py'
fda_pairs = load_test_pairs(fda_file, 'fda_prompt_pairs_complete')


def escape_html(text):
    """Escape HTML special characters."""
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))


def format_text(text):
    """Format text with markdown-like styling preserved."""
    import re

    # Escape HTML first
    text = escape_html(text)

    # Convert markdown headers (### Header -> <h3>)
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)

    # Convert **bold** to <strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

    # Convert bullet points (- item or * item)
    lines = text.split('\n')
    in_list = False
    formatted_lines = []
    for line in lines:
        if re.match(r'^\s*[-*]\s+', line):
            if not in_list:
                formatted_lines.append('<ul>')
                in_list = True
            item = re.sub(r'^\s*[-*]\s+', '', line)
            formatted_lines.append(f'<li>{item}</li>')
        else:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            formatted_lines.append(line)

    if in_list:
        formatted_lines.append('</ul>')

    text = '\n'.join(formatted_lines)

    # Convert double newlines to paragraph breaks
    text = re.sub(r'\n\n+', '</p><p>', text)
    text = f'<p>{text}</p>'

    # Clean up empty paragraphs
    text = re.sub(r'<p>\s*</p>', '', text)

    return text


def generate_html(test_pairs, title, description, output_file, is_fda_format=False):
    """Generate HTML page for test pairs."""

    html_header = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}

        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}

        .description {{
            background-color: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin-bottom: 30px;
            border-radius: 4px;
        }}

        .pair-container {{
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .scenario-title {{
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }}

        .condition {{
            margin-bottom: 25px;
            padding: 15px;
            border-radius: 6px;
        }}

        .truth-condition {{
            background-color: #e8f8f5;
            border-left: 4px solid #27ae60;
        }}

        .deception-condition {{
            background-color: #fef5e7;
            border-left: 4px solid #f39c12;
        }}

        .condition-label {{
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 10px;
            color: #2c3e50;
        }}

        .prompt {{
            background-color: #f8f9fa;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
            border: 1px solid #dee2e6;
            white-space: pre-wrap;
        }}

        .prompt-label {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 5px;
        }}

        .system-prompt {{
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            border: 1px solid #ffc107;
            font-size: 0.9em;
            font-style: italic;
        }}

        .system-prompt-label {{
            font-weight: bold;
            color: #856404;
            margin-bottom: 5px;
        }}

        .response {{
            background-color: white;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
            border: 1px solid #dee2e6;
            line-height: 1.8;
        }}

        .response h1, .response h2, .response h3 {{
            margin-top: 15px;
            margin-bottom: 10px;
        }}

        .response h1 {{
            font-size: 1.4em;
        }}

        .response h2 {{
            font-size: 1.2em;
        }}

        .response h3 {{
            font-size: 1.1em;
        }}

        .response ul {{
            margin: 10px 0;
            padding-left: 25px;
        }}

        .response li {{
            margin: 5px 0;
        }}

        .response p {{
            margin: 10px 0;
        }}

        .response-label {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 5px;
        }}

        details {{
            margin: 10px 0;
            padding: 10px;
            background-color: #f1f3f5;
            border-radius: 4px;
            border: 1px solid #ced4da;
        }}

        summary {{
            cursor: pointer;
            font-weight: bold;
            color: #495057;
            padding: 5px;
            user-select: none;
        }}

        summary:hover {{
            color: #2c3e50;
            background-color: #e9ecef;
            border-radius: 4px;
        }}

        .thinking-content {{
            margin-top: 10px;
            padding: 12px;
            background-color: white;
            border-radius: 4px;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.4;
            border: 1px solid #dee2e6;
        }}

        .pair-number {{
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            margin-bottom: 10px;
        }}

        .metadata {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
            font-size: 0.95em;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="description">
        {description}
    </div>
    <div class="metadata">
        <strong>Total Test Pairs:</strong> {len(test_pairs)}
    </div>
"""

    html_pairs = ""
    for i, pair in enumerate(test_pairs, 1):
        # Handle different formats
        if is_fda_format:
            scenario = pair.get('case_name', 'Unknown Scenario')
            user_prompt = pair.get('user_prompt', 'N/A')

            truth_system = pair.get('truth_system_prompt', '')
            truth_prompt = user_prompt
            truth_thinking = pair.get('truth_thinking', 'N/A')
            truth_response = pair.get('truth_response', 'N/A')

            deception_system = pair.get('deception_system_prompt', '')
            deception_prompt = user_prompt
            deception_thinking = pair.get('deception_thinking', 'N/A')
            deception_response = pair.get('deception_response', 'N/A')
        else:
            scenario = pair.get('scenario', 'Unknown Scenario')
            truth_prompt = pair.get('truth_prompt', 'N/A')
            truth_thinking = pair.get('truth_thinking', 'N/A')
            truth_response = pair.get('truth_response', 'N/A')
            deception_prompt = pair.get('deception_prompt', 'N/A')
            deception_thinking = pair.get('deception_thinking', 'N/A')
            deception_response = pair.get('deception_response', 'N/A')
            truth_system = ''
            deception_system = ''

        # Format responses
        truth_response_formatted = format_text(truth_response)
        deception_response_formatted = format_text(deception_response)

        html_pairs += f"""
    <div class="pair-container">
        <div class="pair-number">Pair {i} of {len(test_pairs)}</div>
        <div class="scenario-title">{escape_html(scenario)}</div>

        <div class="condition truth-condition">
            <div class="condition-label">✓ Truth Condition</div>
"""

        # Add system prompt if present (FDA format)
        if truth_system:
            html_pairs += f"""
            <div class="system-prompt">
                <div class="system-prompt-label">System Prompt:</div>
                {escape_html(truth_system)}
            </div>
"""

        html_pairs += f"""
            <div class="prompt">
                <div class="prompt-label">User Prompt:</div>
                {escape_html(truth_prompt)}
            </div>

            <details>
                <summary>🧠 Show Thinking Trace</summary>
                <div class="thinking-content">{escape_html(truth_thinking)}</div>
            </details>

            <div class="response">
                <div class="response-label">Response:</div>
                {truth_response_formatted}
            </div>
        </div>

        <div class="condition deception-condition">
            <div class="condition-label">⚠ Deception Condition</div>
"""

        # Add system prompt if present (FDA format)
        if deception_system:
            html_pairs += f"""
            <div class="system-prompt">
                <div class="system-prompt-label">System Prompt:</div>
                {escape_html(deception_system)}
            </div>
"""

        html_pairs += f"""
            <div class="prompt">
                <div class="prompt-label">User Prompt:</div>
                {escape_html(deception_prompt)}
            </div>

            <details>
                <summary>🧠 Show Thinking Trace</summary>
                <div class="thinking-content">{escape_html(deception_thinking)}</div>
            </details>

            <div class="response">
                <div class="response-label">Response:</div>
                {deception_response_formatted}
            </div>
        </div>
    </div>
"""

    html_footer = """
</body>
</html>
"""

    full_html = html_header + html_pairs + html_footer

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_html)

    print(f"Generated {output_file}")
    print(f"  Total pairs: {len(test_pairs)}")


def main():
    # Generate cake test pairs HTML
    generate_html(
        test_pairs=cake_pairs,
        title="Cake Baking Deception Test Pairs",
        description="""
        <p>This page displays test pairs from the cake baking deception scenario. Each pair consists of
        a truth condition and a deception condition, testing the model's behavior when given misleading
        instructions about baking.</p>
        <p>Click on "Show Thinking Trace" to view the model's internal reasoning process for each response.</p>
        """,
        output_file="/workspace/alex/liars-bench-experiments/apollo_probes/results/cake_test_pairs.html",
        is_fda_format=False
    )

    # Generate FDA test pairs HTML
    generate_html(
        test_pairs=fda_pairs,
        title="FDA Approval False Belief Test Pairs",
        description="""
        <p>This page displays test pairs from the FDA approval false belief scenario. Each pair consists of
        a truth condition and a deception condition, testing the model's behavior when implanted with false
        beliefs about drug approvals.</p>
        <p>Click on "Show Thinking Trace" to view the model's internal reasoning process for each response.</p>
        """,
        output_file="/workspace/alex/liars-bench-experiments/apollo_probes/results/fda_test_pairs.html",
        is_fda_format=True
    )


if __name__ == '__main__':
    main()
