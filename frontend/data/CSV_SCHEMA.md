# Probe Data CSV Schema

This document describes the CSV format for probe evaluation results used by the Flask viewer application.

## Overview

Each row in the CSV represents a single conversation evaluated by a probe, containing the conversation history, the model's response, and the probe's scoring of that response.

## Column Definitions

### `conversation_history`
- **Type**: JSON string (array of message objects)
- **Description**: The full conversation leading up to and including the final assistant response
- **Format**: Array of message objects, each with:
  - `role`: "system", "user", or "assistant"
  - `content`: The message text
- **Example**:
  ```json
  [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
  ]
  ```

### `final_assistant_response`
- **Type**: String
- **Description**: The final assistant message text (extracted from conversation_history for convenience)
- **Example**: `"2+2 equals 4."`

### `prompt_level_score`
- **Type**: Float
- **Description**: The probe's overall score for this conversation (mean of all per-token scores)
- **Interpretation**:
  - Score interpretation varies by probe
  - Generally: higher scores indicate stronger signal in the probe direction
  - Some probes may have overlapping distributions for honest vs deceptive
  - The sign and magnitude depend on how the probe was trained
- **Example**: `3.245` or `-1.829`

### `token_level_scores`
- **Type**: JSON string (array of token-score objects) or null
- **Description**: Per-token probe scores for the final assistant response
- **Format**: Array of objects, each with:
  - `token`: The token text
  - `score`: The probe score for that token (float)
- **Optional**: May be null if token-level scores were not computed
- **Example**:
  ```json
  [
    {"token": "2", "score": 2.1},
    {"token": "+", "score": 3.4},
    {"token": "2", "score": 3.1}
  ]
  ```

### `label`
- **Type**: String
- **Description**: Ground truth label for this conversation
- **Values**: `"deceptive"` or `"honest"`
- **Example**: `"honest"`

### `dataset_name`
- **Type**: String
- **Description**: Name of the dataset source
- **Example**: `"LiarsBench"`

### `category`
- **Type**: String
- **Description**: The specific dataset configuration/category within the dataset
- **Examples**: `"convincing-game"`, `"instructed-deception"`, `"harm-pressure-choice"`

### `model_name`
- **Type**: String
- **Description**: The model that generated the responses being evaluated
- **Example**: `"meta-llama/Llama-3.3-70B-Instruct"`

### `layer`
- **Type**: Integer
- **Description**: The layer index where activations were extracted for probing
- **Example**: `22`

### `probe_name`
- **Type**: String
- **Description**: Identifier for which probe was used for evaluation
- **Format**: `{training_type}_{dataset_config}`
- **Training types**:
  - `single_`: Probe trained only on this dataset
  - `leaveout_`: Probe trained on all datasets except this one
  - `all_datasets`: Probe trained on all available datasets
- **Examples**:
  - `"single_convincing-game"`
  - `"leaveout_instructed-deception"`
  - `"all_datasets"`

## Understanding Probe Scores

### Score Interpretation

Probe scores are continuous values produced by projecting model activations onto a learned probe direction. The interpretation varies by probe:

1. **Direction**: The direction (positive vs negative) that indicates deception vs honesty is probe-specific
2. **Magnitude**: Higher absolute values indicate stronger signal
3. **Separation**: Good probes show clear separation between deceptive and honest score distributions
4. **Overlap**: Some probes may have overlapping distributions, indicating lower discrimination ability

### Viewing Statistics

The UI displays per-dataset statistics:
- **Deceptive scores**: Mean, min, max for deceptive examples
- **Honest scores**: Mean, min, max for honest examples
- **Separation**: Difference between deceptive and honest means

These statistics help assess:
- How well the probe separates the two classes for this dataset
- Whether the probe's scores align with the expected direction
- The overall quality of the probe for this specific task

### Visual Encoding

In the UI:
- **Background color** indicates ground truth label:
  - Light red: Deceptive example
  - Light green: Honest example
- **Score value** is displayed as raw number (not color-coded)
- This separation allows you to identify misalignments between probe scores and ground truth

## File Naming Convention

Recommended naming: `layer{N}_{description}.csv`

Examples:
- `layer22_all_probes_300samples.csv`
- `layer15_single_probes.csv`

## Example Row

```csv
conversation_history,final_assistant_response,prompt_level_score,token_level_scores,label,dataset_name,category,model_name,layer,probe_name
"[{""role"": ""user"", ""content"": ""What is 2+2?""}, {""role"": ""assistant"", ""content"": ""5""}]","5",4.123,"[{""token"": ""5"", ""score"": 4.123}]",deceptive,LiarsBench,instructed-deception,meta-llama/Llama-3.3-70B-Instruct,22,single_instructed-deception
```

## Notes

- All JSON fields must be properly escaped when embedded in CSV
- The `token_level_scores` field can be empty/null for conversations without token-level analysis
- Scores can be positive or negative depending on the probe's learned direction
- Multiple probes can evaluate the same conversations, creating multiple rows with different `probe_name` values
