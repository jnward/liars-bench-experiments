# SAE Feature Similarity Analysis

Compute cosine similarities between probe directions and Goodfire SAE decoder features for Llama 3.3 70B.

## Quick Start

```bash
python compute_sae_similarities.py --probe probes-layer50/single_convincing-game.pkl
```

That's it! One command does everything:
- ✅ Loads the probe direction
- ✅ Downloads SAE decoder weights from HuggingFace (cached after first run)
- ✅ Computes similarities for all 65,536 SAE features (~3 seconds)
- ✅ Automatically fetches labels from Goodfire API
- ✅ Saves top 10 positive AND top 10 negative features
- ✅ Saves complete distribution for plotting

## Output Files

For each probe, creates `outputs/probe-name/` directory with:

### `top_10_positive.json`
**Features most aligned with the probe** (highest positive cosine similarity)

```json
[
  {
    "feature_id": 36173,
    "similarity": 0.2961,
    "label": "Requests for the AI to provide intentionally incorrect information"
  },
  ...
]
```

### `top_10_negative.json`
**Features most anti-correlated with the probe** (most negative cosine similarity)

These represent behaviors opposite to what the probe detects.

### `all_similarities.json`
Complete list of all 65,536 features with similarity scores (5.7 MB)

### `similarity_distribution.npy`
NumPy array of all similarity values for distribution plotting

### `summary.json`
Metadata and statistics including both positive and negative top features

## Usage Examples

### Basic Usage
```bash
# Analyze any probe from probes-layer50/
python compute_sae_similarities.py --probe probes-layer50/apollo_probe.pkl
python compute_sae_similarities.py --probe probes-layer50/single_harm-pressure-choice.pkl
python compute_sae_similarities.py --probe probes-layer50/leaveout_convincing-game.pkl
```

### Skip Label Fetching (faster)
```bash
python compute_sae_similarities.py --probe probes-layer50/single_convincing-game.pkl --skip-labels
```

### Custom Output Directory
```bash
python compute_sae_similarities.py \
  --probe probes-layer50/single_convincing-game.pkl \
  --output-dir my_results
```

## Understanding the Results

### Positive vs Negative Similarity

**Positive similarity (+)**: SAE feature aligns with probe direction
- Example: Feature 36173 (+0.296) "intentionally incorrect information" aligns with deception detection

**Negative similarity (-)**: SAE feature is anti-correlated with probe
- These represent behaviors *opposite* to what the probe detects
- Example: Feature 17368 (-0.195) "AI explaining its limitations" is opposite of deception

### Magnitude

- Typical range: -0.3 to +0.3
- Values > +0.20: Strong alignment
- Values < -0.20: Strong anti-correlation

### Label Availability

The script automatically attempts to fetch labels via Goodfire API, but labels are only available for features that **activate** in test conversations.

Features without labels will show:
```
"Feature 15424 (label not available - check Goodfire dashboard)"
```

You can look up any feature by index at the Goodfire dashboard.

## Example Results

For `single_convincing-game.pkl`:

**Top 3 Positive (Aligned with Deception):**
1. Feature 36173 (+0.296): "Intentionally incorrect information"
2. Feature 42834 (+0.197): "Whimsical or nonsensical narrative"
3. Feature 41630 (+0.183): Label not available

**Top 3 Negative (Anti-correlated with Deception):**
1. Feature 15424 (-0.245): Label not available
2. Feature 26971 (-0.240): Label not available
3. Feature 21570 (-0.231): Label not available

Statistics:
- Mean similarity: 0.002 (centered near zero)
- Std deviation: 0.025
- Range: -0.245 to +0.296

## Creating Distribution Plots

```python
import numpy as np
import matplotlib.pyplot as plt

# Load similarity distribution
sims = np.load('outputs/single_convincing-game/similarity_distribution.npy')

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(sims, bins=100, alpha=0.7, edgecolor='black')
plt.xlabel('Cosine Similarity', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('SAE Feature Similarity Distribution', fontsize=14)
plt.axvline(0, color='r', linestyle='--', alpha=0.5, label='Zero')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('outputs/single_convincing-game/distribution.png', dpi=300)
```

## Requirements

```bash
pip install torch numpy goodfire huggingface_hub python-dotenv tqdm scikit-learn
```

## Environment Setup

Create a `.env` file:
```bash
GOODFIRE_API_KEY=sk-goodfire-your-key-here
HF_HOME=/path/to/cache/huggingface
HF_TOKEN=hf_your-token-here
```

## How It Works

1. **Load Probe**: Extracts logistic regression coefficients (shape: [8192])
2. **Load SAE Decoder**: Downloads from HuggingFace (shape: [8192, 65536])
3. **Compute Similarities**: Cosine similarity between probe and each of 65,536 decoder features
4. **Fetch Labels**: Attempts to get labels for top 20 features via Goodfire API
5. **Save Results**: Outputs top positive, top negative, full distribution, and summary

## Probe File Formats Supported

**sklearn probes** (from `probe_pipeline/`):
```python
{"model": LogisticRegression(...), "datasets": [...], "config": {...}}
```

**torch probes** (from `apollo_probes/`):
```python
{"directions": tensor([...]), "layers": [50], "dataset": "...", ...}
```

## Troubleshooting

### "Invalid magic number" error
The probe file might be corrupted. Ensure it's a valid pickle file.

### "GOODFIRE_API_KEY not found"
Create a `.env` file with your API key or set environment variable:
```bash
export GOODFIRE_API_KEY=sk-goodfire-...
```

### Out of memory
The SAE model is ~3GB. Close other applications if needed.

### Missing labels
Labels are only available for features that activate in test conversations. Features with unavailable labels can be looked up manually on the Goodfire dashboard.

## Performance Notes

- **First run**: Downloads ~3GB SAE model (cached afterward)
- **Computing similarities**: ~3-4 seconds for all 65,536 features
- **Label fetching**: ~5-10 seconds (tries 4 different conversations)
- **Total runtime**: ~10-15 seconds (after SAE is cached)

## Files in This Directory

- `compute_sae_similarities.py` - Main script (consolidated, all-in-one)
- `probes-layer50/` - All available probe files (16 probes)
- `outputs/` - Results directory (created automatically)
- `.env` - Your API keys (create this)

## Related Analysis

- `sae-decoder-feature-probes/README.md` - Comprehensive analysis of all probes with detailed findings
- `sae-L50-decoder-features-labels/` - Pre-computed rankings and some labels
