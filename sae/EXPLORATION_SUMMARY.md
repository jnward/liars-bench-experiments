# Comprehensive Exploration of /workspace/alex/liars-bench-experiments/sae

## Executive Summary

The `/workspace/alex/liars-bench-experiments/sae` directory contains a sophisticated system for analyzing Sparse Autoencoder (SAE) features from Goodfire's Llama-3.3-70B model and building semantic probes for deception detection. The workflow consists of:

1. **Semantic Feature Discovery**: Finding interpretable SAE features related to deception, refusal, harm, and other behaviors
2. **Probe Construction**: Building synthetic probes from selected SAE decoder features
3. **Comparative Analysis**: Computing cosine similarities between probes and all 65,536 SAE features
4. **Aggregation & Evaluation**: Combining results across multiple datasets and generating visualizations

---

## Directory Structure Overview

### Root Level Organization

```
/workspace/alex/liars-bench-experiments/sae/
├── Python Scripts (1938 LOC total)
│   ├── compute_sae_similarities.py       (346 lines) - Core similarity computation
│   ├── build_semantic_probes.py          (201 lines) - Build semantic probes from CSV
│   ├── build_synthetic_probe.py          (324 lines) - Build probes from SAE features
│   ├── build_category_probes.py          (246 lines) - Build category-based probes
│   ├── build_random_baseline.py          (135 lines) - Generate random baseline probes
│   ├── aggregate_results.py              (208 lines) - Combine results across probes
│   ├── search_confidence_features.py     (109 lines) - Search for confidence features
│   ├── append_confidence_to_csv.py       (63 lines)  - Append to training data
│   └── generate_index_html.py            (306 lines) - Generate HTML visualization
├── Configuration
│   ├── .env                              - API keys and cache paths
│   └── run_all_probes.sh                 - Batch execution script
├── Input Data
│   ├── semantic_probe_results_v3/        - CSV with 245+ labeled features across 7 categories
│   ├── distinctive_features/             - Feature ranking CSVs by dataset
│   ├── honest_vs_deceptive_features/     - Comparative feature lists
│   └── probes-layer50/                   - 16 pre-computed probe files (sklearn format)
├── Output/Results (139 probe result directories)
│   ├── outputs/
│   │   ├── semantic_probes/              - 15 semantic probes (baseline + loo + single)
│   │   ├── random_baseline/              - 100 random baseline probes
│   │   ├── categories/synthetic_probes/  - 11 category-based probes
│   │   ├── combined/                     - Aggregated results from all probes
│   │   ├── single_*.../                  - 6 single-dataset results
│   │   ├── leaveout_*.../                - 5 leave-one-out dataset results
│   │   └── all_datasets/                 - Combined all datasets results
│   └── sae-decoder-feature-probes/       - Detailed probe reconstruction analysis
└── Documentation
    ├── README_SAE_SIMILARITIES.md        - Usage guide and examples
    └── sae-decoder-feature-probes/README.md - Detailed experimental results
```

---

## 1. Input Data & Feature Discovery

### Source Data Structure

**Primary Input File**: `semantic_probe_results_v3/label_validation_sample.csv` (30.5 KB, 245+ rows)

**7 Semantic Categories** (245+ features total):
1. **refusal** (~35 features) - Safety boundaries, limitations, policy enforcement
2. **evasion** (~45 features) - Deflection, strategic avoidance, indirect responses
3. **harm** (~40 features) - Harmful content, discrimination, inappropriate behavior
4. **lying** (~40 features) - Deception, falsehoods, misleading statements
5. **instruction-following** (~35 features) - Task completion, role adherence
6. **roleplay** (~20 features) - Character adoption, narrative content
7. **confidence** (~30 features) - Recently added via `search_confidence_features.py`

---

## 2. Main Python Scripts & Workflows

### Script 1: compute_sae_similarities.py (346 lines) - CORE ANALYSIS

**Purpose**: Computes cosine similarities between a probe and all 65,536 SAE features

**Process**:
1. Load probe (sklearn LogisticRegression coefficients) - shape: [8192]
2. Download SAE decoder from HuggingFace (cached) - shape: [8192, 65536]
3. Compute cosine similarity for each of 65,536 features (~3 seconds)
4. Fetch labels from Goodfire API for top 20 features
5. Save results

**Output** (`outputs/{probe_name}/`):
- `top_10_positive.json` - Most aligned features (highest similarity)
- `top_10_negative.json` - Most anti-correlated features
- `all_similarities.json` - All 65,536 features with scores (5.7 MB)
- `similarity_distribution.npy` - NumPy array for visualization
- `summary.json` - Metadata, statistics, and top features

### Script 2: build_semantic_probes.py (201 lines) - BUILDS 15 SEMANTIC PROBES

**Purpose**: Build 15 semantic probes from feature categories in CSV

**Creates**:
1. **1 Baseline Probe** (`all_categories.pkl`) - All 245 features
2. **7 Leave-One-Out Probes** - Remove one category each
3. **7 Single-Category Probes** - Use only one category

**Construction Method**:
```python
# For each configuration:
feature_ids = [selected features]
decoder_weights = load_from_hf()  # [8192, 65536]
probe = sum(decoder_weights[:, feat_id] for feat_id in feature_ids)
probe_normalized = probe / ||probe||  # Unit vector
```

**Key Characteristics**:
- Uniform weighting (all features weighted 1.0)
- Normalized to unit length
- Saved as torch format with metadata
- No learned weights from data

### Script 3: build_synthetic_probe.py (324 lines) - BUILD FROM TOP FEATURES

**Purpose**: Build synthetic probes from top SAE features using cosine similarity weights

**Usage**:
```bash
python build_synthetic_probe.py --probe single_convincing-game --top-k 10
```

**Construction**:
```python
# Weight each feature by its cosine similarity
probe = sum(similarity_i * decoder_feature[feat_id_i] for i in range(top_k))
probe_normalized = probe / ||probe||
```

### Script 4: build_category_probes.py (246 lines)

**Purpose**: Build 11 hardcoded category-based probes (different from semantic probes)

**Creates**: 
- 1 probe with all 10 features
- 5 ablation probes (remove one category each)
- 5 single-category probes

### Script 5: build_random_baseline.py (135 lines)

**Purpose**: Generate 100 random baseline probes for statistical comparison

**Method**: Sample random vectors from unit hypersphere

### Script 6: aggregate_results.py (208 lines)

**Purpose**: Combine similarity results from all probes

**Outputs**:
- Top 10 highest positive similarities
- Top 10 most frequent positive features
- Top 10 lowest (most negative) similarities
- Top 10 most frequent negative features

### Script 7: search_confidence_features.py (109 lines)

**Purpose**: Find new confidence-related features using Goodfire API

### Script 8: generate_index_html.py (306 lines)

**Purpose**: Create interactive HTML visualization

---

## 3. Output/Results Structure

### semantic_probes/ (15 probes)

Files built from CSV categories:
- `all_categories.pkl` (35 KB) - 245 features
- `no_*.pkl` (7 files) - Leave-one-out
- `only_*.pkl` (7 files) - Single-category
- Corresponding `*_metadata.json` files

### random_baseline/ (100 probes)

- `random_000.pkl` through `random_099.pkl`
- `summary.json` with metadata

### categories/synthetic_probes/ (11 probes)

Hardcoded category-based probes

### single_*/ Results (6 datasets + leaveout variants)

Per-probe analysis:
- `top_10_positive.json` (1.5 KB)
- `top_10_negative.json` (1.5 KB)
- `all_similarities.json` (5.7 MB)
- `similarity_distribution.npy` (513 KB)
- `summary.json` (4.1 KB)

### combined/ Results

Aggregated analyses across all probes

---

## 4. Data Formats

### Probe File Format (.pkl)

**PyTorch Format** (semantic/category/random probes):
```python
{
    "direction": torch.tensor([...], dtype=float32),  # Shape: [8192]
    "metadata": {
        "probe_name": "all_categories",
        "num_features": 245,
        "feature_ids": [788, 34568, ...],
        "weight_scheme": "uniform (1.0)",
        "pre_norm_magnitude": 27.156,
        "categories_included": ["confidence", "evasion", ...]
    }
}
```

### Result JSON Formats

**top_10_positive.json / top_10_negative.json**:
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

---

## 5. Configuration & Dependencies

### .env File
```
HF_TOKEN=...
HF_HOME=/root/.cache/huggingface
GOODFIRE_API_KEY=...
```

### Python Requirements
- torch, numpy, pandas, scikit-learn
- goodfire, huggingface_hub, python-dotenv, tqdm

### External Services
- **Goodfire SAE**: `Goodfire/Llama-3.3-70B-Instruct-SAE-l50` (~3GB)
- **Goodfire API**: Feature labels and search

---

## 6. How Semantic Probes Are Currently Built

### Complete Pipeline

```
1. CSV Input (semantic_probe_results_v3/label_validation_sample.csv)
   └─ 245+ features across 7 categories

2. build_semantic_probes.py
   └─ Creates 15 probes with:
      - Uniform weighting (1.0 per feature)
      - Summation: Σ(decoder[:, feat_id])
      - Normalization: v / ||v||
      - Saved as torch format

3. compute_sae_similarities.py
   └─ For each probe:
      - Load probe [8192]
      - Load SAE decoder [8192 x 65536]
      - Compute cosine similarity with all features
      - Fetch labels from Goodfire API
      - Save results

4. build_synthetic_probe.py (Optional)
   └─ Create new probes from top features with similarity weights

5. aggregate_results.py
   └─ Find universal features across all probes

6. generate_index_html.py
   └─ Create interactive visualization
```

### Key Design Decisions

- **Uniform weighting**: Simple, interpretable, reproducible
- **Decoder features**: Direct supervision signal, interpretable labels
- **Layer 50**: Pre-selected for analysis
- **245 features**: Manually curated from Goodfire API searches

---

## 7. Key Insights from Results

### Top Features

- **Feature 36173**: "Requests for intentionally incorrect information" (+0.296)
- **Feature 42834**: "Whimsical or nonsensical narrative" (+0.197)
- **Feature 27783**: "Expressing limitations or inabilities" (+0.151)

### Distribution

- Mean similarity: ~0.002 (centered near zero)
- Std deviation: ~0.025
- Range: -0.245 to +0.2961
- Most features weakly align with probes

---

## 8. Files & Data Summary

### Inputs
- CSV: 1 file (30 KB)
- Probe files: 16 files (~1 MB)
- Feature ranking CSVs: 20+ files

### Outputs
- Result directories: 139 total
- JSON files: ~400+
- .npy arrays: ~60+ (5.7 MB each)
- HTML: 1 index.html

### Total Size
- Input: ~500 MB
- Output: ~20+ GB
- Cache: ~3 GB (SAE model)

---

## 9. Workflow Summary

### Quick Usage

```bash
# Step 1: Build semantic probes
python build_semantic_probes.py

# Step 2: Compute similarities
python compute_sae_similarities.py --probe outputs/semantic_probes/all_categories.pkl

# Step 3: Build synthetic probes (optional)
python build_synthetic_probe.py --probe semantic_probes/all_categories --top-k 10

# Step 4: Aggregate
python aggregate_results.py

# Step 5: Visualize
python generate_index_html.py

# Or batch process all probes:
./run_all_probes.sh
```

---

## 10. Related Components

### Parent Project Connection
- Probes come from `../probe_pipeline/`
- Used as baselines for comparison
- Related to `../apollo_probes/` and `../black-box/`

### External Integrations
- Goodfire API for feature discovery and labels
- HuggingFace for SAE model and caching
- Goodfire SAE (L50) for interpretable features

---

For detailed information, see the main summary document and quick reference guide.
