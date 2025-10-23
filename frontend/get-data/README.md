# LiarsBench Data Sampling

This directory contains scripts for sampling and preparing datasets for probe visualization.

## Quick Start

```bash
# Activate virtual environment
source ../env/bin/activate

# Run the sampling script
python main.py
```

This will:
- Download all LiarsBench datasets from HuggingFace
- Sample 300 examples from each dataset (maintaining honest/deceptive ratios)
- Generate `sampled_liars_bench.csv` in the correct format for the Flask app

## Configuration

Edit the following variables in `main.py`:

- `SAMPLE_SIZE`: Number of samples per dataset (default: 300)
- `OUTPUT_CSV`: Output filename (default: "sampled_liars_bench.csv")
- `LIARS_BENCH_SUBSETS`: List of datasets to process
- `FILTER_MODEL_DEFAULT`: Model filter for most datasets
- `FILTER_MODEL_SOFT_TRIGGER`: Special model filter for soft-trigger dataset

## Output Format

The generated CSV follows the schema defined in `../CSV_SCHEMA.md`. See that file for complete documentation of the expected format.

**Key points:**
- 9 required columns in specific order
- `conversation_history` as JSON string
- `prompt_level_score` and `token_level_scores` left empty (for pre-probe data)
- Labels as strings: "honest" or "deceptive"

## Adding New Datasets

To add a new HuggingFace dataset:

1. Use the `load_generic_hf_dataset()` function for non-LiarsBench datasets
2. Convert data to the required CSV format (see `CSV_SCHEMA.md`)
3. Ensure all 9 columns are present with correct types

## Dependencies

Required packages (already in `../requirements.txt`):
- pandas
- numpy
- pyarrow (for parquet files)
- huggingface_hub

## Files

- `main.py`: Main sampling script with LiarsBench support
- `sampled_liars_bench.csv`: Generated output (not in git)
- `README.md`: This file

## Moving Data to Flask App

After generating the CSV, move it to the data directory:

```bash
cp sampled_liars_bench.csv ../data/
```

Then access it via the Flask app at `http://localhost:5000`
