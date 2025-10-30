# Probe Pipeline

`probe_pipeline` provides reusable scripts to cache activations, train logistic probes, and plot results across multiple backbones. It currently supports:

- `llama` (default) – `meta-llama/Llama-3.3-70B-Instruct`
- `gemma` – `google/gemma-3-27b-it`
- `qwen` – `Qwen/Qwen2.5-72B-Instruct`

Pick a model by setting `PROBE_MODEL` (or reuse `APOLLO_MODEL`). Artifacts are segregated under:

- `cache/<model>/…`
- `probes/<model>/…`
- `plots/<model>/…`
- `results/<model>/…`

## Quick start

```bash
# Cache activations (single layer 22, Llama)
uv run python probe_pipeline/cache_activations.py --dataset instructed-deception --layer 22

# Train/evaluate probes for cached datasets
uv run python probe_pipeline/sweep_probes.py --layer 22

# Multi-layer sweep
uv run python probe_pipeline/sweep_layers.py
```

For non-default models:

```bash
# Gemma smoke sweep (layers 18 and 30)
PROBE_MODEL=gemma_test LAYER_LIST=18,30 uv run python probe_pipeline/sweep_layers.py

# Qwen full sweep (range-based)
PROBE_MODEL=qwen uv run python probe_pipeline/sweep_layers.py --layer-start 2 --layer-step 4 --layer-end 50
```

## Scripts

- `cache_activations.py` – loads Liars’ Bench data, extracts last-assistant activations, and stores train/eval splits under `cache/<model>/…`. Key options: `--dataset`, `--layers`, `--layer-start`, `--layer-step`, `--batch-size`, `--force`.
- `sweep_probes.py` – trains logistic probes for each cached dataset combination (`single`, `leaveout`, `all`) and writes ROC plots + JSON metrics. Accepts `--layer` and respects cached files.
- `sweep_layers.py` – orchestrates layer-by-layer caching + training. Supports explicit layers via `--layers` or `LAYER_LIST`, and propagates `PROBE_MODEL` to subprocesses.
- `evaluate_probe.py`, `train_logreg.py`, `falsehood/*`, `diff_probes/*` – additional utilities for bespoke experiments (difference probes, falsehood datasets, etc.).

## Custom layer selections

You can specify exact layers in several ways:

- CLI flag: `--layers 18 22 30`
- Environment: `LAYER_LIST=18,22,30`
- Range: `--layer-start 2 --layer-step 4 --layer-end 50`

`sweep_layers.py` passes these selections to the caching/training scripts automatically.

## Dataset filtering

`cache_activations.py` loads `Cadenza-Labs/liars-bench` and filters rows to the config’s generator model (`llama-v3.3-70b-instruct`, `gemma-3-27b-it`, `qwen-2.5-72b-instruct`). To limit the number of rows, set `SAMPLE_SIZE` in the environment:

```bash
SAMPLE_SIZE=500 PROBE_MODEL=gemma uv run python probe_pipeline/cache_activations.py --dataset instructed-deception --layer 22
```

## Notes

- Mixed precision has been removed; forward passes run in the model’s native dtype to avoid NaNs at deeper layers.
- `model_configs.py` is the central registry—add entries there to introduce new backbones or tweak defaults (cache directories, layer ranges, batch sizes).
- `PROBE_MODEL` and `APOLLO_MODEL` can be used interchangeably so Apollo and pipeline scripts share the same configuration.
