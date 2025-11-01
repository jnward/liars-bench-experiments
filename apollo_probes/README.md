# Apollo Probes

The `apollo_probes` package trains and evaluates deception-detection probes on dialogue data. It shares caching and training logic across multiple backbone models (Llama 3.3 70B, Qwen 2.5 72B) while keeping their artifacts separate.

## Model selection

Two configurations ship by default:

- `llama` (default): `meta-llama/Llama-3.3-70B-Instruct`
- `qwen`: `Qwen/Qwen2.5-72B-Instruct`

Select a model by setting `APOLLO_MODEL` (or `PROBE_MODEL`) before running any scripts. Example:

```bash
# Default Llama
uv run python apollo_probes/sweep_layers.py --layers 22

# Qwen
APOLLO_MODEL=qwen uv run python apollo_probes/sweep_layers.py --layers 18
```

Model-specific outputs live under:

- `cache/<model>/…`
- `probes/<model>/…`
- `results/<model>/…`
- `eval_cache/<model>/…`

## Dataset names

Datasets are defined in `apollo_probes/data.py`. Pass either individual names or a `+`-joined combination:

```bash
# Single dataset (default)
uv run python apollo_probes/sweep_layers.py

# Combined dataset
uv run python apollo_probes/sweep_layers.py --dataset repe_honesty__plain+roleplaying__plain --layers 18
```

Rock-playing data is fetched from model-specific rollouts (`llama-70b-3.3` for Llama, `qwen-2.5-72b-instruct` for Qwen), so the same dataset selection works across models.

## Llama + Beaver negative mix

When exploring probes that should stay quiet on benign completions, you can augment the standard Llama training data with BeaverTails rows that are forced to label `honest` (0). The helper scripts are:

- `apollo_probes/cache_llama_beaver_mix.py`
- `apollo_probes/train_llama_beaver_mix.py`

By default the cache step mixes `repe_honesty__plain` and `roleplaying__plain` with BeaverTails rows from the `30k_test` split where the dataset tags include:

- `no_category` (no explicit harm tag)
- `violence` (e.g. `violence,aiding_and_abetting,incitement`)
- `drug_abuse`
- `hate_speech`

All Beaver examples are assigned label 0 before shuffling/splitting. Example workflow (default Llama config):

```bash
# Cache layer 22 activations for the mix (force rebuild if artifacts exist)
uv run python apollo_probes/cache_llama_beaver_mix.py --layers 22 --force

# Train the probe (layers must already be cached)
uv run python apollo_probes/train_llama_beaver_mix.py --layer 22
```

Key options:

- `--beaver-category CATEGORY` (repeatable) – overwrite the default category set.
- `--beaver-split SPLIT` – choose a different BeaverTails split.
- `--beaver-sample-size N` – subsample Beaver rows deterministically.
- `--core-datasets …` – replace the Apollo datasets mixed in (slug names are encoded into the cache/probe directory automatically).

## Layer sweeps

`apollo_probes/sweep_layers.py` caches activations and trains probes across layers. Useful flags:

- `--layers L1 L2 …` – explicit layer indices (overrides `--layer-start` / `--layer-step`).
- `--layer-start`, `--layer-step`, `--layer-end` – range-based sweep parameters.
- `--batch-size` – caching batch size (defaults to the config’s recommendation).
- `--force-cache`, `--force-train` – rebuild activations or retrain probes even if artifacts exist.

Environment overrides:

- `LAYER_LIST` / `LAYERS` – comma-separated layer indices (e.g., `LAYER_LIST=18,22`).

## Regenerating caches manually

Use `apollo_probes/cache_activations.py` to cache specific layers:

```bash
APOLLO_MODEL=llama uv run python apollo_probes/cache_activations.py \
  --dataset repe_honesty__plain \
  --layers 18 22 \
  --batch-size 2
```

## Probe training and evaluation

Once activations are cached, `apollo_probes/train_probe.py` fits a logistic probe for a single layer:

```bash
APOLLO_MODEL=qwen uv run python apollo_probes/train_probe.py \
  --dataset repe_honesty__plain \
  --layer 18
```

Outputs are saved to the model-specific `probes/` and `results/` directories. Evaluation CLI utilities (`evaluate_probes.py`, notebooks under `apollo_probes/notebooks/`) also respect `APOLLO_MODEL`.

## Notes

- Mixed precision (`torch.cuda.amp.autocast`) has been removed to avoid NaNs in deeper layers. For bf16-capable hardware, the model runs in its native dtype automatically.
- `PROBE_MODEL` remains supported as a legacy alias for `APOLLO_MODEL`.
