#!/usr/bin/env bash
set -euo pipefail

TRAIN_DATASET="repe_honesty__plain"
LAYER=22
PROBE_PATH="/workspace/jake/deception-detection/example_results/instructed_pairs/detector.pt"
PROBE_NAME="apollo_reference"

uv run python -m apollo_probes.eval_cache --layers $LAYER
uv run python -m apollo_probes.evaluate_probes \
  --train-dataset "$TRAIN_DATASET" \
  --layers $LAYER \
  --probe-path "$PROBE_PATH" \
  --probe-layer $LAYER \
  --probe-name "$PROBE_NAME"
