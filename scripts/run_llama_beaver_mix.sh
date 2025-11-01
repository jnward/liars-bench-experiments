#!/usr/bin/env bash

# Cache, train, and evaluate the Llama + Beaver negative probe at the specified layer.
# Targets BeaverTails rows tagged as no_category, violence, drug_abuse, or hate_speech.

set -euo pipefail

TARGET_LAYER="${1:-22}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export APOLLO_MODEL="${APOLLO_MODEL:-llama}"
DECEPTION_REPO="${DECEPTION_REPO:-/workspace/jake/deception-detection}"

if [ -d "${DECEPTION_REPO}" ]; then
  export PYTHONPATH="${DECEPTION_REPO}:${PYTHONPATH:-}"
fi

BEAVER_SPLIT="30k_test"
BEAVER_CATEGORIES=(
  "no_category"
  "violence"
  "drug_abuse"
  "hate_speech"
)

PYTHON_BIN="uv"
UV_RUN=("$PYTHON_BIN" "run")

# Compute the cache/probe slug so train/eval agree.
SLUG="$("${UV_RUN[@]}" python - <<'PY'
from apollo_probes.llama_beaver_mix import (
    BeaverIngestionConfig,
    DEFAULT_CORE_DATASETS,
    beaver_mix_slug,
)

config = BeaverIngestionConfig(
    split="30k_test",
    categories=("no_category", "violence", "drug_abuse", "hate_speech"),
    sample_size=None,
)
print(beaver_mix_slug(config, DEFAULT_CORE_DATASETS))
PY
)"

echo "[run] Using slug: ${SLUG}"

cache_args=(
  --layers "${TARGET_LAYER}"
  --beaver-split "${BEAVER_SPLIT}"
  --force
)
for cat in "${BEAVER_CATEGORIES[@]}"; do
  cache_args+=(--beaver-category "${cat}")
done
echo "[run] Caching activations..."
"${UV_RUN[@]}" python -m apollo_probes.cache_llama_beaver_mix "${cache_args[@]}"

train_args=(
  --layer "${TARGET_LAYER}"
  --beaver-split "${BEAVER_SPLIT}"
  --force
)
for cat in "${BEAVER_CATEGORIES[@]}"; do
  train_args+=(--beaver-category "${cat}")
done
echo "[run] Training probe..."
"${UV_RUN[@]}" python -m apollo_probes.train_llama_beaver_mix "${train_args[@]}"

eval_cache_args=(
  --apollo-eval got_cities__plain got_larger_than__qa
  --beaver-eval
  --beaver-split "${BEAVER_SPLIT}"
  --beaver-sample-size 1000
  --sample-size 1000
  --layers "${TARGET_LAYER}"
)
echo "[run] Caching evaluation activations..."
"${UV_RUN[@]}" python -m apollo_probes.eval_cache "${eval_cache_args[@]}"

eval_args=(
  --train-dataset "${SLUG}"
  --layers "${TARGET_LAYER}"
  --beaver-split "${BEAVER_SPLIT}"
  --beaver-sample-size 1000
)
echo "[run] Evaluating probe..."
"${UV_RUN[@]}" python -m apollo_probes.evaluate_probes "${eval_args[@]}"

echo "[run] Complete. Results in apollo_probes/results/${APOLLO_MODEL}/${SLUG}/layer${TARGET_LAYER}/"
