from __future__ import annotations

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_DATASET = "repe_honesty__plain"
VAL_FRACTION = 0.2
DEFAULT_BATCH_SIZE = 2
# Layers in Llama-3.3-70B are indexed 0-based (0-79, with 80 layers total)
DEFAULT_LAYER = 50
LAYER_START = 2
LAYER_STEP = 4
DEFAULT_LAYER_SWEEP = list(range(LAYER_START, 80, LAYER_STEP))
REG_COEFF = 10.0
RANDOM_SEED = 42
DEFAULT_APOLLO_EVALS = []
