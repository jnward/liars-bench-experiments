from __future__ import annotations

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_DATASET = "repe_honesty__plain"
VAL_FRACTION = 0.2
DEFAULT_BATCH_SIZE = 2
# Layers in Llama-3.3 are indexed 0-based; our sweeps usually start at 2 and step by 4.
DEFAULT_LAYER = 22
LAYER_START = 2
LAYER_STEP = 4
DEFAULT_LAYER_SWEEP = list(range(LAYER_START, 80, LAYER_STEP))
REG_COEFF = 10.0
RANDOM_SEED = 42
