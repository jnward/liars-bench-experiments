from __future__ import annotations

import os

from .model_configs import get_model_config


APOLLO_CONFIG = get_model_config(os.environ.get("APOLLO_MODEL") or os.environ.get("PROBE_MODEL"))

MODEL_NAME = APOLLO_CONFIG.model_name
DEFAULT_DATASET = APOLLO_CONFIG.default_dataset
EVAL_FILTER_MODEL = APOLLO_CONFIG.eval_filter_model
VAL_FRACTION = APOLLO_CONFIG.val_fraction
DEFAULT_BATCH_SIZE = APOLLO_CONFIG.default_batch_size
DEFAULT_LAYER = APOLLO_CONFIG.default_layer
LAYER_START = APOLLO_CONFIG.layer_start
LAYER_STEP = APOLLO_CONFIG.layer_step
DEFAULT_LAYER_SWEEP = list(APOLLO_CONFIG.default_layer_sweep)
REG_COEFF = APOLLO_CONFIG.reg_coeff
RANDOM_SEED = APOLLO_CONFIG.random_seed
DEFAULT_APOLLO_EVALS = [
    "got_cities__plain",
    "got_larger_than__qa",
]
