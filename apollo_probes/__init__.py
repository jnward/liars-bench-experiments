from __future__ import annotations

from pathlib import Path

from .config import APOLLO_CONFIG

BASE_DIR = Path(__file__).resolve().parent
CACHE_ROOT = APOLLO_CONFIG.cache_root
EVAL_CACHE_ROOT = APOLLO_CONFIG.eval_cache_root
PROBES_ROOT = APOLLO_CONFIG.probes_root
RESULTS_ROOT = APOLLO_CONFIG.results_root

for _path in (CACHE_ROOT, EVAL_CACHE_ROOT, PROBES_ROOT, RESULTS_ROOT):
    _path.mkdir(parents=True, exist_ok=True)
