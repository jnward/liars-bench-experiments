from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_ROOT = BASE_DIR / "cache"
EVAL_CACHE_ROOT = BASE_DIR / "eval_cache"
PROBES_ROOT = BASE_DIR / "probes"
RESULTS_ROOT = BASE_DIR / "results"

for _path in (CACHE_ROOT, EVAL_CACHE_ROOT, PROBES_ROOT, RESULTS_ROOT):
    _path.mkdir(parents=True, exist_ok=True)
