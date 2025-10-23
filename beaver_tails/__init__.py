"""BeaverTails probe experimentation utilities."""

from __future__ import annotations

__all__ = [
    "CACHE_DIR",
    "SCORES_DIR",
    "PLOTS_DIR",
]

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
SCORES_DIR = BASE_DIR / "scores"
PLOTS_DIR = BASE_DIR / "plots"

for _path in (CACHE_DIR, SCORES_DIR, PLOTS_DIR):
    _path.mkdir(parents=True, exist_ok=True)
