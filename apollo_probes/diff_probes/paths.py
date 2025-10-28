from __future__ import annotations

from pathlib import Path

from .. import CACHE_ROOT, PROBES_ROOT, RESULTS_ROOT


def diff_cache_dir(dataset_slug: str) -> Path:
    path = CACHE_ROOT / "diff_probes" / dataset_slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def diff_cache_path(dataset_slug: str, layer: int) -> Path:
    return diff_cache_dir(dataset_slug) / f"layer{layer:02d}.pt"


def diff_probe_dir(dataset_slug: str, layer: int) -> Path:
    path = PROBES_ROOT / "diff_probes" / dataset_slug / f"layer{layer:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def diff_results_dir(dataset_slug: str, layer: int) -> Path:
    path = RESULTS_ROOT / "diff_probes" / dataset_slug / f"layer{layer:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path

