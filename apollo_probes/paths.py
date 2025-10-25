from __future__ import annotations

from pathlib import Path

from . import CACHE_ROOT, EVAL_CACHE_ROOT, PROBES_ROOT, RESULTS_ROOT


def dataset_cache_dir(dataset: str) -> Path:
    path = CACHE_ROOT / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def layer_cache_path(dataset: str, layer: int) -> Path:
    return dataset_cache_dir(dataset) / f"layer{layer:02d}.pt"


def eval_cache_dir(dataset: str) -> Path:
    path = EVAL_CACHE_ROOT / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def eval_layer_cache_path(dataset: str, layer: int) -> Path:
    return eval_cache_dir(dataset) / f"layer{layer:02d}.pt"


def dataset_probe_dir(dataset: str) -> Path:
    path = PROBES_ROOT / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_results_dir(dataset: str) -> Path:
    path = RESULTS_ROOT / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def probe_layer_dir(dataset: str, layer: int) -> Path:
    path = dataset_probe_dir(dataset) / f"layer{layer:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_layer_dir(dataset: str, layer: int) -> Path:
    path = dataset_results_dir(dataset) / f"layer{layer:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path
