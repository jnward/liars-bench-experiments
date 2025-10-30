from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelConfig:
    key: str
    model_name: str
    eval_filter_model: str
    default_dataset: str
    val_fraction: float
    default_batch_size: int
    default_layer: int
    layer_start: int
    layer_step: int
    default_layer_sweep: list[int]
    reg_coeff: float
    random_seed: int
    cache_root: Path
    eval_cache_root: Path
    probes_root: Path
    results_root: Path


def layers_from_range(start: int, step: int, max_layer: int) -> list[int]:
    return [layer for layer in range(start, max_layer + 1, step)]


LLAMA_MAX_LAYER = 80  # Llama-3.3-70B-Instruct has 80 transformer blocks.

LLAMA_CONFIG = ModelConfig(
    key="llama",
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    eval_filter_model="llama-v3.3-70b-instruct",
    default_dataset="repe_honesty__plain",
    val_fraction=0.2,
    default_batch_size=2,
    default_layer=22,
    layer_start=2,
    layer_step=4,
    default_layer_sweep=layers_from_range(2, 4, LLAMA_MAX_LAYER),
    reg_coeff=10.0,
    random_seed=42,
    cache_root=BASE_DIR / "cache" / "llama",
    eval_cache_root=BASE_DIR / "eval_cache" / "llama",
    probes_root=BASE_DIR / "probes" / "llama",
    results_root=BASE_DIR / "results" / "llama",
)


QWEN_CONFIG = ModelConfig(
    key="qwen",
    model_name="Qwen/Qwen2.5-72B-Instruct",
    eval_filter_model="qwen-2.5-72b-instruct",
    default_dataset="repe_honesty__plain",
    val_fraction=0.2,
    default_batch_size=2,
    default_layer=22,
    layer_start=2,
    layer_step=4,
    default_layer_sweep=layers_from_range(2, 4, 79),  # Qwen 72B has 80 layers.
    reg_coeff=10.0,
    random_seed=42,
    cache_root=BASE_DIR / "cache" / "qwen",
    eval_cache_root=BASE_DIR / "eval_cache" / "qwen",
    probes_root=BASE_DIR / "probes" / "qwen",
    results_root=BASE_DIR / "results" / "qwen",
)


MODEL_CONFIGS: dict[str, ModelConfig] = {
    LLAMA_CONFIG.key: LLAMA_CONFIG,
    QWEN_CONFIG.key: QWEN_CONFIG,
}


def get_model_config(key: str | None = None) -> ModelConfig:
    if key is None:
        key = os.environ.get("APOLLO_MODEL") or os.environ.get("PROBE_MODEL") or "llama"
    normalized = key.lower()
    if normalized not in MODEL_CONFIGS:
        available = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown Apollo model '{key}'. Available: {available}")
    return MODEL_CONFIGS[normalized]


__all__ = ["ModelConfig", "MODEL_CONFIGS", "get_model_config"]
