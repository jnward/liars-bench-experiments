from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelConfig:
    key: str
    model_name: str
    filter_model_name: str
    dataset_configs: list[str]
    cache_dir: Path
    probes_dir: Path
    plots_dir: Path
    results_dir: Path
    layer_start: int
    layer_step: int
    default_batch_size: int
    default_sample_size: int | None = 1000
    assistant_header_override: str | None = None
    tokenizer_padding_side: str | None = "left"
    apollo_probe_path: Path | None = None
    layer_count_path: tuple[str, ...] | None = None


LLAMA_CONFIG = ModelConfig(
    key="llama",
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    filter_model_name="llama-v3.3-70b-instruct",
    dataset_configs=[
        "convincing-game",
        "harm-pressure-choice",
        "harm-pressure-knowledge-report",
        "insider-trading/report",
        "insider-trading/confirmation",
        "instructed-deception",
    ],
    cache_dir=PROJECT_ROOT / "probe_pipeline" / "cache_llama",
    probes_dir=PROJECT_ROOT / "probe_pipeline" / "probes_llama",
    plots_dir=PROJECT_ROOT / "probe_pipeline" / "plots_llama",
    results_dir=PROJECT_ROOT / "probe_pipeline" / "results_llama",
    layer_start=2,
    layer_step=4,
    default_batch_size=2,
    default_sample_size=1000,
    assistant_header_override="<|start_header_id|>assistant<|end_header_id|>",
    tokenizer_padding_side="left",
    apollo_probe_path=None,
    layer_count_path=("num_hidden_layers",),
)

LLAMA_TEST_CONFIG = ModelConfig(
    key="llama_test",
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    filter_model_name="llama-v3.3-70b-instruct",
    dataset_configs=[
        "convincing-game",
        "harm-pressure-choice",
        "harm-pressure-knowledge-report",
        "insider-trading/report",
        "insider-trading/confirmation",
        "instructed-deception",
    ],
    cache_dir=PROJECT_ROOT / "probe_pipeline" / "cache_llama_test",
    probes_dir=PROJECT_ROOT / "probe_pipeline" / "probes_llama_test",
    plots_dir=PROJECT_ROOT / "probe_pipeline" / "plots_llama_test",
    results_dir=PROJECT_ROOT / "probe_pipeline" / "results_llama_test",
    layer_start=34,
    layer_step=100,
    default_batch_size=2,
    default_sample_size=200,
    assistant_header_override="<|start_header_id|>assistant<|end_header_id|>",
    tokenizer_padding_side="left",
    apollo_probe_path=None,
    layer_count_path=("num_hidden_layers",),
)


GEMMA_CONFIG = ModelConfig(
    key="gemma",
    model_name="google/gemma-3-27b-it",
    filter_model_name="gemma-3-27b-it",
    dataset_configs=LLAMA_CONFIG.dataset_configs,
    cache_dir=PROJECT_ROOT / "probe_pipeline" / "cache_gemma",
    probes_dir=PROJECT_ROOT / "probe_pipeline" / "probes_gemma",
    plots_dir=PROJECT_ROOT / "probe_pipeline" / "plots_gemma",
    results_dir=PROJECT_ROOT / "probe_pipeline" / "results_gemma",
    layer_start=2,
    layer_step=3,
    default_batch_size=2,
    default_sample_size=1000,
    assistant_header_override="<start_of_turn>model",
    tokenizer_padding_side="left",
    apollo_probe_path=None,
    layer_count_path=("text_config", "num_hidden_layers"),
)

GEMMA_TEST_CONFIG = ModelConfig(
    key="gemma_test",
    model_name="google/gemma-3-27b-it",
    filter_model_name="gemma-3-27b-it",
    dataset_configs=LLAMA_CONFIG.dataset_configs,
    cache_dir=PROJECT_ROOT / "probe_pipeline" / "cache_gemma_test",
    probes_dir=PROJECT_ROOT / "probe_pipeline" / "probes_gemma_test",
    plots_dir=PROJECT_ROOT / "probe_pipeline" / "plots_gemma_test",
    results_dir=PROJECT_ROOT / "probe_pipeline" / "results_gemma_test",
    layer_start=30,
    layer_step=60, # only train one layer
    default_batch_size=2,
    default_sample_size=200,
    assistant_header_override="<start_of_turn>model",
    tokenizer_padding_side="left",
    apollo_probe_path=None,
    layer_count_path=("text_config", "num_hidden_layers"),
)

QWEN_CONFIG = ModelConfig(
    key="qwen",
    model_name="Qwen/Qwen2.5-72B-Instruct",
    filter_model_name="qwen-2.5-72b-instruct",
    dataset_configs=LLAMA_CONFIG.dataset_configs,
    cache_dir=PROJECT_ROOT / "probe_pipeline" / "cache_qwen",
    probes_dir=PROJECT_ROOT / "probe_pipeline" / "probes_qwen",
    plots_dir=PROJECT_ROOT / "probe_pipeline" / "plots_qwen",
    results_dir=PROJECT_ROOT / "probe_pipeline" / "results_qwen",
    layer_start=2,
    layer_step=4,
    default_batch_size=2,
    default_sample_size=1000,
    assistant_header_override="<|im_start|>assistant",
    tokenizer_padding_side="left",
    apollo_probe_path=None,
    layer_count_path=("num_hidden_layers",),
)

QWEN_TEST_CONFIG = ModelConfig(
    key="qwen_test",
    model_name="Qwen/Qwen2.5-72B-Instruct",
    filter_model_name="qwen-2.5-72b-instruct",
    dataset_configs=LLAMA_CONFIG.dataset_configs,
    cache_dir=PROJECT_ROOT / "probe_pipeline" / "cache_qwen_test",
    probes_dir=PROJECT_ROOT / "probe_pipeline" / "probes_qwen_test",
    plots_dir=PROJECT_ROOT / "probe_pipeline" / "plots_qwen_test",
    results_dir=PROJECT_ROOT / "probe_pipeline" / "results_qwen_test",
    layer_start=22,
    layer_step=100,
    default_batch_size=2,
    default_sample_size=200,
    assistant_header_override="<|im_start|>assistant",
    tokenizer_padding_side="left",
    apollo_probe_path=None,
    layer_count_path=("num_hidden_layers",),
)


MODEL_CONFIGS: dict[str, ModelConfig] = {
    LLAMA_CONFIG.key: LLAMA_CONFIG,
    LLAMA_TEST_CONFIG.key: LLAMA_TEST_CONFIG,
    GEMMA_CONFIG.key: GEMMA_CONFIG,
    GEMMA_TEST_CONFIG.key: GEMMA_TEST_CONFIG,
    QWEN_CONFIG.key: QWEN_CONFIG,
    QWEN_TEST_CONFIG.key: QWEN_TEST_CONFIG,
}


def get_model_config(key: str | None = None) -> ModelConfig:
    if key is None:
        key = os.environ.get("PROBE_MODEL", "llama")
    normalized = key.lower()
    if normalized not in MODEL_CONFIGS:
        available = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown probe model '{key}'. Available: {available}")
    return MODEL_CONFIGS[normalized]


__all__ = [
    "ModelConfig",
    "MODEL_CONFIGS",
    "get_model_config",
]
