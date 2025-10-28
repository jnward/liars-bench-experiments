# %% [markdown]
# Layer sweep driver: caches activations, trains/evaluates probes across layers.

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from transformers import AutoConfig


MODEL_NAME = "Qwen/Qwen3-32B"
LAYER_START = 2
LAYER_STEP = 4
KEEP_CACHE = False  # set True to retain cached activations

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "probe_pipeline" / "cache-qwen-baseline"


def iter_layers(model_name: str) -> list[int]:
    config = AutoConfig.from_pretrained(model_name)
    max_layer = config.num_hidden_layers
    return [layer for layer in range(LAYER_START, max_layer + 1, LAYER_STEP)]


def run_script(script: str, layer: int) -> None:
    env = os.environ.copy()
    env["LAYER_INDEX"] = str(layer)
    subprocess.run(
        ["uv", "run", "python", script],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def clean_cache(layer: int) -> None:
    if KEEP_CACHE:
        return
    pattern = f"*layer{layer}.pt"
    for path in CACHE_DIR.glob(pattern):
        path.unlink()
    for path in CACHE_DIR.glob(f"*layer{layer}.meta.json"):
        path.unlink()


def main() -> None:
    layers = iter_layers(MODEL_NAME)
    for layer in layers:
        print(f"\n=== Layer {layer} ===")
        run_script("probe_pipeline/cache_activations.py", layer)
        run_script("probe_pipeline/sweep_probes.py", layer)
        clean_cache(layer)


if __name__ == "__main__":
    main()

