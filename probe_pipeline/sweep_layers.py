# %% [markdown]
# Layer sweep driver: caches activations, trains/evaluates probes across layers.

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoConfig

from model_configs import get_model_config


CONFIG_KEY = os.environ.get("PROBE_MODEL", "llama")
CONFIG = get_model_config(CONFIG_KEY)

load_dotenv()

MODEL_NAME = CONFIG.model_name
LAYER_START = CONFIG.layer_start
LAYER_STEP = CONFIG.layer_step
KEEP_CACHE = False  # set True to retain cached activations

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = CONFIG.cache_dir


def iter_layers(model_name: str) -> list[int]:
    config = AutoConfig.from_pretrained(model_name)
    path = CONFIG.layer_count_path
    max_layer: int | None = None
    if path:
        cursor = config
        try:
            for key in path:
                if hasattr(cursor, key):
                    cursor = getattr(cursor, key)
                elif isinstance(cursor, dict) and key in cursor:
                    cursor = cursor[key]
                else:
                    raise AttributeError
            if isinstance(cursor, int):
                max_layer = cursor
        except AttributeError:
            max_layer = None

    if max_layer is None:
        # Fallback to simple heuristics for unforeseen models.
        for attr in ("num_hidden_layers", "num_layers", "num_transformer_layers", "n_layer"):
            value = getattr(config, attr, None)
            if isinstance(value, int):
                max_layer = value
                break

    if max_layer is None:
        raise AttributeError(
            f"Could not determine layer count for {model_name}; set layer_count_path in model_configs."
        )
    return [layer for layer in range(LAYER_START, max_layer + 1, LAYER_STEP)]


def run_script(script: str, layer: int) -> None:
    env = os.environ.copy()
    env["LAYER_INDEX"] = str(layer)
    env["PROBE_MODEL"] = CONFIG.key
    subprocess.run(
        ["uv", "run", "python", script],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def clean_cache(layer: int) -> None:
    if KEEP_CACHE:
        return
    for path in CACHE_DIR.glob(f"*layer{layer}.pt"):
        path.unlink()
    for path in CACHE_DIR.glob(f"*layer{layer}.meta.json"):
        path.unlink()


def main() -> None:
    layer_list_env = os.environ.get("LAYER_LIST") or os.environ.get("LAYERS")
    if layer_list_env:
        try:
            layers = [
                int(token.strip())
                for token in layer_list_env.split(",")
                if token.strip()
            ]
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid layer specification: {layer_list_env!r}") from exc
        if not layers:
            raise ValueError("No valid layers parsed from LAYER_LIST / LAYERS.")
    else:
        layers = iter_layers(MODEL_NAME)

    for layer in layers:
        print(f"\n=== Layer {layer} ===")
        run_script("probe_pipeline/cache_activations.py", layer)
        run_script("probe_pipeline/sweep_probes.py", layer)
        clean_cache(layer)


if __name__ == "__main__":
    main()
