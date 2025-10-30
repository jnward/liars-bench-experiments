from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from transformers import AutoConfig

from .config import APOLLO_CONFIG, MODEL_NAME, DEFAULT_DATASET
from .data import prepare_datasets


def iter_layers(start: int, step: int, model_name: str) -> list[int]:
    config = AutoConfig.from_pretrained(model_name)
    max_layer = config.num_hidden_layers
    return [layer for layer in range(start, max_layer + 1, step)]


def run_command(cmd: list[str], env: dict[str, str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Apollo probe training across layers.")
    parser.add_argument("--dataset", nargs="+", default=[DEFAULT_DATASET], help="Training dataset name(s).")
    parser.add_argument("--layer-start", type=int, default=APOLLO_CONFIG.layer_start, help="First layer to evaluate.")
    parser.add_argument("--layer-end", type=int, default=None, help="Last layer to evaluate (inclusive).")
    parser.add_argument("--layer-step", type=int, default=APOLLO_CONFIG.layer_step, help="Stride between layers.")
    parser.add_argument("--layers", type=int, nargs="+", help="Explicit layer indices to evaluate.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=APOLLO_CONFIG.default_batch_size,
        help="Activation extraction batch size.",
    )
    parser.add_argument("--force-cache", action="store_true", help="Regenerate caches even if present.")
    parser.add_argument("--force-train", action="store_true", help="Retrain probes even if artifacts exist.")
    args = parser.parse_args()

    env = os.environ.copy()
    env["APOLLO_MODEL"] = APOLLO_CONFIG.key
    env.setdefault("PROBE_MODEL", APOLLO_CONFIG.key)

    dataset_names, dataset_slug = prepare_datasets(args.dataset)
    if args.layers:
        layers = sorted(set(args.layers))
    else:
        layers = iter_layers(args.layer_start, args.layer_step, MODEL_NAME)
        if args.layer_end is not None:
            layers = [layer for layer in layers if layer <= args.layer_end]

    layer_list_env = env.get("LAYER_LIST") or env.get("LAYERS")
    if layer_list_env:
        try:
            layers = sorted(
                set(
                    int(token.strip())
                    for token in layer_list_env.split(",")
                    if token.strip()
                )
            )
        except ValueError as exc:
            raise ValueError(f"Invalid LAYER_LIST/LAYERS specification: {layer_list_env!r}") from exc

    print(f"[sweep] Training datasets: {dataset_slug} ({', '.join(dataset_names)})")

    for dataset in sorted(set(dataset_names)):
        cache_cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "apollo_probes.cache_activations",
            "--dataset",
            dataset,
            "--batch-size",
            str(args.batch_size),
            "--layers",
            *[str(layer) for layer in layers],
        ]
        if args.force_cache:
            cache_cmd.append("--force")
        run_command(cache_cmd, env)

    for layer in layers:
        train_cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "apollo_probes.train_probe",
            "--dataset",
            *dataset_names,
            "--layer",
            str(layer),
        ]
        if args.force_train:
            train_cmd.append("--force")
        run_command(train_cmd, env)


if __name__ == "__main__":
    main()
