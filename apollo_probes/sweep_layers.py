from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from transformers import AutoConfig

from .config import MODEL_NAME, DEFAULT_DATASET


def iter_layers(start: int, step: int, model_name: str) -> list[int]:
    config = AutoConfig.from_pretrained(model_name)
    max_layer = config.num_hidden_layers
    return [layer for layer in range(start, max_layer + 1, step)]


def run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Apollo probe training across layers.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Training dataset name.")
    parser.add_argument("--layer-start", type=int, default=2, help="First layer to evaluate.")
    parser.add_argument("--layer-step", type=int, default=4, help="Stride between layers.")
    parser.add_argument("--batch-size", type=int, default=2, help="Activation extraction batch size.")
    parser.add_argument("--force-cache", action="store_true", help="Regenerate caches even if present.")
    parser.add_argument("--force-train", action="store_true", help="Retrain probes even if artifacts exist.")
    args = parser.parse_args()

    layers = iter_layers(args.layer_start, args.layer_step, MODEL_NAME)

    cache_cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "apollo_probes.cache_activations",
        "--dataset",
        args.dataset,
        "--batch-size",
        str(args.batch_size),
        "--layers",
        *[str(layer) for layer in layers],
    ]
    if args.force_cache:
        cache_cmd.append("--force")
    run_command(cache_cmd)

    for layer in layers:
        train_cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "apollo_probes.train_probe",
            "--dataset",
            args.dataset,
            "--layer",
            str(layer),
        ]
        if args.force_train:
            train_cmd.append("--force")
        run_command(train_cmd)


if __name__ == "__main__":
    main()
