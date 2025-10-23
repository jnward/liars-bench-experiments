from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from tqdm.auto import tqdm

from . import CACHE_DIR, SCORES_DIR
from .probe import DEFAULT_PROBE_PATH, LoadedProbe, default_probe_paths, load_probe


MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
LAYER_INDEX = 22


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_probes(
    layer_index: int,
    explicit_paths: Sequence[Path] | None,
    include_defaults: bool,
) -> list[LoadedProbe]:
    paths: list[Path] = []
    seen: set[Path] = set()

    if include_defaults:
        if DEFAULT_PROBE_PATH.exists():
            paths.append(DEFAULT_PROBE_PATH)
            seen.add(DEFAULT_PROBE_PATH.resolve())
        for candidate in default_probe_paths(layer_index):
            resolved = candidate.resolve()
            if resolved not in seen:
                paths.append(candidate)
                seen.add(resolved)

    if explicit_paths:
        for path in explicit_paths:
            resolved = path.resolve()
            if resolved not in seen:
                paths.append(path)
                seen.add(resolved)

    loaded: list[LoadedProbe] = []
    for path in paths:
        loaded.append(load_probe(path, layer_index=layer_index))
    return loaded


def score_probes(
    manifest_path: Path,
    probe_paths: Sequence[Path] | None = None,
    output_dir: Path | None = None,
    include_defaults: bool = True,
) -> dict[str, Path]:
    manifest = _load_manifest(manifest_path)
    probes = _resolve_probes(LAYER_INDEX, probe_paths, include_defaults)
    if not probes:
        raise ValueError("No probes provided for scoring.")

    cache_dir = manifest_path.parent
    scores_dir = output_dir or SCORES_DIR
    scores_dir.mkdir(parents=True, exist_ok=True)

    handles: dict[str, tuple[LoadedProbe, Path, object]] = {}
    for probe in probes:
        out_path = scores_dir / f"{probe.name}_layer{LAYER_INDEX:02d}_scores.jsonl"
        fout = out_path.open("w", encoding="utf-8")
        handles[probe.name] = (probe, out_path, fout)

    try:
        for shard in tqdm(manifest["shards"], desc="Scoring probes"):
            shard_file = cache_dir / shard["path"]
            payload = torch.load(shard_file, map_location="cpu")
            for example in payload["examples"]:
                hidden_fp32 = example["hidden"].to(torch.float32)
                base_record = {
                    "row_id": example["row_id"],
                    "num_tokens": int(example.get("num_tokens", hidden_fp32.size(0))),
                    "categories": example["categories"],
                }
                for probe_name, (probe, _path, fout) in handles.items():
                    acts = hidden_fp32
                    if probe.normalize and probe.scaler_mean is not None and probe.scaler_scale is not None:
                        acts = (acts - probe.scaler_mean) / probe.scaler_scale
                    token_scores = torch.mv(acts, probe.direction)
                    average_score = float(token_scores.mean().item())
                    record = dict(base_record)
                    record.update(
                        {
                            "probe": probe_name,
                            "score": average_score,
                        }
                    )
                    if probe.intercept:
                        record["intercept"] = float(probe.intercept)
                    fout.write(json.dumps(record) + "\n")
    finally:
        for probe, path, fout in handles.values():
            fout.close()

    return {name: path for name, (_, path, _) in handles.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score cached BeaverTails activations with a linear probe.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CACHE_DIR / f"beavertails_layer{LAYER_INDEX:02d}_manifest.json",
        help="Path to the activation cache manifest.",
    )
    parser.add_argument(
        "--probe-path",
        type=Path,
        action="append",
        default=None,
        help="Additional probe path to evaluate (repeatable).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write score JSONL files (defaults to beaver_tails/scores).",
    )
    parser.add_argument(
        "--no-defaults",
        action="store_true",
        help="Do not include Apollo and layer directory probes unless explicitly provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = score_probes(
        manifest_path=args.manifest,
        probe_paths=args.probe_path,
        output_dir=args.output_dir,
        include_defaults=not args.no_defaults,
    )
    for name, path in outputs.items():
        print(f"Wrote scores for {name} -> {path}")


if __name__ == "__main__":
    main()
