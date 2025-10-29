#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from transformers import AutoConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probe_pipeline.model_configs import get_model_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect transformer layer attributes for a probe model.")
    parser.add_argument(
        "--model",
        "-m",
        default=os.environ.get("PROBE_MODEL", "llama"),
        help="Model key defined in probe_pipeline.model_configs (default: %(default)s).",
    )
    args = parser.parse_args()

    load_dotenv()
    config_entry = get_model_config(args.model)

    print(f"Config key: {config_entry.key}")
    print(f"Model name: {config_entry.model_name}")

    hf_config = AutoConfig.from_pretrained(config_entry.model_name)
    print(f"Config class: {hf_config.__class__.__name__}")

    # Collect all attributes that look like they could signal the layer count.
    layer_candidates = {}
    for attr in dir(hf_config):
        if "layer" in attr.lower():
            value = getattr(hf_config, attr)
            if isinstance(value, int):
                layer_candidates[attr] = value

    if not layer_candidates:
        print("No integer attributes containing 'layer' found on the config.")
    else:
        print("Layer-related integer attributes:")
        for name, value in sorted(layer_candidates.items()):
            print(f"  {name}: {value}")

    # Print the first few keys from the raw dict for extra context.
    config_dict = hf_config.to_dict()
    print("Config dict keys (sample):")
    for key in list(config_dict.keys())[:20]:
        print(f"  {key}: {config_dict[key]!r}")


if __name__ == "__main__":
    main()
