# %% [markdown]
# Probe Direction Cosine Similarity
#
# This short notebook compares the steering directions from two probes by
# computing their cosine similarity.

# %%
from __future__ import annotations

import pickle
from pathlib import Path

import torch

# %%
PROBE_A = Path("/workspace/jake/deception-detection/example_results/instructed_pairs/detector.pt")
PROBE_B = Path("probe-convincing-game.pkl")
TARGET_LAYER_INDEX = 22


def load_probe_direction(path: Path, layer_index: int) -> torch.Tensor:
    with path.open("rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "directions" in obj:
        layers = obj["layers"]
        directions = torch.tensor(obj["directions"], dtype=torch.float32)
        try:
            idx = list(layers).index(layer_index)
        except ValueError as exc:
            raise ValueError(f"Layer {layer_index} not found in probe {path}") from exc
        direction = directions[idx]
    elif obj.__class__.__name__ == "LogisticRegression":
        direction = torch.tensor(obj.coef_, dtype=torch.float32).squeeze(0)
    else:
        raise ValueError(f"Unsupported probe format for {path}: {type(obj)}")

    norm = direction.norm()
    if norm <= 0:
        raise ValueError(f"Zero-norm direction in probe {path}")
    return direction / norm


dir_a = load_probe_direction(PROBE_A, TARGET_LAYER_INDEX)
dir_b = load_probe_direction(PROBE_B, TARGET_LAYER_INDEX)

cosine_similarity = torch.dot(dir_a, dir_b).item()

print(f"Cosine similarity between probe directions: {cosine_similarity:.6f}")


# %%
