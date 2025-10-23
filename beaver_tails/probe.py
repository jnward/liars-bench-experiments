from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch


class ProbeLoadError(RuntimeError):
    """Raised when a probe file cannot be interpreted."""


@dataclass(slots=True)
class LoadedProbe:
    name: str
    path: Path
    direction: torch.Tensor
    normalize: bool
    scaler_mean: torch.Tensor | None
    scaler_scale: torch.Tensor | None
    intercept: float


def _to_tensor(data) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().to(torch.float32)
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(torch.float32)
    return torch.tensor(data, dtype=torch.float32)


def _sanitize_name(path: Path) -> str:
    return path.stem.replace(" ", "_")


DEFAULT_PROBE_PATH = Path("/workspace/jake/deception-detection/example_results/instructed_pairs/detector.pt")


def load_probe(path: Path, layer_index: int) -> LoadedProbe:
    with path.open("rb") as f:
        try:
            payload = pickle.load(f)
        except ModuleNotFoundError as exc:
            raise ProbeLoadError(f"Missing dependency while loading {path}: {exc}") from exc

    # Probe dict with per-layer directions (e.g., Apollo)
    if isinstance(payload, dict) and "directions" in payload:
        layers = list(payload.get("layers", []))
        if not layers:
            raise ProbeLoadError(f"{path} missing layer metadata.")
        try:
            idx = layers.index(layer_index)
        except ValueError as exc:
            raise ProbeLoadError(f"Layer {layer_index} unavailable in {path} (found {layers}).") from exc

        direction = _to_tensor(payload["directions"][idx]).view(-1)
        normalize = bool(payload.get("normalize", False))
        scaler_mean = _to_tensor(payload["scaler_mean"][idx]).view(-1) if normalize else None
        scaler_scale = _to_tensor(payload["scaler_scale"][idx]).view(-1) if normalize else None
        intercept_obj = payload.get("logistic_intercept", 0.0)
        if isinstance(intercept_obj, (list, tuple)):
            intercept = float(intercept_obj[idx])
        else:
            intercept = float(intercept_obj or 0.0)

        return LoadedProbe(
            name=_sanitize_name(path),
            path=path,
            direction=direction,
            normalize=normalize,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            intercept=intercept,
        )

    # LogisticRegression pickles (either raw estimator or wrapped dict)
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        probe_name = payload.get("name") or _sanitize_name(path)
    else:
        model = payload
        probe_name = _sanitize_name(path)

    cls_name = model.__class__.__name__
    if cls_name != "LogisticRegression":
        raise ProbeLoadError(f"Unsupported probe object in {path}: {cls_name}")

    coef = getattr(model, "coef_", None)
    intercept_arr = getattr(model, "intercept_", None)
    if coef is None:
        raise ProbeLoadError(f"Logistic probe {path} missing coefficients.")

    direction = _to_tensor(coef).view(-1)
    if intercept_arr is not None:
        intercept = float(_to_tensor(intercept_arr).view(-1)[0])
    else:
        intercept = 0.0

    return LoadedProbe(
        name=probe_name,
        path=path,
        direction=direction,
        normalize=False,
        scaler_mean=None,
        scaler_scale=None,
        intercept=intercept,
    )


def default_probe_paths(layer_index: int) -> List[Path]:
    base = Path("probe_pipeline/probes") / f"layer{layer_index}"
    paths: list[Path] = []
    if base.exists():
        paths.extend(sorted(base.glob("*.pkl")))
    return paths
