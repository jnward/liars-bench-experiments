from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from tqdm.auto import tqdm

from ..config import MODEL_NAME, RANDOM_SEED
from ..eval_datasets import liars_bench_specs, slugify_config
from .config import DEFAULT_LAYER_INDEX
from .paths import diff_probe_dir, diff_results_dir
from .processing import (
    DifferenceGroup,
    assemble_difference_vectors,
    extract_variant_results,
    prepare_requests,
)
from .types import DialogueInfo, VariantResult

from deception_detection.types import Dialogue, Message  # type: ignore
from probe_pipeline.utils import init_model, load_filtered_dataset


def _load_probe(dataset_slug: str, layer: int, variant: str) -> dict:
    probe_path = diff_probe_dir(dataset_slug, layer) / f"{variant}_probe.pkl"
    if not probe_path.exists():
        raise FileNotFoundError(f"Diff probe not found at {probe_path}")
    return torch.load(probe_path, map_location="cpu")


def _score_features(features: np.ndarray, probe: dict) -> np.ndarray:
    mean = probe["mean"].to(torch.float32).numpy()
    std = probe["std"].to(torch.float32).numpy()
    coef = probe["coef"].to(torch.float32).numpy().reshape(-1)
    intercept = probe["intercept"].to(torch.float32).numpy().reshape(-1)[0]

    std = np.clip(std, 1e-8, None)
    normalized = (features - mean) / std
    logits = normalized @ coef + intercept
    return logits


def _logit_to_prob(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def _dialogue_from_messages(messages: Sequence[Dict[str, str]]) -> Dialogue:
    return [Message(role=msg["role"], content=msg["content"], detect=False) for msg in messages]


def _prepare_eval_entries(dataset: Sequence[dict], dataset_slug: str) -> List[tuple[DialogueInfo, Dialogue]]:
    entries: List[tuple[DialogueInfo, Dialogue]] = []
    for idx, row in enumerate(dataset):
        messages = row.get("messages") or []
        dialogue = _dialogue_from_messages(messages)
        label = int(bool(row.get("deceptive", False)))
        info = DialogueInfo(dataset=dataset_slug, dialogue_index=idx, label=label, meta={})
        entries.append((info, dialogue))
    return entries


def _compute_difference_group(
    entries: Sequence[tuple[DialogueInfo, Dialogue]],
    layer: int,
    batch_size: int,
    tokenizer,
    model,
    *,
    store_variant_hidden: bool = False,
) -> DifferenceGroup:
    requests = prepare_requests(entries)
    results: List[VariantResult] = extract_variant_results(
        requests,
        layer_index=layer,
        batch_size=batch_size,
        tokenizer=tokenizer,
        model=model,
    )
    return assemble_difference_vectors(results, store_variant_hidden=store_variant_hidden)


def _features_from_group(group: DifferenceGroup, variant: str) -> np.ndarray:
    tensor_map = {
        "user": group.user_diff,
        "assistant": group.assistant_diff,
        "combined": group.combined_diff,
    }
    if variant not in tensor_map:
        raise ValueError(f"Unsupported feature variant '{variant}'.")
    return tensor_map[variant].to(torch.float32).numpy()


def _violin_groups(scores: np.ndarray, labels: np.ndarray) -> dict[str, List[float]]:
    groups = {
        "negative": scores[labels == 0].tolist(),
        "positive": scores[labels == 1].tolist(),
    }
    return groups


def _plot_violin(scores: np.ndarray, labels: np.ndarray, title: str, output_path: Path) -> None:
    groups = _violin_groups(scores, labels)
    if not any(groups.values()):
        return
    plt.figure(figsize=(6, 4))
    data = [groups[name] for name in ["negative", "positive"] if groups[name]]
    labels_to_plot = [name.capitalize() for name in ["negative", "positive"] if groups[name]]
    plt.violinplot(data, showmeans=True)
    plt.xticks(range(1, len(labels_to_plot) + 1), labels_to_plot)
    plt.ylabel("Probe score")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_roc(labels: np.ndarray, probs: np.ndarray, title: str, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUROC={roc_auc_score(labels, probs):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate_diff_probe(
    dataset_slug: str,
    variant: str = "user",
    layer: int = DEFAULT_LAYER_INDEX,
    dataset_filters: Iterable[str] | None = None,
    batch_size: int = 2,
    sample_size: int | None = None,
    seed: int = RANDOM_SEED,
    chunk_size: int = 64,
) -> dict:
    probe = _load_probe(dataset_slug, layer, variant)
    tokenizer, model, device, _dtype = init_model(MODEL_NAME, seed=seed)
    model.eval()

    specs = liars_bench_specs()
    if dataset_filters:
        filters = set(dataset_filters)
        specs = [spec for spec in specs if spec.config in filters]
        if not specs:
            raise ValueError(f"No evaluation datasets match filters: {sorted(filters)}")

    summary: dict[str, dict] = {
        "dataset": dataset_slug,
        "layer_index": layer,
        "feature_variant": variant,
        "datasets": {},
    }

    results_dir = diff_results_dir(dataset_slug, layer)

    for spec in tqdm(specs, desc="Eval datasets"):
        dataset = load_filtered_dataset(
            spec.dataset_id,
            spec.config,
            spec.split,
            spec.filter_model,
            sample_size,
            seed,
        )

        dataset_size = len(dataset)
        if dataset_size == 0:
            continue
        chunk_size = max(1, int(chunk_size))
        logits_chunks: List[np.ndarray] = []
        labels_chunks: List[np.ndarray] = []

        for start in range(0, dataset_size, chunk_size):
            end = min(start + chunk_size, dataset_size)
            subset = dataset.select(range(start, end))
            entries = _prepare_eval_entries(subset, spec.config)
            if not entries:
                continue
            group = _compute_difference_group(
                entries,
                layer,
                batch_size,
                tokenizer,
                model,
                store_variant_hidden=False,
            )
            features = _features_from_group(group, variant)
            if features.size == 0:
                continue
            labels = group.labels.numpy()
            logits = _score_features(features, probe)

            logits_chunks.append(logits)
            labels_chunks.append(labels)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not logits_chunks:
            continue

        logits = np.concatenate(logits_chunks, axis=0)
        labels = np.concatenate(labels_chunks, axis=0)

        probs = _logit_to_prob(logits)
        preds = (probs > 0.5).astype(int)

        metrics = {
            "samples": int(labels.size),
            "accuracy": float(accuracy_score(labels, preds)),
        }
        if len(np.unique(labels)) > 1:
            metrics["auroc"] = float(roc_auc_score(labels, probs))
        else:
            metrics["auroc"] = float("nan")

        slug = slugify_config(spec.config)
        roc_path = results_dir / f"{variant}_{slug}_roc.png"
        violin_path = results_dir / f"{variant}_{slug}_violin.png"
        _plot_roc(labels, probs, f"{spec.config} - {variant}", roc_path)
        _plot_violin(logits, labels, f"{spec.config} - logits", violin_path)

        metrics.update(
            {
                "roc_path": str(roc_path),
                "violin_path": str(violin_path),
            }
        )
        summary["datasets"][spec.config] = metrics

    summary_path = results_dir / f"{variant}_evaluation.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary
