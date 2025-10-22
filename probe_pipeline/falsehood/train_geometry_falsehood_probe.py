# %% [markdown]
# Train a falsehood probe direction using the Geometry-of-Truth cities dataset.

from __future__ import annotations

import csv
import json
import os
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils import init_model
from falsehood.utils import FalsehoodVariant, TokenizedFalsehoodExample, pad_batch, tokenize_falsehood_variant


GEOMETRY_DATA_PATH = Path(
    os.environ.get(
        "GEOMETRY_CITIES_PATH",
        "probe_pipeline/falsehood/data/cities.csv",
    )
)

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
DEFAULT_LAYER_INDEX = 22
LAYER_INDEX = int(os.environ.get("LAYER_INDEX", DEFAULT_LAYER_INDEX))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 2))
TRAIN_FRACTION = float(os.environ.get("TRAIN_FRACTION", 0.5))
RANDOM_SEED = 42
LOGREG_C = float(os.environ.get("LOGREG_C", 1.0))
MAX_ITER = int(os.environ.get("MAX_ITER", 1000))

# Output locations
GEOMETRY_PROBE_DIR = Path(f"probe_pipeline/falsehood/geometry_probes/layer{LAYER_INDEX}")
GEOMETRY_RESULTS_DIR = Path(f"probe_pipeline/falsehood/geometry_results/layer{LAYER_INDEX}")
GEOMETRY_PLOTS_DIR = Path(f"probe_pipeline/falsehood/geometry_plots/layer{LAYER_INDEX}")

TRANSPLANT_PROBE_DIR = Path(f"probe_pipeline/falsehood/probes/layer{LAYER_INDEX}")


GEOMETRY_PROBE_DIR.mkdir(parents=True, exist_ok=True)
GEOMETRY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GEOMETRY_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TRANSPLANT_PROBE_DIR.mkdir(parents=True, exist_ok=True)


USER_PROMPT_TEMPLATE = "Where is the city of {city}?"
FOLLOWUP_PROMPT = "Was that statement true?\nA: Yes\nB: No\n\nI think the answer is {guess}"
TAG_GUESS_A = "truth_yes_guess_a"
TAG_GUESS_B = "truth_yes_guess_b"


@dataclass
class GeometryRow:
    city: str
    statement: str
    is_true: bool
    deception_label: int  # 1 if statement false, else 0


def read_geometry_dataset(path: Path) -> List[GeometryRow]:
    if not path.exists():
        raise FileNotFoundError(f"Geometry cities dataset not found at {path}")

    rows: List[GeometryRow] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for entry in reader:
            city = entry["city"].strip()
            statement = entry["statement"].strip()
            is_true = bool(int(entry["label"]))
            deception_label = 0 if is_true else 1
            rows.append(GeometryRow(city=city, statement=statement, is_true=is_true, deception_label=deception_label))
    if not rows:
        raise ValueError("Geometry dataset is empty.")
    return rows


def build_variants(rows: Sequence[GeometryRow]) -> Tuple[List[FalsehoodVariant], List[int], List[str], List[str], List[str]]:
    variants: List[FalsehoodVariant] = []
    labels: List[int] = []
    tags: List[str] = []
    cities: List[str] = []
    statements: List[str] = []

    for idx, row in enumerate(rows):
        base_messages = [
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(city=row.city)},
            {"role": "assistant", "content": row.statement},
        ]

        for guess, tag in (("A", TAG_GUESS_A), ("B", TAG_GUESS_B)):
            new_messages = base_messages + [
                {
                    "role": "user",
                    "content": FOLLOWUP_PROMPT.format(guess=guess),
                }
            ]
            variants.append(
                FalsehoodVariant(
                    messages=new_messages,
                    option_a="Yes",
                    option_b="No",
                    guess=guess,
                    label=row.deception_label,
                    tag=tag,
                )
            )
            labels.append(row.deception_label)
            tags.append(tag)
            cities.append(row.city)
            statements.append(row.statement)

    return variants, labels, tags, cities, statements


def tokenize_variants(
    variants: Sequence[FalsehoodVariant],
    tokenizer,
) -> List[TokenizedFalsehoodExample]:
    tokenized: List[TokenizedFalsehoodExample] = []
    for variant in tqdm(variants, desc="Tokenizing geometry variants", leave=False):
        tokenized.append(tokenize_falsehood_variant(variant, tokenizer))
    return tokenized


def extract_activations(
    examples: Sequence[TokenizedFalsehoodExample],
    tokenizer,
    model,
    device,
) -> torch.Tensor:
    activations: List[torch.Tensor] = []
    for start in tqdm(range(0, len(examples), BATCH_SIZE), desc="Extracting activations", leave=False):
        batch = examples[start : start + BATCH_SIZE]
        input_ids = pad_batch([ex.input_ids for ex in batch], tokenizer.pad_token_id)
        attention_mask = pad_batch([ex.attention_mask for ex in batch], 0)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"), torch.no_grad():
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_layer = outputs.hidden_states[LAYER_INDEX]
        for sample_idx, example in enumerate(batch):
            token_vector = hidden_layer[sample_idx, example.target_index, :]
            if token_vector.dtype != torch.float32:
                token_vector = token_vector.to(torch.float32)
            activations.append(token_vector.detach().cpu())

    if not activations:
        raise ValueError("No activations extracted; check dataset preparation.")
    return torch.stack(activations, dim=0)


def build_difference_dataset(
    activations: torch.Tensor,
    labels: Sequence[int],
    tags: Sequence[str],
    cities: Sequence[str],
    statements: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    if not (
        len(activations)
        == len(labels)
        == len(tags)
        == len(cities)
        == len(statements)
    ):
        raise ValueError("Mismatch in activations/labels/tags/cities/statements lengths.")

    per_pair_indices: Dict[Tuple[str, str], Dict[str, int]] = {}
    for idx, (city, statement, tag) in enumerate(zip(cities, statements, tags)):
        key = (city, statement)
        if key not in per_pair_indices:
            per_pair_indices[key] = {}
        per_pair_indices[key][tag] = idx

    diff_vectors: List[np.ndarray] = []
    diff_labels: List[int] = []
    diff_pairs: List[Tuple[str, str]] = []

    for (city, statement), mapping in per_pair_indices.items():
        idx_a = mapping.get(TAG_GUESS_A)
        idx_b = mapping.get(TAG_GUESS_B)
        if idx_a is None or idx_b is None:
            continue
        vec_a = activations[idx_a].numpy()
        vec_b = activations[idx_b].numpy()
        diff_vectors.append(vec_a - vec_b)
        diff_labels.append(labels[idx_a])
        diff_pairs.append((city, statement))

    if not diff_vectors:
        raise ValueError("No paired difference vectors constructed.")

    diff_X = np.stack(diff_vectors, axis=0).astype(np.float32)
    diff_y = np.array(diff_labels, dtype=np.int64)
    return diff_X, diff_y, diff_pairs


def split_by_pair(
    X: np.ndarray,
    y: np.ndarray,
    pairs: Sequence[Tuple[str, str]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, str]], List[Tuple[str, str]]]:
    num_samples = X.shape[0]
    if num_samples == 0:
        raise ValueError("No samples to split.")

    index_pairs = list(zip(range(num_samples), pairs))
    random.Random(RANDOM_SEED).shuffle(index_pairs)

    split_idx = max(1, int(num_samples * TRAIN_FRACTION))
    if split_idx >= num_samples:
        split_idx = num_samples - 1

    train_info = index_pairs[:split_idx]
    eval_info = index_pairs[split_idx:]

    train_idx = [idx for idx, _ in train_info]
    eval_idx = [idx for idx, _ in eval_info]

    X_train, y_train = X[train_idx], y[train_idx]
    X_eval, y_eval = X[eval_idx], y[eval_idx]
    train_pairs = [pair for _idx, pair in train_info]
    eval_pairs = [pair for _idx, pair in eval_info]

    return X_train, y_train, X_eval, y_eval, train_pairs, eval_pairs


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(
        C=LOGREG_C,
        max_iter=MAX_ITER,
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_on_split(
    clf: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
) -> Dict[str, float]:
    if X.size == 0:
        return {"accuracy": float("nan"), "auroc": float("nan")}
    probs = clf.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(y, preds)
    try:
        auroc = roc_auc_score(y, probs)
    except ValueError:
        auroc = float("nan")
    return {
        f"{split_name}_accuracy": float(acc),
        f"{split_name}_auroc": float(auroc),
    }


def plot_roc(
    clf: LogisticRegression,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    probe_name: str,
) -> Path:
    if X_eval.size == 0 or len(np.unique(y_eval)) < 2:
        plt.figure(figsize=(6, 5))
        plt.text(0.5, 0.5, "Insufficient data for ROC", ha="center", va="center")
        plt.axis("off")
    else:
        probs = clf.predict_proba(X_eval)[:, 1]
        fpr, tpr, _ = roc_curve(y_eval, probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"Eval (AUROC={roc_auc_score(y_eval, probs):.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Geometry Cities Probe ROC")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.4)
    plot_path = GEOMETRY_PLOTS_DIR / f"{probe_name}_roc.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def save_geometry_probe(
    clf: LogisticRegression,
    metadata: dict,
    probe_name: str,
) -> Path:
    out_path = GEOMETRY_PROBE_DIR / f"{probe_name}.pkl"
    with out_path.open("wb") as f:
        pickle.dump(
            {
                "model": clf,
                "metadata": metadata,
                "layer_index": LAYER_INDEX,
                "config": {
                    "C": LOGREG_C,
                    "max_iter": MAX_ITER,
                    "seed": RANDOM_SEED,
                },
            },
            f,
        )
    return out_path


def save_transplanted_probe(
    clf: LogisticRegression,
    metadata: dict,
    probe_name: str = "cities_transplanted",
) -> Path:
    out_path = TRANSPLANT_PROBE_DIR / f"{probe_name}.pkl"
    with out_path.open("wb") as f:
        pickle.dump(
            {
                "model": clf,
                "source": "geometry_cities",
                "trained_on": ["geometry_cities"],
                "datasets": ["geometry_cities"],
                "layer_index": LAYER_INDEX,
                "config": {
                    "C": LOGREG_C,
                    "max_iter": MAX_ITER,
                    "seed": RANDOM_SEED,
                },
                "metadata": metadata,
            },
            f,
        )
    return out_path


def main() -> None:
    rows = read_geometry_dataset(GEOMETRY_DATA_PATH)

    variants, labels, tags, cities, statements = build_variants(rows)

    tokenizer, model, device, _dtype = init_model(MODEL_NAME, seed=RANDOM_SEED)
    tokenized_examples = tokenize_variants(variants, tokenizer)
    activations = extract_activations(tokenized_examples, tokenizer, model, device)
    del tokenized_examples

    diff_X, diff_y, diff_pairs = build_difference_dataset(activations, labels, tags, cities, statements)
    del activations
    X_train, y_train, X_eval, y_eval, train_pairs, eval_pairs = split_by_pair(diff_X, diff_y, diff_pairs)
    train_city_list = [city for city, _ in train_pairs]
    eval_city_list = [city for city, _ in eval_pairs]
    train_pair_set = sorted(set(train_pairs))
    eval_pair_set = sorted(set(eval_pairs))
    train_pair_records = [{"city": city, "statement": stmt} for city, stmt in train_pair_set]
    eval_pair_records = [{"city": city, "statement": stmt} for city, stmt in eval_pair_set]

    clf = train_logistic_regression(X_train, y_train)

    train_metrics = evaluate_on_split(clf, X_train, y_train, "train")
    eval_metrics = evaluate_on_split(clf, X_eval, y_eval, "eval")

    probe_name = "geometry_cities"
    metadata = {
        "train_samples": int(X_train.shape[0]),
        "eval_samples": int(X_eval.shape[0]),
        "train_cities": sorted(set(train_city_list)),
        "eval_cities": sorted(set(eval_city_list)),
        "train_pairs": train_pair_records,
        "eval_pairs": eval_pair_records,
        "train_fraction": TRAIN_FRACTION,
        "geometry_dataset_path": str(GEOMETRY_DATA_PATH),
        "tags": {
            "guess_a": TAG_GUESS_A,
            "guess_b": TAG_GUESS_B,
        },
    }

    plot_path = plot_roc(clf, X_eval, y_eval, probe_name)

    metrics = {
        "probe_name": probe_name,
        "layer_index": LAYER_INDEX,
        "train_fraction": TRAIN_FRACTION,
        **train_metrics,
        **eval_metrics,
        "train_samples": int(X_train.shape[0]),
        "eval_samples": int(X_eval.shape[0]),
        "train_city_count": len(set(train_city_list)),
        "eval_city_count": len(set(eval_city_list)),
        "train_pair_count": len(train_pair_set),
        "eval_pair_count": len(eval_pair_set),
        "geometry_dataset_path": str(GEOMETRY_DATA_PATH),
        "roc_plot": str(plot_path),
    }

    metrics_path = GEOMETRY_RESULTS_DIR / f"{probe_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    geometry_probe_path = save_geometry_probe(clf, metadata, probe_name)
    transplanted_path = save_transplanted_probe(clf, metadata)

    summary = {
        "geometry_probe_path": str(geometry_probe_path),
        "transplanted_probe_path": str(transplanted_path),
        "metrics_path": str(metrics_path),
        "roc_plot": str(plot_path),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }

    summary_path = GEOMETRY_RESULTS_DIR / "geometry_cities_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Geometry cities probe saved to {geometry_probe_path}")
    print(f"Transplanted probe saved to {transplanted_path}")
    print(f"Metrics written to {metrics_path}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
