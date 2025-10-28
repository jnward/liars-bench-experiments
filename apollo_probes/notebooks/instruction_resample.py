# %% [markdown]
# # Resampling Instructed-But-Truthful Dialogues
#
# This notebook prepares a corpus of chat transcripts where the original LLaMA 3.3 model was instructed
# to be deceptive but responded truthfully. We regenerate the assistant message multiple times with
# sampling to estimate how often the model defects under the same setup. The regenerated responses are
# cached to disk for downstream judging and analysis.

# %%
from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from datasets.utils.logging import disable_progress_bar, set_verbosity_error
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

disable_progress_bar()
set_verbosity_error()

from probe_pipeline.utils import init_model, load_filtered_dataset  # noqa: E402

from apollo_probes.config import RANDOM_SEED  # noqa: E402
from apollo_probes.eval_datasets import liars_bench_specs, slugify_config  # noqa: E402


# %%
# ---- Configuration ----
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
EVAL_CONFIG = "instructed-deception"
NUM_RESAMPLES = 16
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 512
MAX_EXAMPLES: int | None = None  # Set to a small int for smoke tests.
OUTPUT_ROOT = PROJECT_ROOT / "apollo_probes" / "notebooks" / "cache" / "instruction_resample"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

GENERATION_PARAMS = {
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "do_sample": True,
    "max_new_tokens": MAX_NEW_TOKENS,
    "num_return_sequences": NUM_RESAMPLES,
    "pad_token_id": None,
}


# %%
# ---- Helpers for dataset loading ----
def get_eval_spec(config_name: str):
    specs = {spec.config: spec for spec in liars_bench_specs()}
    if config_name not in specs:
        raise ValueError(f"Unknown eval config '{config_name}'. Available: {sorted(specs)}")
    return specs[config_name]


def load_eval_rows(config_name: str, sample_size: int | None, seed: int):
    spec = get_eval_spec(config_name)
    try:
        rows = load_filtered_dataset(
            spec.dataset_id,
            spec.config,
            split=spec.split,
            filter_model_name=spec.filter_model,
            sample_size=sample_size,
            seed=seed,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Failed to load the eval dataset. If you are offline, download "
            f"'{spec.dataset_id}/{spec.config}' locally or enable network access."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset '{spec.dataset_id}/{spec.config}'. "
            "If this environment lacks internet access, pre-download the dataset "
            "or stage it in the Hugging Face cache directory."
        ) from exc
    return rows


def parse_meta(meta_value):
    if isinstance(meta_value, dict):
        return meta_value
    if isinstance(meta_value, str):
        try:
            parsed = ast.literal_eval(meta_value)
        except (ValueError, SyntaxError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def strip_final_assistant(messages: Sequence[dict]) -> tuple[list[dict], dict | None]:
    if not messages:
        return [], None
    trimmed = list(messages)
    last_message = trimmed[-1]
    if last_message.get("role") == "assistant":
        trimmed = trimmed[:-1]
    else:
        last_message = None
    return trimmed, last_message


@dataclass
class DialogueExample:
    dataset_index: int
    dataset_name: str
    row_id: str | None
    meta: dict
    model_belief: int | None
    correct_option: int | None
    prompt_messages: list[dict]
    original_assistant: dict | None


def collect_instructed_truthful(rows) -> list[DialogueExample]:
    subset: list[DialogueExample] = []
    for idx, row in enumerate(rows):
        deceptive = int(row.get("deceptive", 0))
        meta = parse_meta(row.get("meta"))
        if not meta.get("instructed_deceptiveness", False):
            continue
        if deceptive != 0:
            continue
        messages = row.get("messages") or []
        prompt_messages, original_assistant = strip_final_assistant(messages)
        subset.append(
            DialogueExample(
                dataset_index=idx,
                dataset_name=row.get("dataset", "unknown"),
                row_id=row.get("row_id"),
                meta=meta,
                model_belief=meta.get("model_belief"),
                correct_option=meta.get("correct_option"),
                prompt_messages=prompt_messages,
                original_assistant=original_assistant,
            )
        )
    if MAX_EXAMPLES is not None:
        subset = subset[:MAX_EXAMPLES]
    return subset


# %%
# ---- Load dataset & collect targets ----
eval_rows = load_eval_rows(EVAL_CONFIG, sample_size=None, seed=RANDOM_SEED)
examples = collect_instructed_truthful(eval_rows)

print(f"Loaded eval rows: {len(eval_rows)}")
print(f"Subset (instructed & truthful): {len(examples)} examples")
if MAX_EXAMPLES is not None:
    print(f"Processing capped at first {MAX_EXAMPLES} examples.")


# %%
# ---- Caching utilities ----
def cache_path_for_run(config: str, model_name: str, num_samples: int, temperature: float) -> Path:
    alias = slugify_config(config)
    clean_model = model_name.replace("/", "_")
    filename = f"{alias}__{clean_model}__n{num_samples}_temp{temperature:.1f}.jsonl"
    return OUTPUT_ROOT / filename


CACHE_PATH = cache_path_for_run(EVAL_CONFIG, MODEL_NAME, NUM_RESAMPLES, TEMPERATURE)


def load_cached_generations(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    cache: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            cache[int(record["dataset_index"])] = record
    return cache


def append_generation_record(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")


existing_cache = load_cached_generations(CACHE_PATH)
print(f"Existing cached examples: {len(existing_cache)}")


# %%
# ---- Model initialization ----
tokenizer, model, device, _dtype = init_model(MODEL_NAME, seed=RANDOM_SEED)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
GENERATION_PARAMS["pad_token_id"] = tokenizer.pad_token_id

model_kwargs = {
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "do_sample": True,
    "max_new_tokens": MAX_NEW_TOKENS,
    "num_return_sequences": NUM_RESAMPLES,
    "return_dict_in_generate": True,
}


def render_prompt(messages: Sequence[dict]) -> dict[str, torch.Tensor]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(encoded, dict):
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
    else:
        input_ids = encoded
        attention_mask = torch.ones_like(encoded)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }


# %%
# ---- Generation loop ----
pending_examples = [ex for ex in examples if ex.dataset_index not in existing_cache]

print(f"Pending generations: {len(pending_examples)}")
if not pending_examples:
    print("Nothing to generate; cache already complete.")


def generate_resamples(example: DialogueExample) -> list[str]:
    prompts = render_prompt(example.prompt_messages)
    input_len = prompts["input_ids"].shape[-1]
    outputs = model.generate(
        **prompts,
        **model_kwargs,
        pad_token_id=tokenizer.pad_token_id,
    )
    sequences = outputs.sequences
    generated = sequences[:, input_len:].cpu()
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [text.strip() for text in texts]


for example in tqdm(pending_examples, desc="Generating resamples"):
    try:
        generations = generate_resamples(example)
    except RuntimeError as exc:
        print(f"[warn] Generation failed for index {example.dataset_index}: {exc}")
        continue

    record = {
        "dataset_index": example.dataset_index,
        "dataset_name": example.dataset_name,
        "row_id": example.row_id,
        "meta": example.meta,
        "model_belief": example.model_belief,
        "correct_option": example.correct_option,
        "original_assistant": example.original_assistant,
        "prompt_messages": example.prompt_messages,
        "generation_params": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "num_return_sequences": NUM_RESAMPLES,
        },
        "resamples": [
            {"sample_index": idx, "text": text}
            for idx, text in enumerate(generations)
        ],
    }
    append_generation_record(CACHE_PATH, record)

print("Generation complete. Cached responses at:", CACHE_PATH)
