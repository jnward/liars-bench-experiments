# Helper functions shared across probe caching/training/evaluation.

from __future__ import annotations

import ast
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


# Global tokenizer/model (lazy init)
_tokenizer: AutoTokenizer | None = None
_model: AutoModelForCausalLM | None = None
_device: torch.device | None = None
_dtype: torch.dtype | None = None


def init_model(model_name: str, seed: int = 42) -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.device, torch.dtype]:
    global _tokenizer, _model, _device, _dtype
    if _tokenizer is None or _model is None:
        load_dotenv()
        torch.set_grad_enabled(False)
        random.seed(seed)
        torch.manual_seed(seed)

        requested_model_name = os.environ.get("MODEL_NAME", model_name)

        _tokenizer = AutoTokenizer.from_pretrained(requested_model_name)
        _tokenizer.padding_side = "left"
        if _tokenizer.pad_token_id is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        load_in_4bit = str(os.environ.get("LOAD_IN_4BIT", "")).lower() in {"1", "true", "yes"}
        load_in_8bit = str(os.environ.get("LOAD_IN_8BIT", "")).lower() in {"1", "true", "yes"}
        if load_in_4bit and load_in_8bit:
            raise ValueError("Only one of LOAD_IN_4BIT or LOAD_IN_8BIT may be set.")

        model_kwargs: dict[str, Any] = {
            "torch_dtype": _dtype,
            "device_map": os.environ.get("MODEL_DEVICE_MAP", "auto"),
            "low_cpu_mem_usage": True,
        }

        offload_env = os.environ.get("HF_OFFLOAD_DIR")
        if offload_env:
            offload_dir = Path(offload_env).expanduser()
            offload_dir.mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = str(offload_dir)

        max_cpu_mem = os.environ.get("HF_MAX_CPU_MEMORY")
        if max_cpu_mem:
            model_kwargs["max_memory"] = {"cpu": str(max_cpu_mem)}

        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:  # pragma: no cover - helpful runtime message
                raise ImportError(
                    "bitsandbytes is required for 4-bit/8-bit loading. Install it or unset LOAD_IN_4BIT/LOAD_IN_8BIT."
                ) from exc

            quant_kwargs = {"load_in_4bit": load_in_4bit, "load_in_8bit": load_in_8bit}
            if load_in_4bit:
                quant_kwargs.update(
                    {
                        "bnb_4bit_use_double_quant": True,
                        "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_compute_dtype": _dtype,
                    }
                )

            quant_config = BitsAndBytesConfig(**quant_kwargs)
            model_kwargs["quantization_config"] = quant_config
            model_kwargs.pop("torch_dtype", None)

        _model = AutoModelForCausalLM.from_pretrained(
            requested_model_name,
            **model_kwargs,
        )
        _model.eval()

    return _tokenizer, _model, _device, _dtype


def _get_turn_name(meta_value: Any) -> str:
    if isinstance(meta_value, dict):
        return str(meta_value.get("turn_name", ""))
    if isinstance(meta_value, str):
        try:
            parsed = ast.literal_eval(meta_value)
        except (ValueError, SyntaxError):
            return ""
        if isinstance(parsed, dict):
            return str(parsed.get("turn_name", ""))
    return ""


def load_filtered_dataset(
    dataset_id: str,
    config: str,
    split: str,
    filter_model_name: str,
    sample_size: int | None,
    seed: int,
) -> Dataset:
    base_config = config
    turn_name_filter: str | None = None

    if "/" in config:
        base_config, sub_key = config.split("/", 1)
        if base_config == "insider-trading":
            turn_name_filter = sub_key

    dataset: Dataset = load_dataset(dataset_id, base_config, split=split)

    if turn_name_filter is not None:
        dataset = dataset.filter(
            lambda row: _get_turn_name(row.get("meta")) == turn_name_filter
        )

    dataset = dataset.filter(lambda row: row.get("model") == filter_model_name)

    if sample_size is not None and sample_size < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    return dataset


@dataclass
class TokenizedConversation:
    rendered: str
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    assistant_token_indices: list[int]
    label: int


def _get_assistant_header_ids(tok: AutoTokenizer) -> list[int]:
    header_str = "<|start_header_id|>assistant<|end_header_id|>"
    return tok.encode(header_str, add_special_tokens=False)


def locate_last_assistant_indices(input_ids: Sequence[int], header_ids: Sequence[int]) -> list[int]:
    matches: list[int] = []
    header_len = len(header_ids)
    for pos in range(0, len(input_ids) - header_len + 1):
        if list(input_ids[pos : pos + header_len]) == list(header_ids):
            matches.append(pos)
    if not matches:
        return []
    chosen_idx = matches[-1]
    if chosen_idx + header_len >= len(input_ids) and len(matches) >= 2:
        chosen_idx = matches[-2]
    return list(range(chosen_idx, len(input_ids)))


def tokenize_conversation(row: dict, tokenizer: AutoTokenizer, header_ids: Sequence[int]) -> TokenizedConversation:
    messages = row["messages"]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
    if isinstance(token_ids, torch.Tensor):
        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze(0)
        token_ids = token_ids.tolist()

    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]

    input_ids = torch.tensor(token_ids, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    assistant_indices = locate_last_assistant_indices(input_ids.tolist(), header_ids)
    return TokenizedConversation(
        rendered=rendered,
        input_ids=input_ids,
        attention_mask=attention_mask,
        assistant_token_indices=assistant_indices,
        label=int(row.get("deceptive", 0)),
    )


def tokenize_dataset_rows(dataset: Dataset, tokenizer: AutoTokenizer) -> list[TokenizedConversation]:
    header_ids = _get_assistant_header_ids(tokenizer)
    tokenized: list[TokenizedConversation] = []
    for row in tqdm(dataset, desc="Tokenizing", leave=False):
        tokenized.append(tokenize_conversation(row, tokenizer, header_ids))
    return tokenized


def pad_batch(tensors: Sequence[torch.LongTensor], pad_value: int) -> torch.LongTensor:
    max_len = max(t.size(0) for t in tensors)
    padded = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
    for i, tensor in enumerate(tensors):
        padded[i, : tensor.size(0)] = tensor
    return padded


def collate_conversations(
    conversations: Sequence[TokenizedConversation],
    tokenizer: AutoTokenizer,
) -> tuple[torch.LongTensor, torch.LongTensor, list[list[int]], list[int]]:
    input_ids = pad_batch([c.input_ids for c in conversations], tokenizer.pad_token_id)
    attention_mask = pad_batch([c.attention_mask for c in conversations], 0)
    assistant_positions = [c.assistant_token_indices for c in conversations]
    labels = [c.label for c in conversations]
    return input_ids, attention_mask, assistant_positions, labels


def extract_last_assistant_activations(
    conversations: Sequence[TokenizedConversation],
    layer_index: int,
    batch_size: int,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    all_vectors: list[Tensor] = []
    all_labels: list[int] = []

    for start in tqdm(range(0, len(conversations), batch_size), desc="Extracting", leave=False):
        batch = conversations[start : start + batch_size]
        input_ids, attention_mask, assistant_positions, labels = collate_conversations(batch, tokenizer)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"), torch.no_grad():
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden = outputs.hidden_states[layer_index].to(torch.float32).cpu()

        for sample_idx, positions in enumerate(assistant_positions):
            if not positions:
                continue
            token_vecs = hidden[sample_idx, positions, :]
            pooled = token_vecs.mean(dim=0)
            all_vectors.append(pooled)
            all_labels.append(labels[sample_idx])

    if not all_vectors:
        raise ValueError("No activations extracted; check dataset filtering.")

    return torch.stack(all_vectors, dim=0), torch.tensor(all_labels, dtype=torch.long)
