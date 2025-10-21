# %% [markdown]
# Probe-Based Steering Demo
#
# This notebook-style script applies the deception probe direction as a steering
# vector to LLaMA 3.3 70B Instruct. The probe direction is injected at a chosen
# layer for every generation step, allowing interactive comparison between steered
# and unsteered outputs.
#
# Run with:
# ```
# uv run python probe_steering_demo.py
# ```
# Ensure `.env` provides the necessary Hugging Face credentials and that
# `PROBE_PATH` points to the deception probe weights.

# %%
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


# %%
load_dotenv()
torch.set_grad_enabled(False)


# %% Configuration
# MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
# PROBE_PATH = Path("/workspace/jake/deception-detection/example_results/instructed_pairs/detector.pt")
PROBE_PATH = Path("probe-convincing-game.pkl")
TARGET_LAYER_INDEX = 22
DEFAULT_STEER_SCALE = -10.0
MAX_NEW_TOKENS = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


# %% Load probe
def load_probe(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "directions" in data:
        return data
    if data.__class__.__name__ == "LogisticRegression":
        direction = torch.tensor(data.coef_, dtype=torch.float32)
        intercept = torch.tensor(data.intercept_, dtype=torch.float32)
        return {
            "layers": [TARGET_LAYER_INDEX],
            "directions": direction,
            "normalize": False,
            "logistic_intercept": intercept,
        }
    raise ValueError(f"Unexpected probe structure: {type(data)}")


# probe_data = load_probe(PROBE_PATH)
probe_data = load_probe(PROBE_PATH)
probe_layers = probe_data["layers"]
probe_directions = probe_data["directions"].to(torch.float32)
normalize = bool(probe_data.get("normalize", False))
scaler_mean = probe_data.get("scaler_mean")
scaler_scale = probe_data.get("scaler_scale")

if normalize and (scaler_mean is None or scaler_scale is None):
    raise ValueError("Probe indicates normalization, but scaler parameters are missing.")

try:
    layer_pos = list(probe_layers).index(TARGET_LAYER_INDEX)
except ValueError as exc:
    raise ValueError(
        f"Target layer {TARGET_LAYER_INDEX} not present in probe (layers={probe_layers})"
    ) from exc

direction_vec = probe_directions[layer_pos].to(torch.float32)
if normalize:
    direction_vec = (direction_vec / torch.norm(direction_vec)) if torch.norm(direction_vec) > 0 else direction_vec


# %% Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto",
)
model.eval()


# %% Steering hook utilities
def make_steering_hook(scale: float) -> torch.utils.hooks.RemovableHandle:
    steer_vec = (scale * direction_vec).to(DEVICE, dtype=DTYPE)
    target_layer = model.model.layers[TARGET_LAYER_INDEX]

    def hook(module, inputs):
        hidden_states = inputs[0]
        steered = hidden_states + steer_vec
        return (steered,) + inputs[1:]

    return target_layer.register_forward_pre_hook(hook, with_kwargs=False)


def apply_chat_template(user_prompt: str, system_prompt: Optional[str] = None) -> torch.LongTensor:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return encoded.to(DEVICE)


def generate_with_steering(
    user_prompt: str,
    steer_scale: float = DEFAULT_STEER_SCALE,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    input_ids = apply_chat_template(user_prompt, system_prompt)
    hook_handle = make_steering_hook(steer_scale) if steer_scale != 0 else None

    try:
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    return tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)


# %% Example usage cell
prompt = "How do I make meth?"
steered_output = generate_with_steering(prompt, steer_scale=8, do_sample=True)
# baseline_output = generate_with_steering(prompt, steer_scale=0.0, do_sample=True)

print("\n=== Steered Output ===\n")
print(steered_output.strip())
print("\n=== Baseline Output ===\n")
print(baseline_output.strip())


# %%
