# %% Imports
from dotenv import load_dotenv
import torch
from datasets import load_dataset
from transformers import AutoModel

# %% Load environment variables from .env file
load_dotenv()

# %% Load Liars Bench
alpaca = load_dataset("Cadenza-Labs/liars-bench", "alpaca", split="test")
convincing_game = load_dataset("Cadenza-Labs/liars-bench", "convincing-game", split="test")
hp_choice = load_dataset("Cadenza-Labs/liars-bench", "harm-pressure-choice", split="test")
hp_kr = load_dataset("Cadenza-Labs/liars-bench", "harm-pressure-knowledge-report", split="test")
insider_trading = load_dataset("Cadenza-Labs/liars-bench", "insider-trading", split="test")
instructed_deception = load_dataset("Cadenza-Labs/liars-bench", "instructed-deception", split="test")
soft_trigger = load_dataset("Cadenza-Labs/liars-bench", "soft-trigger", split="test")
print(alpaca)

# %%
insider_trading[0]


# %%
import ast
from collections import defaultdict

turn_names = defaultdict(int)
not_deception_counts = defaultdict(int)
deception_counts = defaultdict(int)

for row in insider_trading:
    # print(row)
    turn_name = ast.literal_eval(row['meta'].split('\n')[-1])['turn_name']
    turn_names[turn_name] += 1
    deception_counts[turn_name] += row['deceptive']
    not_deception_counts[turn_name] += (1 - row['deceptive'])

turn_names, deception_counts, not_deception_counts






# %%
import ast
from collections import defaultdict

turn_names = defaultdict(int)
last_row = None

for row in insider_trading:
    # print(row)
    turn_name = ast.literal_eval(row['meta'].split('\n')[-1])['turn_name']
    turn_names[turn_name] += 1
    if turn_name == "confirmation":
        print(row)
        break
    last_row = row

print(turn_names)
# %%
row['messages'][-6:]
# %%
last_row['messages'][-4:]
# %%
last_row

# %%
convincing_game[1]

# %%
from collections import defaultdict
hp_deceptive_counts = defaultdict(int)

for row in hp_choice:
    hp_deceptive_counts[row['deceptive']] += 1

print(hp_deceptive_counts)


# %%
hp_kr[0]

# %%
models = defaultdict(int)

for row in instructed_deception:
    models[row['model']] += 1

print(list(models.keys()))




# %%
llama = AutoModel.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", torch_dtype=torch.float16, device_map="auto")
# %%