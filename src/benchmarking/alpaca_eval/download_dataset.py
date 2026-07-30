import json
import os

from huggingface_hub import hf_hub_download

from src.constants import constants

path = hf_hub_download(
    repo_id="tatsu-lab/alpaca_eval",
    filename="alpaca_eval.json",
    repo_type="dataset",
)

with open(path) as f:
    dataset = json.load(f)

print(f"Downloaded {len(dataset)} examples")

output_dir = os.path.join(constants.DATA_FOLDER, "alpaca_eval")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "alpaca_eval.json")

with open(output_path, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Saved to {output_path}")