import json
import os

import yaml

from src.constants.constants import DATA_FOLDER


def create_new_dataset(input_path, output_path):
    # Load existing data
    with open(input_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    new_data = []
    for item in data:
        new_data.append(item["user_message"])

    # Save new data as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2)


if __name__ == "__main__":
    input_yaml_path = os.path.join(DATA_FOLDER, "compassion", "conversation_data_tommie.yaml")
    output_json_path = os.path.join(DATA_FOLDER, "compassion", "compassion_train_user_questions.json")
    create_new_dataset(input_yaml_path, output_json_path)
