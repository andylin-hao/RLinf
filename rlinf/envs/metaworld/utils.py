import json

def load_prompt_from_json(json_path, env_name):
    with open(json_path, "r") as f:
        prompt_data = json.load(f)
    return prompt_data.get(env_name, "")
