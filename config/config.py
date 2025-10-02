import json

with open("config/config.json") as f:
    config = json.load(f)

env_config = config["env_config"]
ally_config = config["ally_config"]
enemy_config = config["enemy_config"]
target_config = config["target_config"]