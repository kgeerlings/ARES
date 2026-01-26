import json

with open("config/config.json") as f:
    config = json.load(f)

env_config = config["env_config"]
ally_config = config["ally_config"]
enemy_config = config["enemy_config"]
target_config = config["target_config"]


with open("config/config_model_1.json") as f:
    config_model_1 = json.load(f)

with open("config/config_model_2.json") as f:
    config_model_2 = json.load(f)

with open("config/config_model_3.json") as f:
    config_model_3 = json.load(f)