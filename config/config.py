import json

with open("config/config.json") as f:
    config = json.load(f)

env_config = config["env_config"]
ally_config = config["ally_config"]
ennemy_config = config["ennemy_config"]
