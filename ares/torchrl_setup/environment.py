from ares.main.global_variables import GlobalVariables
from ares.environment.base_env import BaseEnv
from ares.environment.go_and_return_env_wrapper import GoAndReturnEnvWrapper
from ares.environment.dodging_enemies_env_wrapper import DodgingEnemiesEnvWrapper
from config.config import config, config_model_1, config_model_2, config_model_3
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs import ParallelEnv


# Go to target environment
def create_env():
    if GlobalVariables.CONFIG==0:
        base_env = DodgingEnemiesEnvWrapper(config=config)
        return GymWrapper(base_env)
    if GlobalVariables.CONFIG==1:
        base_env = BaseEnv(config=config_model_1)
        return GymWrapper(base_env)
    elif GlobalVariables.CONFIG==2:
        second_env = GoAndReturnEnvWrapper(config=config_model_2)
        return GymWrapper(second_env)
    elif GlobalVariables.CONFIG==3:
        third_env = DodgingEnemiesEnvWrapper(config=config_model_3)
        return GymWrapper(third_env)
    else:
        raise ValueError(f"Unknown environment choice: {GlobalVariables.CONFIG}")


env = ParallelEnv(10, create_env(), mp_start_method="fork")
