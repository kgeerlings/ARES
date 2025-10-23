from ares.environment.base_env import BaseEnv
from config.config import config
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs import ParallelEnv


# Go to target environment
def create_env():
    base_env = BaseEnv(config=config)
    return GymWrapper(base_env)


env = ParallelEnv(10, create_env, mp_start_method="fork")
