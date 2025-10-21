from ares.environment.base_env import BaseEnv
from config.config import config
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs import check_env_specs, ParallelEnv

n_obstacles = config["obstacles"]["n_obstacles"]


# Go to target environment
def create_env():
    base_env = BaseEnv(config=config)
    return GymWrapper(base_env)


env = ParallelEnv(10, create_env, mp_start_method="fork")

# print(check_env_specs(env))
# print(env.observation_spec)
# print(env.full_action_spec)
