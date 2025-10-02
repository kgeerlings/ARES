import gymnasium as gym

class BaseEnv(gym.Env):
    """Base class for the environment."""

    def __init__(self):
        super().__init__()