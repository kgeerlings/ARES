import gymnasium as gym
from gymnasium import Box
from ares.entities.ally import Ally
from ares.entities.enemy import Enemy
from ares.entities.target import Target

class BaseEnv(gym.Env):
    """Base class for the environment."""

    def __init__(self, config=None):
        super().__init__()
        
        # Config
        self.env_config = config.get("env_config", {}) if config else {}
        self.ally_config = config.get("ally_config", {}) if config else {}
        self.enemy_config = config.get("enemy_config", {}) if config else {}
        self.target_config = config.get("target_config", {}) if config else {}

        # Map
        self.width = self.env_config.get("width", 1280)
        self.height = self.env_config.get("height", 720)
        self.background_color = self.env_config.get("background_color", [255, 255, 255])

        # Environment state
        self.truncated = False
        self.terminated = False
        self.done = self.terminated or self.truncated
        self.n_steps = 0
        self.max_n_steps = 1000
        self.reward = 0

        # Spaces
        self.action_space = Box(low=0.0, high=1.0, shape=(2,), dtype=float) # Two action (angle and velocity)
        # self.observation_space = Box(low=-1.0, high=1.0, shape=(8?,), dtype=float) # Determine the shape according to the observations decided
        # Maybe for the observations: pos_to_target, angle_to_target, bool_target_reached, pos_to_base, angle_to_base, pos_to_enemy_1, pos_to_enemy_2, pos_to_enemy_3

        # Entities
        self.ally = Ally(self.ally_config)
        self.enemies = [Enemy(self.enemy_config) for _ in range(self.env_config.get("num_enemies", 1))]
        self.target = Target(self.target_config)
        

    def _reward_shape(self):
        """Define the shape of the reward."""
        # PLACEHOLDER
        pass

    def _get_observation(self):
        """Get the current observation."""
        # PLACEHOLDER
        pass

    def step(self, action):
        """Take a step in the environment."""
        # PLACEHOLDER
        pass

    def reset(self):
        """Reset the environment."""
        # PLACEHOLDER
        pass

    def render(self, mode="rgbarray"):
        """Render the environment."""
        # PLACEHOLDER
        pass