from gymnasium.spaces import Box
import numpy as np
import cv2
from ares.entities.base_area import BaseArea
from ares.environment.base_env import BaseEnv


class GoAndReturnEnvWrapper(BaseEnv):
    """Environment wrapper for the Go and Return task."""

    def __init__(self, config: dict = None):
        super().__init__(config)

        # Change the observation space shape to add new observations
        self.observation_space = Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        # We add a boolean to know if the ally has reached the target
        self.ally_reached_target = False

        # New entity
        self.base_area = BaseArea(self.target_config, ally_init_position=self.ally.position)

    def _reward_shape_reaching_target(self):
        """Define the shape of the reward when the ally tries to reach the target."""

        # Distance difference
        dist_to_target = np.linalg.norm(self.ally.position - self.target.position)
        distance_difference = self.prev_distance - dist_to_target if self.prev_distance is not None else 0
        self.prev_distance = dist_to_target

        distance_max = np.linalg.norm(np.array([self.width, self.height]))

        normalized_distance_difference = distance_difference / distance_max

        return normalized_distance_difference
    
    def _reward_shape_returning_to_base(self):
        """Define the shape of the reward when the ally tries to return to the base area."""

        # Distance difference
        dist_to_base = np.linalg.norm(self.ally.position - self.base_area.position)
        distance_difference = self.prev_distance - dist_to_base if self.prev_distance is not None else 0
        self.prev_distance = dist_to_base

        distance_max = np.linalg.norm(np.array([self.width, self.height]))

        normalized_distance_difference = distance_difference / distance_max

        return normalized_distance_difference
    
    def _reward_shape(self):
        """Define the shape of the reward based on the current state."""
        if self.ally_reached_target:
            return self._reward_shape_returning_to_base()
        else:
            return self._reward_shape_reaching_target()
        
    
    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
            np.ndarray: The current observation.
        """

        # Get the BaseEnv's observation
        base_obs = super()._get_observation()

        distance_max =  np.linalg.norm(np.array([self.width, self.height]))

        # === New observations ===

        # Boolean indicating if the ally has reached the target
        if not self.ally_reached_target and self._collides_with(self.ally, self.target):
            self.ally_reached_target = True

        # Normalized distance between ally and base area
        dist_to_base = np.linalg.norm(self.ally.position - self.base_area.position)
        normalized_dist_to_base = dist_to_base / distance_max

        # Angle between ally and base area
        angle_to_base = np.arctan2(self.base_area.position[1] - self.ally.position[1],
                                     self.base_area.position[0] - self.ally.position[0])
        normalized_angle_to_base = (angle_to_base + np.pi) / (2 * np.pi)

        return np.concatenate([base_obs, np.array([self.ally_reached_target,
                                              normalized_dist_to_base,
                                              normalized_angle_to_base], dtype=np.float32)])


    def reset(self, seed=None):
        """Reset the environment to the initial state."""
        observation, info = super().reset(seed=seed)

        self.base_area.reset(ally_init_position=self.ally.init_position)

        return observation, info
    

    def _is_terminated(self):
        """
        Check if the episode is terminated.

        Returns:
            bool: True if the episode is terminated, False otherwise.
        """
        self.terminated = self._collides_with(self.ally, self.base_area) and self.ally_reached_target
        return self.terminated

    def step(self, action):
        """Take a step in the environment."""
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = "rgb_array"):
        super().render(mode)

        window_name = self.env_config.get("renderer_name", "Environment")

        window = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        self.ally.render(window)
        for enemy in self.enemies:
            enemy.render(window)
        self.target.render(window)
        self.base_area.render(window)

        cv2.imshow(window_name, window)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.close()


if __name__ == "__main__":
    from config.config import config
    env = BaseEnv(config)
    obs, info = env.reset()
    print("Initial observation:", obs)
    done = False
    env.render()
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
