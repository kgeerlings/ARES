import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import cv2
from ares.entities.ally import Ally
from ares.entities.enemy import Enemy
from ares.entities.target import Target
from ares.entities.entity import Entity
from ares.entities.base_area import BaseArea

class BaseEnv(gym.Env):
    """Base class for the environment."""

    def __init__(self, config: dict = None):
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

        # For reward shaping
        self.prev_distance = None
        self.prev_angle = None

        # Spaces
        self.action_space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32) # Two action (angle and velocity)
        self.observation_space = Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32) # Determine the shape according to the observations decided
        # For the first version, two observations: pos_to_target (dist), angle_to_target
        # For the second version, wee add three observations: bool_target_reached, pos_to_base, angle_to_base
        self.ally_reached_target = False
        # Maybe for the observations: pos_to_target, angle_to_target, bool_target_reached, pos_to_base, angle_to_base, pos_to_enemy_1, pos_to_enemy_2, pos_to_enemy_3

        # Entities
        self.ally = Ally(self.ally_config)
        self.enemies = [Enemy(self.enemy_config) for _ in range(self.env_config.get("num_enemies", 1))]
        self.target = Target(self.target_config)
        self.base_area = BaseArea(self.target_config, ally_init_position=self.ally.position)

    def _collides_with(self, entity_1: Entity, entity_2: Entity) -> bool:
        """
        Check if the entity_1 collides with the entity_2.

        Args:
            entity_1 (Entity): The first entity to check for collision.
            entity_2 (Entity): The second entity to check for collision.

        Returns:
            bool: True if the entity_1 collides with the entity_2, False otherwise.
        """
        agent_pos = entity_1.position
        target_pos = entity_2.position
        distance = np.linalg.norm(agent_pos - target_pos)
        return distance < (entity_1.radius + entity_2.radius)

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
        distance_max =  np.linalg.norm(np.array([self.width, self.height]))

        # === First model ===

        # Normalized distance between ally and target
        dist_to_target = np.linalg.norm(self.ally.position - self.target.position)
        normalized_dist_to_target = dist_to_target / distance_max

        # Angle between ally and target
        angle_to_target = np.arctan2(self.target.position[1] - self.ally.position[1],
                                      self.target.position[0] - self.ally.position[0])
        normalized_angle_to_target = (angle_to_target + np.pi) / (2 * np.pi)

        # === Second model ===

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

        # # If the agent is reaching the target, the distance and angle to the base are -1.0, if he reached the target, the distance and angle to the target are -1.0
        # if not self.ally_reached_target:
        #     normalized_dist_to_base = -1.0
        #     normalized_angle_to_base = -1.0
        # else:
        #     normalized_dist_to_target = -1.0
        #     normalized_angle_to_target = -1.0

        return np.array([normalized_dist_to_target, normalized_angle_to_target, self.ally_reached_target,
                         normalized_dist_to_base, normalized_angle_to_base], dtype=np.float32)

    def reset(self, seed: int = None):
        """
        Reset the environment.
        
        Args:
            seed (int, optional): The seed for the random number generator.
            
        Returns:
            observation (np.ndarray): The initial observation.
            info (dict): Additional information.
        """

        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        # Reset environment's entities
        self.ally.reset()
        for enemy in self.enemies:
            enemy.reset()
            if self._collides_with(enemy, self.target):
                enemy.reset()
        self.target.reset()
        self.base_area.reset(ally_init_position=self.ally.init_position)

        # Reset environment state
        self.truncated = False
        self.terminated = False
        self.n_steps = 0
        self.reward = 0
        self.prev_distance = None
        self.prev_angle = None
        self.ally_reached_target = False

        return self._get_observation(), {}

    def _is_truncated(self):
        """
        Check if the episode is truncated.

        Returns:
            bool: True if the episode is truncated, False otherwise.
        """
        self.truncated = self.n_steps >= self.max_n_steps
        return self.truncated

    def _is_terminated(self):
        """
        Check if the episode is terminated.

        Returns:
            bool: True if the episode is terminated, False otherwise.
        """
        self.terminated = self._collides_with(self.ally, self.base_area) and self.ally_reached_target
        return self.terminated

    def _is_done(self):
        """
        Check if the episode is done.

        Returns:
            bool: True if the episode is done, False otherwise.
        """
        self.done = self._is_terminated() or self._is_truncated()
        return self.done

    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (np.ndarray): The action to take.
            
        Returns:
            observation (np.ndarray): The next observation.
            reward (float): The reward for the action.
            terminated (bool): Whether the episode is terminated.
            truncated (bool): Whether the episode is truncated.
            info (dict): Additional information.
        """

        self.n_steps += 1

        # Check for done state first
        if self._is_done():
            return self._get_observation(), self.reward, self.terminated, self.truncated, {}
        
        # Let the agent take one step
        self.ally.step(action)
        self.reward = self._reward_shape()

        # Let the enemies take one step
        for enemy in self.enemies:
            enemy.step(self.ally.position)

        return self._get_observation(), self.reward, self.terminated, self.truncated, {}
    
    def close(self):
        """Close the environment."""
        cv2.destroyAllWindows()

    def render(self, mode: str = "rgb_array"):
        """
        Render the environment.

        Args:
            mode (str): The mode to render the environment in.
        """
        window_name = self.env_config.get("renderer_name", "Environment")
        cv2.namedWindow(window_name)
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
        # print(f"Step: {env.n_steps}, Action: {action}, Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")