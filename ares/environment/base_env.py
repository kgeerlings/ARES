import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import cv2
from ares.entities.ally import Ally
from ares.entities.enemy import Enemy
from ares.entities.target import Target
from ares.entities.entity import Entity

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

        # Spaces
        self.action_space = Box(low=0.0, high=1.0, shape=(2,), dtype=float) # Two action (angle and velocity)
        self.observation_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=float) # Determine the shape according to the observations decided
        # For the first version, two observations: pos_to_target (dist), angle_to_target
        # Maybe for the observations: pos_to_target, angle_to_target, bool_target_reached, pos_to_base, angle_to_base, pos_to_enemy_1, pos_to_enemy_2, pos_to_enemy_3

        # Entities
        self.ally = Ally(self.ally_config)
        self.enemies = [Enemy(self.enemy_config) for _ in range(self.env_config.get("num_enemies", 1))]
        self.target = Target(self.target_config)


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
    

    # def _replace_if_collision(self, entity_1: Entity, entity_2: Entity) -> None:
    #     """
    #     Replace the entity_2 if it collides with the entity_1.

    #     Args:
    #         entity_1 (Entity): The entity to check for collision.
    #         entity_2 (Entity): The entity to replace if a collision occurs.
    #     """
    #     if self._collides_with(entity_1, entity_2):
    #         entity_2.position = np.random.uniform(low=0, high=[self.width, self.height])
        

    def _reward_shape(self):
        """Define the shape of the reward."""
        # PLACEHOLDER
        pass

    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
            np.ndarray: The current observation.
        """

        # Normalized distance between ally and target
        dist_to_target = np.linalg.norm(self.ally.position - self.target.position)
        distance_max =  np.linalg.norm(np.array([self.width, self.height]))
        normalized_dist_to_target = dist_to_target / distance_max

        # Angle between ally and target
        angle_to_target = np.arctan2(self.target.position[1] - self.ally.position[1],
                                      self.target.position[0] - self.ally.position[0])
        normalized_angle_to_target = (angle_to_target + np.pi) / (2 * np.pi)

        return np.array([normalized_dist_to_target, normalized_angle_to_target], dtype=np.float32)
    

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
        self.target.reset()

        # Reset environment state
        self.truncated = False
        self.terminated = False
        self.n_steps = 0
        self.reward = 0

        # RESET REWARD UTILS (dist_to_target, etc...)

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
        self.terminated = self._collides_with(self.agent, self.target)
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

        cv2.imshow(window_name, window)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            self.close()

    


if __name__ == "__main__":
    env = BaseEnv()
    obs, info = env.reset()
    print("Initial observation:", obs)
    done = False
    env.render()
    # while not done:
    #     action = env.action_space.sample()  # Random action
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated
    #     print(f"Step: {env.n_steps}, Action: {action}, Observation: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")