import cv2
from gymnasium.spaces import Box
import numpy as np
from ares.environment.go_and_return_env_wrapper import GoAndReturnEnvWrapper


class DodgingEnemiesEnvWrapper(GoAndReturnEnvWrapper):
    """Environment wrapper for the Dodging Enemies task."""

    def __init__(self, config: dict = None):
        super().__init__(config)

        # For reward shaping
        self.prev_distance_to_target = None
        self.prev_distance_to_base = None
        self.prev_angle = None
        self.target_reward_reached = 0
        self.base_reward_reached = 0

        # Change the observation space shape to add new observations
        self.observation_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    def _reward_shape_reaching_target(self):
        """Define the shape of the reward when the ally tries to reach the target."""

        return (super()._reward_shape_reaching_target() * 10)
    
    def _reward_shape_returning_to_base(self):
        """Define the shape of the reward when the ally tries to return to the base area."""

        return (super()._reward_shape_returning_to_base() * 10)
    
    def _reward_shape_for_enemies_avoidance(self):
        """
        Define the shape of the reward for avoiding enemies.
        If an enemy is closer to the ally than 300 pixels, the reward is negative.
        The farther the enemy is, the higher the reward.
        """
        reward = 0
        for enemy in self.enemies:
            dist_to_enemy = np.linalg.norm(self.ally.position - enemy.position)
            if dist_to_enemy < 50:
                reward -= ((50 - dist_to_enemy) / 50) * 0.06  # Negative reward when too close
            if dist_to_enemy < self.ally.radius + enemy.radius:
                reward -= 3  # Collision penalty
        return reward / len(self.enemies) # Average reward over all enemies
    

    def _reward_shape(self):
        """Define the shape of the reward based on the current state."""

        reaching_target_reward = 0
        reaching_base_reward = 0
        time_penalty = -0.03  # Small negative reward at each step to encourage faster completion

        if self.target_reward_reached==0 and self._collides_with(self.ally, self.target):
            reaching_target_reward += 5
            self.target_reward_reached = 1
        if self.base_reward_reached==0 and self.ally_reached_target and self._collides_with(self.ally, self.base_area):
            reaching_base_reward += 5
            self.base_reward_reached = 1

        if self.ally_reached_target:
            return self._reward_shape_returning_to_base()+self._reward_shape_for_enemies_avoidance()+reaching_base_reward+time_penalty
        else:
            return self._reward_shape_reaching_target()+self._reward_shape_for_enemies_avoidance()+reaching_target_reward+time_penalty


    def _get_observation(self):
        """
        Get the current observation.

        Returns:
            np.ndarray: The current observation.
        """

        # Get the GoAndReturnEnvWrapper's observation
        go_and_return_obs = super()._get_observation()

        # Enemies observations
        enemies_obs = []
        distance_max =  np.linalg.norm(np.array([self.width, self.height]))
        for enemy in self.enemies:
            dist_to_enemy = np.linalg.norm(self.ally.position - enemy.position)
            normalized_dist_to_enemy = dist_to_enemy / distance_max

            # angle_to_enemy = np.arctan2(enemy.position[1] - self.ally.position[1],
            #                             enemy.position[0] - self.ally.position[0])
            # normalized_angle_to_enemy = (angle_to_enemy + np.pi) / (2 * np.pi)

            enemies_obs.extend([normalized_dist_to_enemy])#, normalized_angle_to_enemy])

        return np.concatenate([go_and_return_obs, np.array(enemies_obs, dtype=np.float32)])


    def reset(self, seed: int = None):
        obs, info = super().reset(seed=seed)

        self.prev_distance_to_target = None
        self.prev_distance_to_base = None

        return obs, info
    
    def render(self, mode: str = "human"):
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

        if mode == "rgb_array":
            # Convert BGR (OpenCV) to RGB and return the array
            rgb_array = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
            return rgb_array
        elif mode == "human":
            # Display the window for human viewing
            cv2.namedWindow(window_name)
            cv2.imshow(window_name, window)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.close()
            return None
        else:
            # Default behavior (for backward compatibility)
            cv2.namedWindow(window_name)
            cv2.imshow(window_name, window)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.close()
            return None