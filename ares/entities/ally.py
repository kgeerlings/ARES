import numpy as np
from ares.entities.entity import Entity

class Ally(Entity):
    """Ally drone controlled by RL agent"""
    
    def __init__(self, ally_config=None):
        super().__init__()
        self.config = ally_config if ally_config else {}
        self.init_position = np.array(self.config.get("initial_position", [100, 100]), dtype=np.float32)
        # random position
        # self.init_position = np.array([
        #     np.random.uniform(100, 1175),
        #     np.random.uniform(100, 650)
        # ], dtype=np.float32)
        self.position = self.init_position.copy()
        self.color = self.config.get("color", [0, 255, 0])

    def reset(self):
        self.init_position = np.array(self.config.get("initial_position", [100, 100]), dtype=np.float32)
        # self.init_position = np.array([
        #     np.random.uniform(100, 1175),
        #     np.random.uniform(100, 650)
        # ], dtype=np.float32)
        self.position = self.init_position.copy()

    def step(self, action):
        """
        Update the ally's position based on the action.

        Args:
            action (np.ndarray): Array containing angle (radians) and velocity.
        """
        action = np.asarray(action, dtype=np.float32)
        angle, velocity = action

        angle = angle * 2 * np.pi  # Convert normalized angle [0,1] to radians [0, 2π]
        velocity = velocity * 10

        dx = velocity * np.cos(angle)
        dy = velocity * np.sin(angle)
        self.position += np.array([dx, dy], dtype=np.float32)
        


