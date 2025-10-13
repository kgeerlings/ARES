import numpy as np
from ares.entities.entity import Entity

class Ally(Entity):
    """Ally drone controlled by RL agent"""
    
    def __init__(self, ally_config=None):
        super().__init__()
        self.config = ally_config if ally_config else {}
        self.position = np.array(self.config.get("start_position", [100, 100]))
        self.color = self.config.get("color", [0, 255, 0])

    def reset(self):
        self.position = np.array(self.config.get("start_position", [100, 100]))

    def step(self, action):
        """
        Update the ally's position based on the action.

        Args:
            action (np.ndarray): Array containing angle (radians) and velocity.
        """
        # angle, velocity = action
        # direction = np.array([np.cos(angle), np.sin(angle)])
        # self.position += direction * velocity
        pass
        

