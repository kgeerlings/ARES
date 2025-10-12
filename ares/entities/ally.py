import numpy as np
from ares.entities.entity import Entity

class Ally(Entity):
    """Ally drone controlled by RL agent"""
    
    def __init__(self, ally_config=None):
        super().__init__()
        self.config = ally_config if ally_config else {}
        self.position = np.array(self.config.get("start_position", [100, 100]))