from ares.entities.drone import Drone

class Ally(Drone):
    """PLACEHOLDER for ally drone controlled by RL agent"""
    
    def __init__(self, ally_config=None):
        super().__init__()
        self.config = ally_config if ally_config else {}