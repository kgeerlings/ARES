from ares.entities.drone import Drone

class Enemy(Drone):
    """PLACEHOLDER for scripted ennemy drone"""

    def __init__(self, enemy_config=None):
        super().__init__()
        self.config = enemy_config if enemy_config else {}
