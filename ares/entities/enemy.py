from ares.entities.entity import Entity

class Enemy(Entity):
    """PLACEHOLDER for scripted ennemy drone"""

    def __init__(self, enemy_config=None):
        super().__init__()
        self.config = enemy_config if enemy_config else {}
        
