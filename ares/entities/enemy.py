import numpy as np
from ares.entities.entity import Entity

class Enemy(Entity):
    """Class for scripted enemy drone"""

    def __init__(self, enemy_config=None):
        super().__init__()
        self.config = enemy_config if enemy_config else {}
        self.color = self.config.get("color", [0, 0, 255])
        self.position = np.array([0, 0])
        self.initial_position = np.array([0, 0])
        self.speed = self.config.get("speed", 2)

    def reset(self):
        """Place the enemy at a random position at the right side of the map."""
        self.position = np.array([np.random.randint(600, self.config.get("width", 1280)-150), np.random.randint(150, self.config.get("height", 720)-150)])
        self.initial_position = self.position.copy()

    def _detect_ally(self, ally_position: np.ndarray) -> bool:
        """Detect the position of the ally in his vision range."""
        if np.linalg.norm(self.position - ally_position) < self.config.get("vision_range", 400):
            return True
        return False
    
    def _move_angle(self, angle: float):
        angle_rad = np.deg2rad(angle)
        dx = self.speed * np.cos(angle_rad)
        dy = self.speed * np.sin(angle_rad)
        self.position[0] += dx
        self.position[1] += dy

    def step(self, ally_position: np.ndarray):
        """Scripted enemy behavior"""

        if self._detect_ally(ally_position):
            # Move towards the ally
            ally_x, ally_y = ally_position[0], ally_position[1]
            dx = ally_x - self.position[0]
            dy = ally_y - self.position[1]
            angle = np.degrees(np.arctan2(dy, dx))
            # Normalize angle to 0-360 range
            if angle < 0:
                angle += 360
            self._move_angle(angle)
        else:
            # # Patrol around initial position
            # if np.linalg.norm(self.position - self.initial_position) > 400:
            #     angle = np.degrees(np.arctan2(self.initial_position[1] - self.position[1],
            #                                   self.initial_position[0] - self.position[0]))
            #     if angle < 0:
            #         angle += 360
            #     self._move_angle(angle)
            # else:
            #     angle = np.random.uniform(0, 360)
            #     self._move_angle(angle)

            # --- patrouille circulaire fluide ---
            if not hasattr(self, "_patrol_angle"):
                self._patrol_angle = np.random.uniform(0, 360)
            if not hasattr(self, "_patrol_dir"):
                self._patrol_dir = np.random.choice([-1, 1])

            # rayon et vitesse angulaire
            patrol_radius = 100
            angular_speed = 1  # degrés par frame environ

            # mise à jour de l’angle
            self._patrol_angle = (self._patrol_angle + self._patrol_dir * angular_speed) % 360

            # calcul de la position sur le cercle
            new_x = self.initial_position[0] + patrol_radius * np.cos(np.radians(self._patrol_angle))
            new_y = self.initial_position[1] + patrol_radius * np.sin(np.radians(self._patrol_angle))

            # déplacer l’ennemi directement vers cette position
            self.position = np.array([new_x, new_y])