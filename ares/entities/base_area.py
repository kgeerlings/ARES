import cv2
import numpy as np
from ares.entities.entity import Entity

class BaseArea(Entity):
    def __init__(self, base_area_config=None, ally_init_position=None):
        self.config = base_area_config if base_area_config else {}
        # self.position = np.array(self.config.get("position", [1175, 375]))
        # ally initial position
        self.position = ally_init_position if ally_init_position is not None else np.array([
            np.random.uniform(100, 1175),
            np.random.uniform(100, 650)
        ], dtype=np.float32)
        self.radius = self.config.get("radius")
        self.color = [15, 15, 15]

    def reset(self, ally_init_position=None):
        # self.position = np.array(self.config.get("position", [1175, 375]))
        self.position = ally_init_position if ally_init_position is not None else np.array([
            np.random.uniform(100, 1175),
            np.random.uniform(100, 650)
        ], dtype=np.float32)
    def render(self, window):
        """
        Render the base_area in the given window.
        
        Args:
            window: The window to render the base_area in.
        """
        overlay = window.copy()
        cv2.circle(
            window,
            (int(self.position[0]), int(self.position[1])),
            self.radius,
            self.color,
            -1,  # Filled circle
        )
        alpha = 0.7  
        cv2.addWeighted(overlay, alpha, window, 1 - alpha, 0, window)

        # Contour
        contour_color = (0, 0, 0) 
        thickness = 2
        cv2.circle(window, (int(self.position[0]), int(self.position[1])), self.radius, contour_color, thickness)

