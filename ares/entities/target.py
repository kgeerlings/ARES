import cv2
import numpy as np
from ares.entities.entity import Entity

class Target(Entity):
    def __init__(self, target_config=None):
        self.config = target_config if target_config else {}
        self.position = np.array(self.config.get("position", [1175, 375]))
        self.radius = self.config.get("radius")
        self.color = self.config.get("color", [0, 255, 0])

    def reset(self):
        self.position = np.array(self.config.get("position", [1175, 375]))

    def render(self, window):
        """
        Render the target in the given window.
        
        Args:
            window: The window to render the target in.
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
        contour_color = (0, 255, 0) 
        thickness = 2
        cv2.circle(window, (int(self.position[0]), int(self.position[1])), self.radius, contour_color, thickness)

