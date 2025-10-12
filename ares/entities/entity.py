import cv2
import numpy as np


class Entity:

    def __init__(self, config: dict = None):
        self.position = np.array([0,0])
        self.radius = config.get("radius", 10) if config else 10
        self.color = config.get("color", [0, 0, 255]) if config else [0, 0, 255]

    def reset(self):
        self.position = np.array([np.random.randint(0, 1280), np.random.randint(0, 720)])

    def render(self, window):
        """
        Render the entity in the given window.
        
        Args:
            window: The window to render the entity in.
        """
        cv2.circle(
            window,
            (int(self.position[0]), int(self.position[1])),
            self.radius,
            self.color,
            3,
        )
