import cv2
import numpy as np
from ares.entities.entity import Entity

class Target(Entity):
    def __init__(self, target_config=None):
        self.config = target_config if target_config else {}
        self.position = np.array(self.config.get("position", [1175, 375]))
        self.radius = self.config.get("radius", 15)
        self.color = self.config.get("color", [0, 0, 255])