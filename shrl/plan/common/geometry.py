from abc import ABC

import numpy as np


class ObjectBase(ABC):
    def __init__(self, state: np.ndarray):
        self.state = state


class Circle(ObjectBase):
    def __init__(self, state: np.ndarray, radius: float):
        super(Circle, self).__init__(state)
        self.radius = radius
