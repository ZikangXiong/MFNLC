from typing import List

import numpy as np


class Path:
    def __init__(self, vertex_state_list: List[np.ndarray]):
        self.path_array = np.array(vertex_state_list)

    def __getitem__(self, item):
        return self.path_array[item]

    def __len__(self):
        return len(self.path_array)
