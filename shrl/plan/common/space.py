from typing import List

import numpy as np
from safety_gym.envs.engine import Engine

from shrl.envs import Continuous2DNav
from shrl.envs.base import SafetyGymBase
from shrl.plan.common.geometry import ObjectBase, Circle


class SearchSpace:
    def __init__(self,
                 lb: np.ndarray,
                 ub: np.ndarray,
                 initial_state: np.ndarray,
                 goal_state: np.ndarray,
                 obstacles: List[ObjectBase]):
        self.lb = lb
        self.ub = ub
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.obstacles = obstacles

    @classmethod
    def build_from_env(cls, env) -> 'SearchSpace':
        if isinstance(env.unwrapped, Continuous2DNav):
            env: Continuous2DNav = env.unwrapped

            lb = env.floor_lb
            ub = env.floor_ub

            initial_state = env.robot_pos
            goal_state = env.goal

            obstacles_centers = env.obstacle_centers
            obstacle_radius = env.obstacle_radius
            obstacles = [Circle(obst, radius=obstacle_radius) for obst in obstacles_centers]

        elif issubclass(type(env.unwrapped), SafetyGymBase):
            env: Engine = env.unwrapped.env

            floor_extends = env.placements_extents
            lb = np.array(floor_extends[:2])
            ub = np.array(floor_extends[2:])

            initial_state = env.robot_pos[:2]
            goal_state = env.goal_pos[:2]

            obstacles_centers = env.hazards_pos
            obstacle_radius = env.hazards_size
            obstacles = [Circle(obst[:2], radius=obstacle_radius) for obst in obstacles_centers]
        else:
            raise NotImplementedError

        return SearchSpace(lb, ub, initial_state, goal_state, obstacles)

    def sample(self, n_samples: int) -> np.ndarray:
        return np.random.uniform(self.lb, self.ub, (n_samples,) + self.lb.shape)
