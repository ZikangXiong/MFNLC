import numpy as np
from typing import Union

from shrl.config import env_config
from shrl.envs import Continuous2DNav
from shrl.envs.base import SafetyGymBase


def choose_level(env,
                 level: int,
                 fix_init_and_goal: bool = True):
    robot_name = env.robot_name
    difficulty_config = env_config[robot_name]["difficulty"][level]
    floor_lb, floor_ub = np.array(difficulty_config[1], dtype=np.float32)

    env: Union[Continuous2DNav, SafetyGymBase] = env.unwrapped
    if isinstance(env, Continuous2DNav):
        env.update_env_config({
            "obstacle_num": difficulty_config[0],
            "floor_lb": floor_lb,
            "floor_ub": floor_ub,
        })

        if fix_init_and_goal:
            env.fixed_init_and_goal = True
            init = 0.9 * floor_lb
            goal = 0.9 * floor_ub
            env.update_env_config({
                "init": init,
                "goal": goal
            })
    elif issubclass(type(env), SafetyGymBase):
        env.update_env_config({
            "hazards_num": difficulty_config[0],
            "placements_extents": np.concatenate([floor_lb, floor_ub]).tolist(),
            "hazards_keepout": 0.45
        })

        if fix_init_and_goal:
            env.fixed_init_and_goal = True
            init = 0.9 * floor_lb
            goal = np.array([0.9, 0.8]) * floor_ub
            env.update_env_config({
                "robot_locations": [init.tolist()],
                "goal_locations": [goal.tolist()]
            })
    else:
        raise NotImplementedError()
