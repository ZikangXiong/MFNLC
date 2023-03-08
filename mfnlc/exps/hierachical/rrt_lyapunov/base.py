import os
import random
from typing import Dict

import numpy as np
import pandas as pd
from gym.wrappers.monitor import Monitor as VideoMonitor

from mfnlc.config import get_path
from mfnlc.envs import get_env
from mfnlc.envs.base import ObstacleMaskWrapper
from mfnlc.envs.difficulty import choose_level
from mfnlc.evaluation.model import load_model
from mfnlc.evaluation.simulation import simu
from mfnlc.learn.lyapunov_td3 import LyapunovTD3
from mfnlc.monitor.monitor import Monitor, LyapunovValueTable
from mfnlc.plan import Planner
from mfnlc.plan.common.plot import plot_path_2d

ALGO = "rrt_lyapunov"


def evaluate(env_name,
             n_rollout: int = 1,
             n_steps: int = 1000,
             level: int = 1,
             planning_algo: str = "rrt*",
             planner_max_iter: int = 200,
             planning_algo_kwargs: Dict = {},  # noqa
             arrive_radius: float = 0.1,
             monitor_max_step_size: float = 0.2,
             monitor_search_step_size: float = 0.01,
             render: bool = False,
             check_plan: bool = False,
             video: bool = False,
             render_config: Dict = {},  # noqa
             seed: int = None
             ):
    model: LyapunovTD3 = load_model(env_name, algo=ALGO)
    env = ObstacleMaskWrapper(get_env(env_name))

    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    choose_level(env, level)
    if video:
        video_path = get_path(robot_name=env.robot_name,
                              algo=ALGO, task="video") + f"{planning_algo}-level-{level}"
        env = VideoMonitor(env, video_path, force=True)
    planner = Planner(env, planning_algo)
    lv_table = LyapunovValueTable.load(get_path(env.robot_name, ALGO, "lv_table"))
    monitor = Monitor(lv_table, max_step_size=monitor_max_step_size, search_step_size=monitor_search_step_size)

    i = 0
    running_data = {
        "total_step": [],
        "goal_met": [],
        "collision": [],
        "reward_sum": []
    }

    re_plan = False
    while i < n_rollout:
        if not re_plan:
            # reset video monitor
            env.reset()
        else:
            env.unwrapped.reset()
        path = planner.plan(planner_max_iter, **planning_algo_kwargs)
        if check_plan:
            plot_path_2d(planner.algo.search_space, path, planner.algo.tree)
        if len(path) == 0:
            re_plan = True
            continue
        re_plan = False

        res = simu(env=env,
                   model=model,
                   n_steps=n_steps,
                   path=path,
                   arrive_radius=arrive_radius,
                   monitor=monitor,
                   render=render,
                   render_config=render_config)
        for k in res:
            running_data[k].append(res[k])
        i += 1

    stat = pd.DataFrame(running_data)

    res_dir = get_path(robot_name=env.robot_name, algo=ALGO, task="evaluation") + f"/{planning_algo}"
    os.makedirs(res_dir, exist_ok=True)
    stat.to_csv(res_dir + f"/{level}.csv")
    print("results are saved to:", res_dir + f"/{level}.csv")

    env.close()
    return stat, env.env.traj


def build_lyapunov_table(env_name: str,
                         obs_lb: np.ndarray,
                         obs_ub: np.ndarray,
                         n_levels: int = 10,
                         pgd_max_iter: int = 100,
                         pgd_lr: float = 1e-3,
                         n_range_est_sample: int = 10,
                         n_radius_est_sample: int = 10,
                         bound_cnst: float = 100):
    model: LyapunovTD3 = load_model(env_name, algo=ALGO)
    lv_table = LyapunovValueTable(model.tclf,
                                  obs_lb,
                                  obs_ub,
                                  n_levels=n_levels,
                                  pgd_max_iter=pgd_max_iter,
                                  pgd_lr=pgd_lr,
                                  n_range_est_sample=n_range_est_sample,
                                  n_radius_est_sample=n_radius_est_sample,
                                  bound_cnst=bound_cnst)
    lv_table.build()
    print(lv_table.lyapunov_values)
    print(lv_table.lyapunov_radius)
    robot_name = env_name.split("-")[0]
    lv_table.save(get_path(robot_name, algo=ALGO, task="lv_table"))
