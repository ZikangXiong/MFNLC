import os

import pandas as pd

from mfnlc.config import get_path
from mfnlc.envs import get_env
from mfnlc.envs.difficulty import choose_level
from mfnlc.evaluation.model import load_model
from mfnlc.evaluation.simulation import simu

ALGO = "e2e"


def evaluate(env_name,
             n_rollout: int = 1,
             n_steps: int = 1000,
             level: int = 1,
             arrive_radius: float = 0.3,
             render: bool = False):
    model = load_model(env_name, algo=ALGO)
    env = get_env(env_name)
    choose_level(env, level)

    i = 0
    running_data = {
        "total_step": [],
        "goal_met": [],
        "collision": [],
        "reward_sum": []
    }
    while i < n_rollout:
        env.reset()
        res = simu(env=env,
                   model=model,
                   n_steps=n_steps,
                   path=None,
                   arrive_radius=arrive_radius,
                   render=render)
        for k in res:
            running_data[k].append(res[k])
        i += 1

    stat = pd.DataFrame(running_data)

    res_dir = get_path(robot_name=env.robot_name, algo=ALGO, task="evaluation")
    os.makedirs(res_dir, exist_ok=True)
    stat.to_csv(res_dir + f"/{level}.csv")
    print("results are saved to:", res_dir + f"/{level}.csv")

    print(stat)
    return stat
