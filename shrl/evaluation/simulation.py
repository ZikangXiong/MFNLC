from typing import Dict

import numpy as np

from shrl.config import env_config
from shrl.envs import get_env
from shrl.evaluation.model import load_model
from shrl.monitor.monitor import Monitor
from shrl.plan.common.path import Path


def inspect_training_simu(env_name: str,
                          algo: str,
                          n_rollout: int,
                          render: bool = False):
    env = get_env(env_name)
    robot_name = env_name.split("-")[0]

    model = load_model(env_name, algo)

    simu_data = {
        "rewards": [],
        "obs": [],
        "actions": [],
        "infos": []
    }

    for ep in range(n_rollout):
        reward_list = []
        obs_list = []
        action_list = []
        info_list = []

        obs = env.reset()
        obs_list.append(obs)
        for i in range(env_config[robot_name]["max_step"]):
            action = model.predict(obs)[0]
            obs, reward, done, info = env.step(action)

            action_list.append(action)
            obs_list.append(obs)
            reward_list.append(reward)
            info_list.append(info)

            if render:
                env.render()

            if done:
                break

        simu_data["obs"].append(obs_list)
        simu_data["actions"].append(action_list)
        simu_data["rewards"].append(reward_list)
        simu_data["infos"].append(info_list)

    return simu_data


def simu(env,
         model,
         n_steps: int,
         path: Path = None,
         arrive_radius: float = 0.0,
         monitor: Monitor = None,
         render: bool = False,
         render_config: Dict = {}  # noqa
         ):
    obs = env.get_obs()

    if monitor is not None:
        monitor.reset()

    subgoal_index = 0
    if path is not None:
        env.set_subgoal(path[subgoal_index])

    env.set_render_config(render_config)
    if render:
        env.render()

    total_step = 0
    goal_met = False
    collision = False
    reward_sum = 0.0

    for i in range(n_steps):
        action = model.predict(obs)[0]
        obs, reward, done, info = env.step(action)
        total_step += 1
        reward_sum += reward

        if path is not None:
            if np.linalg.norm(env.robot_pos - path[subgoal_index]) < arrive_radius:
                subgoal_index += 1
                subgoal_index = min(len(path) - 1, subgoal_index)
                subgoal = path[subgoal_index]
                env.set_subgoal(subgoal, store=True)
            else:
                subgoal = path[subgoal_index]

            if monitor is not None:
                subgoal, lyapunov_r = monitor.select_subgoal(env, subgoal)
            env.set_subgoal(subgoal, store=False)
            env.set_roa(subgoal, lyapunov_r)  # noqa

        if render:
            if path is not None and monitor is not None:
                env.render()
            else:
                env.render()

        if done:
            goal_met = info.get("goal_met", False)
            collision = info.get("collision", False)
            break

    return {"total_step": total_step,
            "collision": collision,
            "goal_met": goal_met,
            "reward_sum": reward_sum}
