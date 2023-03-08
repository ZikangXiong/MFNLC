import os

from gym.wrappers import TimeLimit
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from mfnlc.config import env_config
from .base import EnvBase
from .car import CarNav
from .doggo import DoggoNav
from .nav import Continuous2DNav
from .point import PointNav


def get_env(env_name: str):
    robot_name = env_name.split("-")[0]

    # Training envs for controllers that consider obstacles
    if env_name == "Nav":
        env = TimeLimit(Continuous2DNav(), env_config[robot_name]["max_step"])
    elif env_name == "Point":
        env = TimeLimit(PointNav(), env_config[robot_name]["max_step"])
    elif env_name == "Car":
        env = TimeLimit(CarNav(), env_config[robot_name]["max_step"])
    elif env_name == "Doggo":
        env = TimeLimit(DoggoNav(), env_config[robot_name]["max_step"])
    # Training envs for controllers that do not consider obstacles
    elif env_name == "Nav-no-obst":
        env = TimeLimit(Continuous2DNav(no_obstacle=True), env_config[robot_name]["max_step"])
    elif env_name == "Point-no-obst":
        env = TimeLimit(PointNav(no_obstacle=True), env_config[robot_name]["max_step"])
    elif env_name == "Car-no-obst":
        env = TimeLimit(CarNav(no_obstacle=True), env_config[robot_name]["max_step"])
    elif env_name == "Doggo-no-obst":
        env = TimeLimit(DoggoNav(no_obstacle=True), env_config[robot_name]["max_step"])
    # evaluation envs
    elif env_name == "Nav-eval":
        env = Continuous2DNav(end_on_collision=True)
    elif env_name == "Point-eval":
        env = PointNav(end_on_collision=True)
    elif env_name == "Car-eval":
        env = CarNav(end_on_collision=True)
    elif env_name == "Doggo-eval":
        env = DoggoNav(end_on_collision=True)
    else:
        raise NotImplementedError(f"Unsupported environment - {env_name}")

    return env


def get_sub_proc_env(env_name: str, n_envs: int):
    env = make_vec_env(env_id=get_env,  # noqa
                       env_kwargs={"env_name": env_name},
                       vec_env_cls=SubprocVecEnv,
                       seed=os.getpid(),
                       n_envs=n_envs)

    return env
