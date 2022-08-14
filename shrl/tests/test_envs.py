import time

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

from shrl.envs import get_sub_proc_env, get_env, Continuous2DNav, PointNav
from shrl.envs.difficulty import choose_level
from shrl.evaluation.model import load_model


def test_sub_proc_env():
    env_names = ["Nav-no-obst", "Point-no-obst", "Car-no-obst", "Doggo-no-obst"]
    for env_name in env_names:
        env = get_sub_proc_env(env_name, 4)
        obs = env.reset()
        assert len(obs) == 4
        assert isinstance(env, SubprocVecEnv)
        print(env)
        print(obs)


def test_make_env():
    env_names = ["Point", "Car", "Doggo", "Point-no-obst", "Car-no-obst", "Doggo-no-obst"]
    for env_name in env_names:
        get_env(env_name)


def test_spaces():
    env_names = ["Point", "Car", "Doggo", "Point-no-obst", "Car-no-obst", "Doggo-no-obst"]
    observation_shape = {
        "Point": (18,),
        "Car": (30,),
        "Doggo": (62,),
        "Point-no-obst": (14,),
        "Car-no-obst": (26,),
        "Doggo-no-obst": (58,)
    }
    action_shape = {
        "Point": (2,),
        "Car": (2,),
        "Doggo": (12,),
        "Point-no-obst": (2,),
        "Car-no-obst": (2,),
        "Doggo-no-obst": (12,)
    }

    for env_name in env_names:
        env = get_env(env_name)
        obs = env.reset()
        print(f"---- {env_name} ----")
        for k in env.unwrapped.env.obs().keys():
            _obs = env.unwrapped.env.obs()[k]
            print(f"{k}: {_obs}")

        assert obs.shape == observation_shape[env_name]
        assert env.action_space.shape == action_shape[env_name]


def test_step():
    env_names = ["Point", "Car", "Doggo", "Point-no-obst", "Car-no-obst", "Doggo-no-obst"]
    for env_name in env_names:
        env = get_env(env_name)
        obs = env.reset()
        obs_shape = obs.shape
        for _ in range(10):
            obs, rew, done, info = env.step(env.action_space.sample())
            assert obs.shape == obs_shape
            assert "collision" in info.keys()
            assert "cost" in info.keys()


def check_sensors():
    env_names = ["Point-no-obst", "Car-no-obst", "Doggo-no-obst"]
    for env_name in env_names:
        print(f"====={env_name}=====")
        env = get_env(env_name)
        env.reset()
        obs_dict = env.env.env.obs()
        for k in sorted(obs_dict.keys()):
            if "lidar" not in k:
                print(f"{k}: {obs_dict[k]}")


def choose_sink():
    env_names = ["Point-no-obst", "Car-no-obst", "Doggo-no-obst"]
    for env_name in env_names:
        env = get_env(env_name)
        obs = env.reset()
        for _ in range(1000):
            obs, rew, done, info = env.step(np.zeros_like(env.action_space.low))

        obs[:2] = 0
        print(f"{env_name}'s sink: {np.round(obs, 2).tolist()}")


def test_render():
    env_names = ["Point-no-obst", "Car-no-obst", "Doggo-no-obst"]
    for env_name in env_names:
        env = get_env(env_name)
        env.reset()
        env.render()
        for _ in range(1000):
            env.step(np.zeros_like(env.action_space.low))
            env.render(mode="human")


def test_reward():
    env_names = [
        # "Point-no-obst",
        # "Car-no-obst",
        "Doggo-no-obst"]
    for env_name in env_names:
        env = get_env(env_name)
        env.reset()
        for _ in range(100):
            _, r, _, _ = env.step(np.random.random_sample(env.action_space.high.shape) - 0.5)
            print(r)
            env.render()
            time.sleep(0.1)


def test_update_env_config():
    env_name = "Point"
    env = get_env(env_name)

    env.reset()
    env.render()
    for _ in range(1000):
        env.step(np.zeros_like(env.action_space.low))
        env.render()

    # map size
    env.update_env_config({"placements_extents": [-10, -10, 10, 10]})
    env.reset()
    env.render()
    for _ in range(1000):
        env.step(np.zeros_like(env.action_space.low))
        env.render()

    # obstacle number
    env.update_env_config({"placements_extents": [-5, -5, 5, 5],
                           "hazards_num": 100})
    env.reset()
    env.render()
    for _ in range(1000):
        env.step(np.zeros_like(env.action_space.low))
        env.render()


def test_nav_subgoal():
    env = get_env("Nav-no-obst")
    model = load_model("Nav-no-obst", "td3")

    env.reset()
    env.unwrapped.robot_pos = np.array([-0.9, -0.9])
    env.unwrapped.goal = np.array([0.9, 0.9])
    subgoals = np.linspace(env.unwrapped.robot_pos, env.unwrapped.goal, 20, endpoint=True)
    obs = env.get_obs()
    env.render()

    for i in range(200):
        if i % 10 == 0:
            env.unwrapped.set_subgoal(subgoals[i // 10])
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)
        env.render()


def test_point_subgoal():
    env = get_env("Point-no-obst").unwrapped
    model = load_model("Point-no-obst", "td3")

    init = np.array([-4, -4])
    goal = np.array([4, 4])
    env.update_env_config({
        "num_steps": 1000,
        "placements_extents": [-5, -5, 5, 5],
        "robot_locations": [init],
        "goal_locations": [goal],
        "continue_goal": True,
        "hazards_num": 0})
    obs = env.reset()
    subgoals = np.linspace(init, goal, 20, endpoint=True)
    env.render()

    for i in range(1000):
        if i % 50 == 0:
            env.unwrapped.set_subgoal(subgoals[i // 50])
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        if done:
            break
        env.render()


def test_nav():
    env = Continuous2DNav()
    env.reset()
    env.render()

    env = Continuous2DNav(fixed_init_and_goal=True)
    env.reset()
    env.render()
    env.reset()
    env.render()


def test_nav_choose_level(robot_name):
    if robot_name == "Nav":
        env = Continuous2DNav(fixed_init_and_goal=True)
    elif robot_name == "Point":
        env = PointNav(fixed_init_and_goal=True)
    else:
        raise NotImplementedError()

    choose_level(env, 1)
    env.reset()
    env.render()
    time.sleep(5)

    choose_level(env, 2)
    env.reset()
    env.render()
    time.sleep(5)

    choose_level(env, 3)
    env.reset()
    env.render()
    time.sleep(5)


def test_render_with_rgb_array():
    for level in range(1, 2):
        env = get_env("Doggo-eval")
        choose_level(env, level)
        env.reset()
        image = env.render(mode="rgb_array", camera_id=1, width=8000, height=8000)

        plt.figure(figsize=[100, 100])
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(f"./quadruped-lv-{level}.png", bbox_inches='tight', pad_inches=0)


def test_nav_rgb_array():
    for level in range(1, 4):
        env = get_env("Nav-eval")
        choose_level(env, level)
        env.reset()
        image = env.render(mode="rgb_array")

        plt.imshow(image)
        plt.show()


def test_fix_hazardous_zone():
    env = get_env("Doggo-eval")
    choose_level(env, 2)
    env.reset()
    hazards_pos = env.env.hazards_pos
    print(hazards_pos)
    env.update_env_config({"hazards_locations": [pos[:2] for pos in hazards_pos]})
    env.reset()
    print(hazards_pos)
    np.save("loc.npy", [pos[:2] for pos in hazards_pos])
    env.render()
    input()


if __name__ == '__main__':
    # test_make_env()
    # test_spaces()
    # test_step()
    # check_sensors()
    # choose_sink()
    # test_render()
    # test_reward()
    # test_sub_proc_env()
    # test_update_env_config()
    # test_nav_subgoal()
    # test_point_subgoal()
    # test_nav()
    # test_nav_choose_level("Nav")
    # test_nav_choose_level("Point")
    # test_render_with_rgb_array()
    test_nav_rgb_array()
    # test_fix_hazardous_zone()
