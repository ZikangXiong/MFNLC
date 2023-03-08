import matplotlib.pyplot as plt
import numpy as np

from mfnlc.envs import get_env
from mfnlc.envs.difficulty import choose_level
from mfnlc.plan import Planner
from mfnlc.plan.common.geometry import Circle
from mfnlc.plan.common.path import Path
from mfnlc.plan.common.plot import plot_path_2d
from mfnlc.plan.common.space import SearchSpace
from mfnlc.plan.rrt import RRT
from mfnlc.plan.rrt_star import RRTStar


def build_space():
    obstacle_states = np.array([
        [0, 0],
        [0.5, 0.7],
        [-0.7, -0.5],
        [0.5, 0.5],
        [-0.5, -0.5],
        [0.5, -0.5],
        [0.7, -0.2]
    ])
    obstacles = [Circle(state, 0.1) for state in obstacle_states]
    space = SearchSpace(
        lb=np.array([-1, -1]),
        ub=np.array([1, 1]),
        initial_state=np.array([-0.9, -0.9]),
        goal_state=np.array([0.9, 0.9]),
        obstacles=obstacles
    )

    return space


def test_plot2d():
    space = build_space()

    path = Path([
        np.array([-0.9, -0.9]),
        np.array([-0.9, -0.5]),
        np.array([-0.5, 0.2]),
        np.array([0.5, 0.3]),
        np.array([0.9, 0.9]),
    ])

    plot_path_2d(space, path)


def test_rrt():
    space = build_space()
    rrt = RRT(space,
              robot=Circle(np.array([0.9, 0.9]), radius=0.1),
              arrive_radius=0.1,
              collision_checker_resolution=0.01)
    path = rrt.search(500)
    print(path.path_array)
    plot_path_2d(space, path, rrt.tree)


def test_rrt_with_heuristic():
    space = build_space()
    goal = np.array([0.9, 0.9])
    rrt = RRT(space,
              robot=Circle(goal, radius=0.1),
              arrive_radius=0.1,
              collision_checker_resolution=0.01)

    heuristic = lambda x: np.linalg.norm(x - goal)
    n_sample = 1000
    path = rrt.search(500, heuristic=heuristic, n_sample=n_sample)

    print(path.path_array)
    plot_path_2d(space, path, rrt.tree)


def test_rrt_star():
    space = build_space()
    goal = np.array([0.9, 0.9])
    rrt = RRTStar(space,
                  robot=Circle(goal, radius=0.1),
                  arrive_radius=0.1,
                  collision_checker_resolution=0.01)

    path = rrt.search(200, ucb_cnst=5)

    print(path.path_array)
    plot_path_2d(space, path, rrt.tree)


def test_rrt_star_with_heuristic():
    space = build_space()
    goal = np.array([0.9, 0.9])

    rrt = RRTStar(space,
                  robot=Circle(goal, radius=0.1),
                  arrive_radius=0.1,
                  collision_checker_resolution=0.01)

    heuristic = lambda x: np.linalg.norm(x - goal)
    n_sample = 1000
    path = rrt.search(200, heuristic, n_sample)

    print(path.path_array)
    plot_path_2d(space, path, rrt.tree)


def test_build_from_nav_env():
    env = get_env("Nav")
    env.reset()
    env.render()

    search_space = SearchSpace.build_from_env(env)
    plot_path_2d(search_space, Path([]), None)


def test_build_from_safety_gym_env():
    env = get_env("Point")
    env.reset()
    image = env.render(mode="rgb_array", camera_id=1)
    plt.imshow(image)

    search_space = SearchSpace.build_from_env(env)
    plot_path_2d(search_space, Path([]), None)


def test_nav_planner():
    env = get_env("Nav")
    env.reset()
    env.render()

    planner = Planner(env, "rrt*")
    path = planner.plan(300)
    tree = planner.algo.tree
    plot_path_2d(planner.algo.search_space, path, tree)

    print(path.path_array)


def test_safety_gym_planner():
    env = get_env("Point")
    env.reset()
    image = env.render(mode="rgb_array", camera_id=1)
    plt.imshow(image)
    plt.show()

    planner = Planner(env, "rrt*")
    path = planner.plan(200)
    tree = planner.algo.tree
    plot_path_2d(planner.algo.search_space, path, tree)

    print(path.path_array)


def test_plot_plan():
    for level in range(2, 3):
        env = get_env("Doggo-eval")
        choose_level(env, level)
        env.reset()
        planner = Planner(env, "rrt*")
        path = planner.plan(1000)
        env.unwrapped.subgoal_list = [subgoal for subgoal in path]
        env.render(mode="rgb_array", camera_id=1, width=4096, height=4096)
        image = env.render(mode="rgb_array", camera_id=1, width=4096, height=4096)

        plt.figure(figsize=[20, 20])
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(f"./quadruped-lv-{level}.png", bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    import time

    start = time.time()

    # test_plot2d()
    # test_rrt()
    # test_rrt_with_heuristic()
    # test_rrt_star()
    # test_rrt_star_with_heuristic()
    # test_build_from_nav_env()
    # test_build_from_safety_gym_env()
    # test_safety_gym_planner()
    # test_nav_planner()
    # test_plot_plan()
    # test_gen_traj_and_plan()

    end = time.time()
    print("run time: ", end - start)
