import matplotlib.pyplot as plt
import numpy as np

from shrl.plan.common.geometry import Circle
from shrl.plan.common.path import Path
from shrl.plan.common.space import SearchSpace
from shrl.plan.rrt import Tree


def plot_path_2d(space: SearchSpace,
                 path: Path,
                 tree: Tree = None):
    assert space.ub.shape == (2,)

    fig, ax = plt.subplots(figsize=[5, 5])

    # map
    ax.set_xlim(space.lb[0], space.ub[0])
    ax.set_ylim(space.lb[1], space.ub[1])

    # initial and goal state
    initial = plt.Circle(tuple(space.initial_state),
                         radius=0.05, color="g", alpha=0.5)
    ax.add_patch(initial)
    ax.scatter(*space.goal_state, s=500, marker="*", color="gold", alpha=0.5)

    # obstacle
    if isinstance(space.obstacles[0], Circle):
        obstacles = [plt.Circle(tuple(obs.state), obs.radius,  # noqa
                                color="b", alpha=0.5) for obs in space.obstacles]
    else:
        raise NotImplementedError()

    for obs in obstacles:
        ax.add_patch(obs)

    # tree
    if tree is not None:
        queue = [tree.root]
        while queue:
            parent = queue.pop(0)
            for child in parent.children:
                line = np.array([parent.state, child.state])
                ax.plot(line[:, 0], line[:, 1], marker="x", color="k")
                queue.append(child)

    # path
    if len(path) > 0:
        ax.plot(path[:, 0], path[:, 1], marker="x", color="r")

    plt.show()
