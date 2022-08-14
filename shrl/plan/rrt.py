import copy
from typing import Callable, Union, Tuple

import numpy as np

from shrl.plan.common.collision import CollisionChecker
from shrl.plan.common.geometry import ObjectBase
from shrl.plan.common.path import Path
from shrl.plan.common.space import SearchSpace

# perform better when obstacle number is large
ENABLE_SEQ_TO_SEQ_COLLISION_CHECKING = False


class Tree:
    class Vertex:
        def __init__(self, state: np.ndarray, parent=None):
            self.state = state
            self.parent = parent
            self.children = []
            self.cost = 0.0

    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space
        self.root = Tree.Vertex(search_space.initial_state)
        self.all_vertices = [self.root]
        self.all_vertices_state = [self.root.state]

    def reset(self, search_space: SearchSpace = None):
        if search_space is not None:
            self.search_space = search_space
        self.root = Tree.Vertex(self.search_space.initial_state)
        self.all_vertices = [self.root]
        self.all_vertices_state = [self.root.state]

    def sample(self,
               heuristic: Callable[[np.ndarray], np.ndarray] = None,
               n_sample: int = 1) -> 'Tree.Vertex':
        samples = self.search_space.sample(n_sample)

        if heuristic is not None:
            scores = heuristic(samples)
            best_index = np.argmax(scores)
            return Tree.Vertex(samples[best_index])
        else:
            assert n_sample == 1

        return Tree.Vertex(samples[0])

    def nearest_vertex(self, sampled_vertex: 'Tree.Vertex') -> 'Tree.Vertex':
        dist = np.linalg.norm(sampled_vertex.state - np.array(self.all_vertices_state), axis=-1)
        nearest_index = np.argmin(dist)

        return self.all_vertices[nearest_index]

    def insert_vertex(self, parent: 'Tree.Vertex', vertex: 'Tree.Vertex'):
        self.all_vertices.append(vertex)
        self.all_vertices_state.append(vertex.state)
        vertex.parent = parent
        parent.children.append(vertex)


class RRT:
    def __init__(self,
                 search_space: SearchSpace,
                 robot: ObjectBase,
                 arrive_radius: Union[np.ndarray, float],
                 collision_checker_resolution: float):

        self.collision_checker = CollisionChecker()
        self.tree = Tree(search_space)

        self.search_space = search_space
        self.robot = robot
        self.arrive_radius = arrive_radius
        self.collision_checker_resolution = collision_checker_resolution

    def set_search_space(self, search_space: SearchSpace):
        self.search_space = search_space
        self.tree.reset(search_space)

    def search(self,
               max_iteration: int,
               heuristic: Callable[[np.ndarray], np.ndarray] = None,
               n_sample: int = 1) -> Path:

        final_vertex = None
        for i in range(max_iteration):
            sampled_vertex = self.tree.sample(heuristic, n_sample)
            parent = self.tree.nearest_vertex(sampled_vertex)
            collision, cost = self._steer(parent, sampled_vertex)
            if not collision:
                sampled_vertex.cost = cost
                self.tree.insert_vertex(parent, sampled_vertex)
                if self._arrive(sampled_vertex):
                    final_vertex = sampled_vertex
                    break

        return self._get_path(final_vertex)

    def _steer(self,
               parent: Tree.Vertex,
               vertex: Tree.Vertex) -> Tuple[bool, float]:
        n_mid_state = np.abs(parent.state - vertex.state).max() // self.collision_checker_resolution + 1

        for state in np.linspace(parent.state, vertex.state, int(n_mid_state), endpoint=True):
            self.robot.state = state
            for obstacle in self.search_space.obstacles:
                if self.collision_checker.overlap(self.robot, obstacle):
                    return True, np.inf

        if ENABLE_SEQ_TO_SEQ_COLLISION_CHECKING:
            init = copy.deepcopy(self.robot)
            end = copy.deepcopy(self.robot)
            init.state = parent.state
            end.state = vertex.state
            traj = [init, end]

            if self.collision_checker.seq_to_seq_overlap(traj, self.search_space.obstacles):
                return True, np.inf

        # euclidian cost
        cost = parent.cost + self._default_dist(parent, vertex)

        return False, cost

    def _arrive(self, vertex: Tree.Vertex) -> bool:
        return (np.abs(vertex.state - self.search_space.goal_state) < self.arrive_radius).all()

    def _get_path(self, final_vertex: Tree.Vertex) -> Path:
        if final_vertex is not None:
            print(f"cost: {final_vertex.cost}")
            _vertex = final_vertex
            path = [self.search_space.goal_state]
            while _vertex:
                path.append(_vertex.state)
                _vertex = _vertex.parent

            return Path(list(reversed(path)))

        print("Warning: No path found")
        return Path([])

    @staticmethod
    def _default_dist(start_vertex: Tree.Vertex,
                      end_vertex: Tree.Vertex):
        # cost must be positive
        return np.linalg.norm(end_vertex.state - start_vertex.state)
