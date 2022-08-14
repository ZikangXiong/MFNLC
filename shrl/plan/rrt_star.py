from typing import Callable, List, Tuple

import numpy as np

from shrl.plan.common.path import Path
from shrl.plan.rrt import RRT, Tree


class RRTStar(RRT):
    def search(self,
               max_iteration: int,
               heuristic: Callable[[np.ndarray], np.ndarray] = None,
               n_sample: int = 1,
               ucb_cnst: float = 10.0) -> Path:

        final_vertex = None
        for i in range(max_iteration):
            sampled_vertex = self.tree.sample(heuristic, n_sample)
            near_vertices = self.get_near_vertices(sampled_vertex, ucb_cnst)

            parent = self.tree.nearest_vertex(sampled_vertex)
            collision, cost = self._steer(parent, sampled_vertex)

            if not collision:
                sampled_vertex.cost = cost

                # choose parent and insert
                new_parent, new_cost = self.choose_parent(sampled_vertex, near_vertices)
                if new_parent is not None:
                    self.tree.insert_vertex(new_parent, sampled_vertex)
                    sampled_vertex.cost = new_cost
                else:
                    self.tree.insert_vertex(parent, sampled_vertex)

                self.rewire(sampled_vertex, near_vertices)

                if self._arrive(sampled_vertex):
                    final_vertex = sampled_vertex

        return self._get_path(final_vertex)

    def get_near_vertices(self,
                          sampled_vertex: Tree.Vertex,
                          ucb_cnst: float) -> List[Tree.Vertex]:
        n_nodes = len(self.tree.all_vertices)
        n_dims = len(self.search_space.ub)
        radius = ucb_cnst * (np.log(n_nodes) / n_nodes) ** (1 / n_dims)

        dist = np.linalg.norm(self.tree.all_vertices_state - sampled_vertex.state, axis=-1)
        indx = dist < radius

        near_vertices = [self.tree.all_vertices[i] for i, near in enumerate(indx) if near]
        return near_vertices

    def choose_parent(self,
                      sampled_vertex: Tree.Vertex,
                      near_vertices: List[Tree.Vertex]) -> Tuple[Tree.Vertex, float]:
        best_parent = None
        best_cost = sampled_vertex.cost

        for parent in near_vertices:
            collision, new_cost = self._steer(parent, sampled_vertex)
            if not collision and new_cost < best_cost:
                best_parent = parent
                best_cost = best_cost

        return best_parent, best_cost

    def rewire(self,
               sampled_vertex: Tree.Vertex,
               near_vertices: List[Tree.Vertex]):

        for child in near_vertices:
            collision, new_cost = self._steer(sampled_vertex, child)
            cost_diff = new_cost - child.cost

            # cost_diff < 0 means that it needs to rewire
            if not collision and cost_diff < 0:
                # only need to add to children of sampled_vertex,
                # do not call insert function in Tree
                sampled_vertex.children.append(child)

                # remove child from its original parent
                child.parent.children.remove(child)

                # set new parent
                child.parent = sampled_vertex

                # update children's costs
                queue = [child]
                while queue:
                    child = queue.pop(0)
                    child.cost += cost_diff
                    queue.extend(child.children)
