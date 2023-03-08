from typing import List

import numpy as np

from mfnlc.plan.common.geometry import ObjectBase, Circle


def collision_dist(traj_array, obstacles_array, safe_dist) -> np.ndarray:
    init = traj_array[0]
    end = traj_array[-1]
    lb = np.min([init, end], axis=0)
    ub = np.max([init, end], axis=0)

    dists = np.full(obstacles_array.shape[0], np.inf)
    safe_obj_indx = np.logical_or((obstacles_array < lb - safe_dist).any(axis=-1),
                                  (obstacles_array > ub + safe_dist).any(axis=-1))
    checking_obj_indx = np.logical_not(safe_obj_indx)

    dists[checking_obj_indx] = np.abs(
        np.cross(end - init, obstacles_array[checking_obj_indx] - init)) \
                               / np.linalg.norm(end - init)

    return dists


class CollisionChecker:
    @staticmethod
    def overlap(obj_1: ObjectBase, obj_2: ObjectBase) -> bool:
        if isinstance(obj_1, Circle):
            if isinstance(obj_2, Circle):
                return np.linalg.norm(obj_1.state - obj_2.state, ord=2) <= obj_1.radius + obj_2.radius

    @staticmethod
    def seq_to_seq_overlap(traj: List[ObjectBase],
                           obstacles: List[ObjectBase]):
        traj_array = np.array([obj.state for obj in traj])
        obstacles_array = np.array([obj.state for obj in obstacles])

        if isinstance(traj[0], Circle):
            if isinstance(obstacles[0], Circle):
                safe_dist = traj[0].radius + obstacles[0].radius  # noqa
                dists = collision_dist(traj_array, obstacles_array, safe_dist)
                return (dists < safe_dist).any()
