import copy
import pickle
import time
from typing import Optional, Tuple

import numpy as np
import torch as th

from shrl.envs.base import ObstacleMaskWrapper
from shrl.learn.tclf import TwinControlLyapunovFunction


class LyapunovValueTable:
    def __init__(self,
                 tclf: Optional[TwinControlLyapunovFunction],
                 obs_lb: Optional[np.ndarray],
                 obs_ub: Optional[np.ndarray],
                 n_levels: int = 10,
                 pgd_max_iter: int = 100,
                 pgd_lr: float = 1e-3,
                 n_range_est_sample: int = 10,
                 n_radius_est_sample: int = 10,
                 bound_cnst: float = 1,
                 zero_lb=True):
        self.tclf = tclf
        self.n_levels = n_levels
        self.obs_lb = obs_lb
        self.obs_ub = obs_ub

        self.pgd_max_iter = pgd_max_iter
        self.pgd_lr = pgd_lr
        self.n_range_est_sample = n_range_est_sample
        self.n_radius_est_sample = n_radius_est_sample
        self.bound_cnst = bound_cnst
        self.zero_lb = zero_lb

        self.lyapunov_values = None
        self.lyapunov_radius = None

    def build(self):
        start = time.time()
        v_lb, v_ub = self._find_range()
        self.lyapunov_values = np.arange(v_lb, v_ub, (v_ub - v_lb) / self.n_levels)
        self.lyapunov_radius = self._compute_radius()
        end = time.time()

        print(f"building time: {end - start}s")

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump([self.lyapunov_values, self.lyapunov_radius, self.tclf], f)

    @classmethod
    def load(cls, path: str) -> 'LyapunovValueTable':
        table = cls(None, None, None)
        with open(path, "rb") as f:
            table.lyapunov_values, table.lyapunov_radius, table.tclf = pickle.load(f)
        return table

    def query(self, obs: np.ndarray) -> float:
        assert self.lyapunov_values is not None, "build the table first"
        lv = self.tclf.predict(obs)

        i = 0
        while i < len(self.lyapunov_values) - 1:
            if lv < self.lyapunov_values[i]:
                break
            i += 1

        return self.lyapunov_radius[i]

    def _find_range(self):
        v_lb_list, v_ub_list = [], []

        if not self.zero_lb:
            lb_loss = lambda obs: self.tclf.forward_lf(obs)
            for i in range(self.n_range_est_sample):
                x = np.random.uniform(self.obs_lb, self.obs_ub)
                v_lb_x = self._pgd(x, self.obs_lb, self.obs_ub, lb_loss)
                v_lb_list.append(self.tclf.predict(v_lb_x))
            v_lb = np.min(v_lb_list)
        else:
            v_lb = 0

        ub_loss = lambda obs: -self.tclf.forward_lf(obs)
        for i in range(self.n_range_est_sample):
            x = np.random.uniform(self.obs_lb, self.obs_ub)
            v_ub_x = self._pgd(x, self.obs_lb, self.obs_ub, ub_loss)
            v_ub_list.append(self.tclf.predict(v_ub_x))
        v_ub = np.max(v_ub_list)

        return v_lb, v_ub

    def _compute_radius(self):
        lyapunov_radius = []
        for lv in self.lyapunov_values:
            radius_loss = lambda x: -th.norm(x[:2]) + self.bound_cnst * th.abs(self.tclf.forward_lf(x) - lv)

            radius_list = []
            for _ in range(self.n_range_est_sample):
                init_x = np.random.uniform(self.obs_lb, self.obs_ub)
                radius_x = self._pgd(init_x, self.obs_lb, self.obs_ub, radius_loss)

                radius = np.linalg.norm(radius_x[:2])
                radius_list.append(radius)

            radius_array = np.sort(np.array(radius_list))
            lyapunov_radius.append(np.mean(radius_array[:2]))
        return lyapunov_radius

    def _pgd(self,
             x: np.ndarray,
             lb: np.ndarray,
             ub: np.ndarray,
             loss_fn) -> Optional[np.ndarray]:
        x = th.tensor(x, dtype=th.float32, device=self.tclf.device, requires_grad=True)
        lb = th.tensor(lb, dtype=th.float32, device=self.tclf.device)
        ub = th.tensor(ub, dtype=th.float32, device=self.tclf.device)

        optimizer = th.optim.Adam(params=[x], lr=self.pgd_lr)
        for _ in range(self.pgd_max_iter):
            loss = loss_fn(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            new_x = th.clamp(x, min=lb, max=ub)
            x.data = new_x.data

        return x.cpu().detach().numpy()


class Monitor:
    def __init__(self,
                 lv_table: LyapunovValueTable,
                 goal_dim: int = 2,
                 max_step_size: float = 0.2,
                 search_step_size: float = 0.01):
        self.lv_table = lv_table
        self.goal_dim = goal_dim

        self.current_goal = None
        self.prev_goal = None
        self.direction_vec = None
        self.unit_direction_vec = None
        self.mid_goal = None
        self.max_step_size = max_step_size
        self.search_step_size = search_step_size

    def reset(self):
        self.current_goal = None

    def select_subgoal(self,
                       env: ObstacleMaskWrapper,
                       gt: np.ndarray) -> Tuple[np.ndarray, int]:
        if self.current_goal is None:
            self.current_goal = env.robot_pos

        if (gt != self.current_goal).any():
            self.prev_goal = self.current_goal
            self.mid_goal = self.prev_goal
            self.direction_vec = gt - self.current_goal
            self.unit_direction_vec = self.direction_vec / np.linalg.norm(self.direction_vec)
            self.current_goal = gt

        disp, lyapunov_r = self.choose_step_size(env.get_obs(), env.hazards_pos, env.obstacle_radius, env.robot_radius)
        self.mid_goal = self.mid_goal + disp

        return self.mid_goal, lyapunov_r

    def choose_step_size(self, obs: np.ndarray,
                         hazards_pos: np.ndarray,
                         obs_radius: float = 0.15,
                         robot_radius: float = 0.3):
        step_size = self.max_step_size
        lyapunov_r = 0
        disp = 0
        obs_dist_min = 0

        if hazards_pos.shape[-1] != 2:
            hazards_pos = np.array(hazards_pos).reshape([-1, 3])[:, :2]

        while step_size > self.search_step_size:
            disp = self.unit_direction_vec * step_size
            disp = self.clip_disp(disp)

            query_obs = copy.deepcopy(obs)
            query_obs[:2] += disp

            lyapunov_r = max(self.lv_table.query(query_obs), np.linalg.norm(query_obs[:2]))
            obs_dist = np.linalg.norm(self.mid_goal + disp - hazards_pos, axis=-1)
            obs_dist_min = np.min(obs_dist) - obs_radius - robot_radius
            if obs_dist_min > lyapunov_r:
                break
            step_size -= self.search_step_size

        if step_size <= self.search_step_size:
            disp = self.search_step_size * self.unit_direction_vec
            disp = self.clip_disp(disp)
            lyapunov_r = np.linalg.norm(obs[:2] + disp)
            obs_dist_min = np.min(
                np.linalg.norm(self.mid_goal + disp - hazards_pos, axis=-1)) - obs_radius - robot_radius

        return disp, min(lyapunov_r + robot_radius, obs_dist_min + robot_radius)

    def clip_disp(self, disp):
        total_disp = self.mid_goal + disp - self.prev_goal
        if (np.abs(total_disp) > np.abs(self.direction_vec)).any():
            disp = self.current_goal - self.mid_goal

        return disp
