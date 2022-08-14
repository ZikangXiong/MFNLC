import numpy as np

from shrl.exps.hierachical.rrt_lyapunov.base import evaluate, build_lyapunov_table
from shrl.exps.check_results import print_all_results

ENV_NAME = "Nav-eval"


def rrt_lyapunov(planning_algo):
    for i in range(1, 4):
        print(f"{ENV_NAME} - RRT + Lyapunov-TD3 - level {i}")
        evaluate(ENV_NAME,
                 n_rollout=1,
                 level=i,
                 planning_algo=planning_algo,
                 planning_algo_kwargs={
                     "support_margin": 0.05
                 },
                 render_config={
                     "traj_sample_freq": 10,
                 },
                 monitor_max_step_size=0.1,
                 monitor_search_step_size=5e-3,
                 planner_max_iter=i * i * 500,
                 n_steps=1000 * i * i,
                 arrive_radius=0.05,
                 render=False,
                 seed=0,
                 video=True)


def build_lv_table():
    lb = np.array([-1, -1])
    ub = np.array([1, 1])
    build_lyapunov_table(ENV_NAME,
                         lb, ub,
                         pgd_max_iter=500,
                         n_radius_est_sample=20)


if __name__ == '__main__':
    # build_lv_table()
    rrt_lyapunov("rrt")
    print_all_results(ENV_NAME, "rrt_lyapunov", "rrt")
