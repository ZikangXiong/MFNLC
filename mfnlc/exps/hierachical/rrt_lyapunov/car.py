import numpy as np

from mfnlc.exps.hierachical.rrt_lyapunov.base import evaluate, build_lyapunov_table
from mfnlc.exps.check_results import print_all_results

ENV_NAME = "Car-eval"


def rrt_lyapunov(planning_algo):
    for i in range(1, 4):
        print(f"{ENV_NAME} - RRT + Lyapunov-TD3 - level {i}")
        evaluate(ENV_NAME,
                 n_rollout=1,
                 level=i,
                 planning_algo=planning_algo,
                 planner_max_iter=i * i * 1000,
                 n_steps=1000 * i * i,
                 arrive_radius=0.3,
                 monitor_max_step_size=0.5,
                 check_plan=False,
                 render=False,
                 render_config={
                     "traj_sample_freq": 10,
                     "follow": False,
                     "vertical": True,
                     "scale": 7 * i
                 },
                 video=True,
                 seed=1)


def build_lv_table():
    lb = np.array([-1, -1, -1, -1, 9.8] + [-1] * 21)
    ub = np.array([1, 1, 1, 1, 9.81] + [1] * 21)
    build_lyapunov_table(ENV_NAME,
                         lb, ub,
                         pgd_max_iter=1000,
                         n_radius_est_sample=20)


if __name__ == '__main__':
    # build_lv_table()
    rrt_lyapunov("rrt*")
    print_all_results(ENV_NAME, "rrt_lyapunov", "rrt*")
    input("Press Enter to continue...")
