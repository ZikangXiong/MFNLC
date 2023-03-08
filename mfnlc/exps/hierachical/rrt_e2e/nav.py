from mfnlc.exps.hierachical.rrt_e2e.base import evaluate
from mfnlc.exps.check_results import print_all_results

ENV_NAME = "Nav-eval"


def rrt_e2e(planning_algo):
    for i in range(1, 4):
        print(f"{ENV_NAME} - RRT + TD3 - level {i}")
        evaluate(ENV_NAME,
                 n_rollout=100,
                 level=i,
                 planning_algo=planning_algo,
                 planner_max_iter=i * i * 200,
                 n_steps=1000 * i * i,
                 arrive_radius=0.1)


if __name__ == '__main__':
    # rrt_e2e("rrt")
    print_all_results(ENV_NAME, "rrt_e2e", "rrt")
