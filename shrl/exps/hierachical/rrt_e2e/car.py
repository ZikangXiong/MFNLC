from shrl.exps.hierachical.rrt_e2e.base import evaluate
from shrl.exps.check_results import print_all_results

ENV_NAME = "Car-eval"


def rrt_e2e(planning_algo):
    for i in range(1, 4):
        print(f"{ENV_NAME} - RRT + TD3 - level {i}")
        evaluate(ENV_NAME,
                 n_rollout=1,
                 level=i,
                 planning_algo=planning_algo,
                 planner_max_iter=i * i * 500,
                 n_steps=1000 * i * i,
                 arrive_radius=0.3,
                 render=True,
                 check_plan=True)


if __name__ == '__main__':
    rrt_e2e("rrt")
    # print_all_results(ENV_NAME, "rrt_e2e", "rrt")
