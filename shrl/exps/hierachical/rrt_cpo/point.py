from shrl.exps.hierachical.rrt_cpo.base import evaluate
from shrl.exps.check_results import print_all_results

ENV_NAME = "Point-eval"


def rrt_cpo(planning_algo):
    for i in range(1, 4):
        print(f"{ENV_NAME} - RRT + CPO - level {i}")
        evaluate(ENV_NAME,
                 n_rollout=100,
                 level=i,
                 planning_algo=planning_algo,
                 planner_max_iter=i * i * 200,
                 n_steps=1000 * i * i,
                 arrive_radius=0.3)


if __name__ == '__main__':
    # rrt_cpo("rrt")
    print_all_results(ENV_NAME, "rrt_cpo", "rrt")
