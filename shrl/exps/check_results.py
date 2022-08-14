import pandas as pd

from shrl.evaluation.data import reach_perc, steps_to_reach, safety_vio


def results_stat(env_name: str,
                 algo: str,
                 level: int,
                 planning_algo: str):
    reach_perc_data = reach_perc(env_name, algo, planning_algo, level)
    steps_to_reach_data = steps_to_reach(env_name, algo, planning_algo, level)
    safety_vio_data = safety_vio(env_name, algo, planning_algo, level)

    stat = {
        "Level": [level],
        "Reach Perc.": [reach_perc_data],
        "# Reach Step": ["{0:.2f} ± {1:.2f}".format(*steps_to_reach_data)],
        "Safety Vio.": [safety_vio_data]
    }

    return pd.DataFrame(stat)


def print_all_results(env_name, algo, planning_algo=None):
    level_res = [results_stat(env_name, algo, i, planning_algo) for i in range(1, 4)]
    res_df = pd.concat(level_res, ignore_index=True).set_index("Level")
    print(res_df)
