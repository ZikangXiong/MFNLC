import pandas as pd

from mfnlc.config import get_path


def read_log(env_name, algo, planning_algo, level):
    robot_name = env_name.split("-")[0]
    if planning_algo is None:
        log_path = f"{get_path(robot_name, algo, task='evaluation')}/{level}.csv"
    else:
        log_path = f"{get_path(robot_name, algo, task='evaluation')}/{planning_algo}/{level}.csv"
    return pd.read_csv(log_path)


def reach_perc(env_name, algo, planning_algo, level):
    log_df = read_log(env_name, algo, planning_algo, level)
    return log_df["goal_met"].sum() / len(log_df)


def steps_to_reach(env_name, algo, planning_algo, level):
    log_df = read_log(env_name, algo, planning_algo, level)
    reach_step_df = log_df[log_df["goal_met"]]["total_step"]

    if len(reach_step_df) > 0:
        return reach_step_df.mean(), reach_step_df.std()
    else:
        return -1.0, -1.0


def safety_vio(env_name, algo, planning_algo, level):
    log_df = read_log(env_name, algo, planning_algo, level)
    return log_df["collision"].sum() / len(log_df)
