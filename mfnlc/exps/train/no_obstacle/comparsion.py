import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from mfnlc.config import get_path


def log_to_df(log_dir, max_step, scalar_name, x_name, y_name, step_size):
    event_accumulator = EventAccumulator(log_dir)
    event_accumulator.Reload()

    events = event_accumulator.Scalars(scalar_name)
    xp = [x.step for x in events]
    yp = [x.value for x in events]

    x = np.arange(0, max_step, step_size)
    y = np.interp(x, xp, yp)

    return pd.DataFrame({x_name: x,
                         y_name: y})


def list_all_log_dir(log_root_dir):
    return [f"{log_root_dir}/{dir_name}" for dir_name in os.listdir(log_root_dir) if
            os.path.isdir(f"{log_root_dir}/{dir_name}")]


def generate_comparison_dfs(log_dir_list, max_step, scalar_name="rollout/ep_rew_mean",
                            x_name="step", y_name="reward", step_size=1000):
    comparing_df_list = []
    for log_root_dir in log_dir_list:
        exp_df_list = []
        for log_dir in list_all_log_dir(log_root_dir):
            exp_df_list.append(log_to_df(log_dir, max_step, scalar_name, x_name, y_name, step_size))
        comparing_df_list.append(pd.concat(exp_df_list))

    return comparing_df_list


def plot(comparison_dfs, algo_names, x_name, y_name, save_path, legend=True, ax=None):
    for df, algo_name in zip(comparison_dfs, algo_names):
        df["algorithm"] = algo_name
    df = pd.concat(comparison_dfs).reset_index()

    ax.ticklabel_format(axis="x", style="sci", scilimits=(5, 5))
    plt.rcParams.update({'font.size': 10})
    sns.lineplot(data=df, x=x_name, y=y_name, hue="algorithm", ax=ax)

    if ax is None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.clf()

        if not legend:
            plt.legend([], [], frameon=False)
        else:
            plt.legend()


def step_info(robot_name):
    if robot_name == "Nav":
        max_step = 100_000
        step_size = 1_000
    elif robot_name == "Point":
        max_step = 200_000
        step_size = 1_000
    elif robot_name == "Car":
        max_step = 500_000
        step_size = 1_000
    elif robot_name == "Doggo":
        max_step = 30_000_000
        step_size = 300_000
    else:
        raise NotImplementedError(f"Unsupported robot {robot_name}")

    return max_step, step_size


def compare_training_reward(robot_name):
    scalar_name = "rollout/ep_rew_mean"
    x_name = "step"
    y_name = "reward"
    algo_names = ["TD3", "PPO", "Ours"]
    max_step, step_size = step_info(robot_name)

    log_dir_list = [get_path(robot_name, "td3", "log"),
                    get_path(robot_name, "ppo", "log"),
                    get_path(robot_name, "lyapunov_td3", "log")]

    comparison_dfs = generate_comparison_dfs(log_dir_list, max_step,
                                             scalar_name, x_name, y_name,
                                             step_size)
    plot(comparison_dfs, algo_names, x_name, y_name,
         get_path(robot_name, None, "comparison") + "/training_res.pdf", legend=True)
    print(f"result is saved to {get_path(robot_name, None, 'comparison')}")


def compare_all_training_results():
    scalar_name = "rollout/ep_rew_mean"
    x_name = "step"
    y_name = "reward"
    algo_names = ["TD3", "PPO", "Ours"]
    robot_names = ["Nav", "Point", "Car", "Doggo"]
    save_path = get_path("", None, "comparison") + "/train_res.pdf"

    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    for i, robot_name in enumerate(robot_names):
        max_step, step_size = step_info(robot_name)
        log_dir_list = [get_path(robot_name, "td3", "log"),
                        get_path(robot_name, "ppo", "log"),
                        get_path(robot_name, "lyapunov_td3", "log")]

        comparison_dfs = generate_comparison_dfs(log_dir_list, max_step,
                                                 scalar_name, x_name, y_name,
                                                 step_size)
        ax = axes[i // 2][i % 2]
        plot(comparison_dfs, algo_names, x_name, y_name, save_path=None, ax=ax)

        if robot_name == "Doggo":
            ax.set_title("Quadruped")
        elif robot_name == "Nav":
            ax.set_title("Sweeping")
        else:
            ax.set_title(robot_name)

        # ax.set_xlabel("")
        # ax.set_ylabel("")
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(algo_names))
        _lg = ax.get_legend()
        _lg.remove()

        if i == 2:
            ax.set_ylim([-10, 21])
        ax.yaxis.set_label_coords(-.08, .5)

    fig.tight_layout(pad=0.5)
    plt.savefig(save_path, bbox_inches="tight")


if __name__ == '__main__':
    # compare_training_reward("Nav")
    # compare_training_reward("Point")
    # compare_training_reward("Car")
    # compare_training_reward("Doggo")
    compare_all_training_results()
