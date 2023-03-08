import os
import shutil

from mfnlc.config import get_path


def copy_current_model_to_log_dir(robot_name, algo):
    log_path = get_path(robot_name, algo, "log")
    log_names = os.listdir(log_path)
    log_names.sort(key=lambda x: int(x.split("_")[-1]))

    dest_folder = f"{log_path}/{log_names[-1]}"
    shutil.copy2(get_path(robot_name, algo, "model"), dest_folder)
    if algo == "lyapunov_td3":
        shutil.copy2(get_path(robot_name, algo, "tclf"), dest_folder)
    print(f"The log and model are stored in {dest_folder}")
