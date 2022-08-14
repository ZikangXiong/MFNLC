from typing import Tuple

import numpy as np
from stable_baselines3 import TD3, PPO
import onnxruntime as ort

from shrl.config import get_path
from shrl.envs import get_env
from shrl.learn.lyapunov_td3 import LyapunovTD3


class CPOWrapper:
    def __init__(self, robot_name):
        self.ort_sess = ort.InferenceSession(get_path(robot_name, "cpo", "model"))
        self.op_vals = ['pi']

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, None]:
        x = x.astype("float32").reshape([-1, x.shape[-1]])
        res = self.ort_sess.run(self.op_vals, {'x': x})[0]
        return res.flatten(), None


def load_model(env_name, algo):
    robot_name = env_name.split("-")[0]

    if "lyapunov" in algo:
        env = get_env(f"{robot_name}-no-obst")
    else:
        env = get_env(env_name)

    if algo == "lyapunov_td3" or algo == "rrt_lyapunov":
        model = LyapunovTD3.load(get_path(robot_name, "lyapunov_td3", "model"), env=env)
    elif algo == "td3":
        model = TD3.load(get_path(robot_name, "td3", "model"), env=env)
    elif algo == "ppo":
        model = PPO.load(get_path(robot_name, "ppo", "model"), env=env)
    elif algo == "e2e" or algo == "rrt_e2e":
        model = TD3.load(get_path(robot_name, "e2e", "model"), env=env)
    elif algo == "cpo" or algo == "rrt_cpo":
        return CPOWrapper(robot_name)
    else:
        raise NotImplementedError(f"Does not support {env_name} - {algo}")

    return model
