import os
from typing import Any, Dict, Optional, Tuple, Type, Union

from stable_baselines3 import TD3
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   Schedule)
from stable_baselines3.td3.policies import TD3Policy

from shrl.config import get_path, default_device
from shrl.envs import get_env, get_sub_proc_env
from shrl.exps.train.utils import copy_current_model_to_log_dir


def train(env_name,
          total_timesteps: int,
          policy: Union[str, Type[TD3Policy]] = "MlpPolicy",
          learning_rate: Union[float, Schedule] = 1e-3,
          buffer_size: int = 1_000_000,  # 1e6
          learning_starts: int = 100,
          batch_size: int = 100,
          tau: float = 0.005,
          gamma: float = 0.99,
          train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
          gradient_steps: int = 40,
          action_noise: Optional[ActionNoise] = None,
          replay_buffer_class: Optional[ReplayBuffer] = None,
          replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
          optimize_memory_usage: bool = False,
          policy_delay: int = 2,
          target_policy_noise: float = 0.2,
          target_noise_clip: float = 0.5,
          create_eval_env: bool = False,
          policy_kwargs: Optional[Dict[str, Any]] = None,
          verbose: int = 1,
          seed: Optional[int] = None,
          callback: MaybeCallback = None,
          log_interval: int = 4,
          eval_env: Optional[GymEnv] = None,
          eval_freq: int = -1,
          n_eval_episodes: int = 5,
          tb_log_name: str = "TD3",
          eval_log_path: Optional[str] = None,
          reset_num_timesteps: bool = True,
          n_envs: int = 1,
          ):
    algo = "td3"

    if n_envs == 1:
        env = get_env(env_name)
    elif n_envs > 1:
        env = get_sub_proc_env(env_name, n_envs)
    else:
        raise ValueError(f"n_envs should be greater than 0, but it is {n_envs}")

    robot_name = env_name.split("-")[0]

    tensorboard_log = get_path(robot_name, algo, "log")

    model = TD3(
        policy, env, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma,
        train_freq, gradient_steps, action_noise, replay_buffer_class, replay_buffer_kwargs,
        optimize_memory_usage, policy_delay, target_policy_noise, target_noise_clip,
        tensorboard_log, create_eval_env, policy_kwargs, verbose, seed, default_device)

    model.learn(total_timesteps, callback, log_interval, eval_env, eval_freq,
                n_eval_episodes, tb_log_name, eval_log_path, reset_num_timesteps)

    model_path = get_path(robot_name, "td3", "model")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    copy_current_model_to_log_dir(robot_name, algo)
