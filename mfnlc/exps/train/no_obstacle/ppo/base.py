import os
from typing import Any, Dict, Optional, Type, Union

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback

from mfnlc.config import get_path, default_device
from mfnlc.envs import get_env, get_sub_proc_env
from mfnlc.exps.train.utils import copy_current_model_to_log_dir


def train(env_name,
          total_timesteps: int,
          policy: Union[str, Type[ActorCriticPolicy]] = "MlpPolicy",
          callback: MaybeCallback = None,
          log_interval: int = 1,
          eval_env: Optional[GymEnv] = None,
          eval_freq: int = -1,
          n_eval_episodes: int = 5,
          tb_log_name: str = "PPO",
          eval_log_path: Optional[str] = None,
          reset_num_timesteps: bool = True,
          learning_rate: Union[float, Schedule] = 3e-4,
          n_steps: int = 2048,
          batch_size: int = 64,
          n_epochs: int = 10,
          gamma: float = 0.99,
          gae_lambda: float = 0.95,
          clip_range: Union[float, Schedule] = 0.2,
          clip_range_vf: Union[None, float, Schedule] = None,
          ent_coef: float = 0.0,
          vf_coef: float = 0.5,
          max_grad_norm: float = 0.5,
          use_sde: bool = False,
          sde_sample_freq: int = -1,
          target_kl: Optional[float] = None,
          create_eval_env: bool = False,
          policy_kwargs: Optional[Dict[str, Any]] = None,
          verbose: int = 1,
          seed: Optional[int] = None,
          device: Union[th.device, str] = default_device,
          _init_setup_model: bool = True,
          n_envs: int = 1):
    algo = "ppo"

    if n_envs == 1:
        env = get_env(env_name)
    elif n_envs > 1:
        env = get_sub_proc_env(env_name, n_envs)
    else:
        raise ValueError(f"n_envs should be greater than 0, but it is {n_envs}")

    robot_name = env_name.split("-")[0]

    tensorboard_log = get_path(robot_name, algo, "log")

    model = PPO(policy, env, learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, clip_range_vf,
                ent_coef, vf_coef, max_grad_norm, use_sde, sde_sample_freq, target_kl, tensorboard_log, create_eval_env,
                policy_kwargs, verbose, seed, device, _init_setup_model)

    model.learn(total_timesteps, callback, log_interval, eval_env, eval_freq,
                n_eval_episodes, tb_log_name, eval_log_path, reset_num_timesteps)

    model_path = get_path(robot_name, algo, "model")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    copy_current_model_to_log_dir(robot_name, algo)
