import numpy as np
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from shrl.evaluation.simulation import inspect_training_simu
from shrl.exps.train.obstacle.td3.base import train


def learn():
    action_noise = OrnsteinUhlenbeckActionNoise(np.array([0] * 12), np.array([0.3] * 12))
    train(env_name="Doggo",
          total_timesteps=40_000_000,
          action_noise=action_noise,
          policy_kwargs={"net_arch": [512, 256],
                         "optimizer_kwargs": {"weight_decay": 1e-5}},  # add L2 reg to increase numeric stability
          batch_size=6000,
          n_envs=16,
          train_freq=(1000, "step"),
          log_interval=16,
          gradient_steps=200)


def evaluate_controller():
    inspect_training_simu(env_name="Doggo",
                          algo="e2e",
                          n_rollout=5,
                          render=True)


if __name__ == '__main__':
    learn()
    # evaluate_controller()
