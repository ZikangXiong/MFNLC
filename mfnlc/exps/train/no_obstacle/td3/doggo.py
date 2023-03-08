import numpy as np
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from mfnlc.evaluation.simulation import inspect_training_simu
from mfnlc.exps.train.no_obstacle.td3.base import train


def learn():
    action_noise = OrnsteinUhlenbeckActionNoise(np.array([0] * 12), np.array([0.3] * 12))
    train(env_name="Doggo-no-obst",
          total_timesteps=40_000_000,
          action_noise=action_noise,
          policy_kwargs={"net_arch": [512, 256],
                         "optimizer_kwargs": {"weight_decay": 1e-5}},
          batch_size=6000,
          n_envs=16,
          train_freq=(4000, "step"),
          log_interval=16,
          gradient_steps=100)


def evaluate_controller():
    inspect_training_simu(env_name="Doggo-no-obst",
                          algo="td3",
                          n_rollout=5,
                          render=True)


if __name__ == '__main__':
    for _ in range(5):
        learn()
    evaluate_controller()
