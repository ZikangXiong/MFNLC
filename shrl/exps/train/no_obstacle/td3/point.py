import numpy as np
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from shrl.evaluation.simulation import inspect_training_simu
from shrl.exps.train.no_obstacle.td3.base import train


def learn():
    action_noise = OrnsteinUhlenbeckActionNoise(np.array([0, 0]), np.array([0.1, 0.1]))
    train(env_name="Point-no-obst",
          total_timesteps=200_000,
          action_noise=action_noise,
          policy_kwargs={"net_arch": [64, 64]},
          train_freq=(4, "episode"),
          gradient_steps=-1,
          batch_size=1024,
          log_interval=4)


def evaluate_controller():
    inspect_training_simu(env_name="Point-no-obst",
                          algo="td3",
                          n_rollout=20,
                          render=True)


if __name__ == '__main__':
    for _ in range(5):
        learn()
    evaluate_controller()
