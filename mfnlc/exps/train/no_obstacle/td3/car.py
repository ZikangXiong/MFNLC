import numpy as np
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from mfnlc.evaluation.simulation import inspect_training_simu
from mfnlc.exps.train.no_obstacle.td3.base import train


def learn():
    action_noise = OrnsteinUhlenbeckActionNoise(np.array([0, 0]), np.array([0.2, 0.2]))
    train(env_name="Car-no-obst",
          total_timesteps=1_000_000,
          action_noise=action_noise,
          policy_kwargs={"net_arch": [128, 128]},
          batch_size=1024,
          gradient_steps=-1,
          train_freq=(4000, "step"),
          log_interval=4,
          n_envs=4)


def evaluate_controller():
    inspect_training_simu(env_name="Car-no-obst",
                          algo="td3",
                          n_rollout=5,
                          render=True)


if __name__ == '__main__':
    for _ in range(5):
        learn()
    evaluate_controller()
