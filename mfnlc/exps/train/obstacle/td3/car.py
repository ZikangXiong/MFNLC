import numpy as np
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from mfnlc.evaluation.simulation import inspect_training_simu
from mfnlc.exps.train.obstacle.td3.base import train


def learn():
    action_noise = OrnsteinUhlenbeckActionNoise(np.array([0, 0]), np.array([0.2, 0.2]))
    train(env_name="Car",
          total_timesteps=2_000_000,
          action_noise=action_noise,
          policy_kwargs={"net_arch": [256, 256]},
          batch_size=4096,
          gradient_steps=100,
          train_freq=(1000, "step"),
          log_interval=4,
          n_envs=4)


def evaluate_controller():
    inspect_training_simu(env_name="Car",
                          algo="e2e",
                          n_rollout=20,
                          render=True)


if __name__ == '__main__':
    # learn()
    evaluate_controller()
