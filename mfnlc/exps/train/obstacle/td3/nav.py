import numpy as np
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from mfnlc.evaluation.simulation import inspect_training_simu
from mfnlc.exps.train.obstacle.td3.base import train


def learn():
    action_noise = OrnsteinUhlenbeckActionNoise(np.array([0, 0]), np.array([0.01, 0.01]))
    train(env_name="Nav",
          total_timesteps=800_000,
          action_noise=action_noise,
          gradient_steps=200,
          n_envs=4,
          train_freq=(200, "step"),
          batch_size=400,
          policy_kwargs={"net_arch": [64, 64]})


def evaluate_controller():
    inspect_training_simu(env_name="Nav",
                          algo="e2e",
                          n_rollout=20,
                          render=True)


if __name__ == '__main__':
    # learn()
    evaluate_controller()
