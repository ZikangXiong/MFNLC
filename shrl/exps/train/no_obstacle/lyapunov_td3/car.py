import numpy as np
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from shrl.evaluation.simulation import inspect_training_simu
from shrl.exps.train.no_obstacle.lyapunov_td3.base import train, evaluate_lyapunov_of


def colearn():
    action_noise = OrnsteinUhlenbeckActionNoise(np.array([0] * 2), np.array([0.2] * 2))
    train(env_name="Car-no-obst",
          lf_structure=[26, 128, 128, 1],
          lqf_structure=[26 + 2, 128, 128, 1],
          tclf_ub=15,
          tclf_q_sigma=0,
          total_timesteps=2_000_000,
          action_noise=action_noise,
          policy_kwargs={"net_arch": [128, 128]},
          batch_size=2048,
          gradient_steps=-1,
          train_freq=(4000, "step"),
          log_interval=4,
          n_envs=4)


def evaluate_controller():
    inspect_training_simu(env_name="Car-no-obst",
                          algo="lyapunov_td3",
                          n_rollout=5,
                          render=True)


def evaluate_lyapunov():
    evaluate_lyapunov_of("Car")


if __name__ == "__main__":
    # for i in range(5):
    #     colearn()
    # evaluate_controller()
    evaluate_lyapunov()
