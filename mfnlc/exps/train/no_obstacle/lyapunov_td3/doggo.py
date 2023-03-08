import numpy as np
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from mfnlc.evaluation.simulation import inspect_training_simu
from mfnlc.exps.train.no_obstacle.lyapunov_td3.base import train, evaluate_lyapunov_of


def colearn():
    action_noise = OrnsteinUhlenbeckActionNoise(np.array([0] * 12), np.array([0.3] * 12))
    train(env_name="Doggo-no-obst",
          lf_structure=[58, 256, 256, 1],
          lqf_structure=[58 + 12, 256, 256, 1],
          tclf_ub=15,
          tclf_q_sigma=-4,
          total_timesteps=40_000_000,
          action_noise=action_noise,
          policy_kwargs={"net_arch": [512, 256],
                         "optimizer_kwargs": {"weight_decay": 1e-5}},
          batch_size=6000,
          n_envs=16,
          train_freq=(1000, "step"),
          log_interval=16,
          gradient_steps=200)


def evaluate_controller():
    inspect_training_simu(env_name="Doggo-no-obst",
                          algo="lyapunov_td3",
                          n_rollout=5,
                          render=True)


def evaluate_lyapunov():
    evaluate_lyapunov_of("Doggo")


if __name__ == "__main__":
    # for i in range(5):
    #     colearn()
    # evaluate_controller()
    evaluate_lyapunov()
