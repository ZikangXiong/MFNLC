from mfnlc.evaluation.simulation import inspect_training_simu
from mfnlc.exps.train.no_obstacle.ppo.base import train


def learn():
    train(env_name="Doggo-no-obst",
          total_timesteps=30_000_000,
          policy_kwargs={"net_arch": [512, 256],
                         "optimizer_kwargs": {"weight_decay": 1e-5}},
          batch_size=32768,
          n_envs=16,
          log_interval=16)


def evaluate_controller():
    inspect_training_simu(env_name="Doggo-no-obst",
                          algo="ppo",
                          n_rollout=5,
                          render=True)


if __name__ == '__main__':
    for _ in range(5):
        learn()
    evaluate_controller()
