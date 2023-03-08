from mfnlc.evaluation.simulation import inspect_training_simu
from mfnlc.exps.train.no_obstacle.ppo.base import train


def learn():
    train(env_name="Point-no-obst",
          total_timesteps=200_000,
          policy_kwargs={"net_arch": [64, 64]},
          batch_size=1024,
          log_interval=1,
          n_steps=1024,
          n_envs=4)


def evaluate_controller():
    inspect_training_simu(env_name="Point-no-obst",
                          algo="ppo",
                          n_rollout=20,
                          render=True)


if __name__ == '__main__':
    for _ in range(5):
        learn()
    evaluate_controller()
