from mfnlc.evaluation.simulation import inspect_training_simu
from mfnlc.exps.train.no_obstacle.ppo.base import train


def learn():
    train(env_name="Nav-no-obst",
          total_timesteps=100_000,
          policy_kwargs={"net_arch": [32, 32]},
          n_steps=1024 * 8)


def evaluate_controller():
    inspect_training_simu(env_name="Nav-no-obst",
                          algo="ppo",
                          n_rollout=5,
                          render=True)


if __name__ == '__main__':
    for _ in range(5):
        learn()
    evaluate_controller()
