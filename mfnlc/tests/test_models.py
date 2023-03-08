from mfnlc.envs import get_env
from mfnlc.evaluation.model import load_model


def test_cpo_models():
    robots = [
        # "Nav",
        "Point",
        "Car",
        # "Doggo"
    ]

    for robot in robots:
        model = load_model(robot, "cpo")
        env = get_env(env_name=f"{robot}-eval")
        env.update_env_config({
            "obstacle_in_obs": 8
        })

        print(f"=== {robot} CPO test ===")
        obs = env.reset()
        print("observation shape: ", obs.shape)
        action = model.predict(obs)[0]
        print("expected action shape: ", env.action_space.low.shape)
        print("actual action shape: ", action.shape)

        for i in range(1000):
            action = model.predict(obs)[0]
            obs, _, done, _ = env.step(action)
            env.render()
            if done:
                break


if __name__ == '__main__':
    test_cpo_models()
