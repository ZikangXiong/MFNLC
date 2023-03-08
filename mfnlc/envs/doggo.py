from mfnlc.envs.base import SafetyGymBase


class DoggoNav(SafetyGymBase):
    def __init__(self,
                 no_obstacle=False,
                 end_on_collision=False,
                 fixed_init_and_goal=False) -> None:
        super().__init__('Safexp-DoggoGoal1-v0', no_obstacle,
                         end_on_collision, fixed_init_and_goal)
        self.arrive_reward = 100
