from shrl.envs.base import SafetyGymBase


class PointNav(SafetyGymBase):
    def __init__(self,
                 no_obstacle=False,
                 end_on_collision=False,
                 fixed_init_and_goal=False) -> None:
        super().__init__('Safexp-PointGoal1-v0', no_obstacle,
                         end_on_collision, fixed_init_and_goal)
