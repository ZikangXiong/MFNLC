from abc import abstractmethod
from typing import Dict

import gym
import numpy as np
import safety_gym  # noqa
from gym import Env

from shrl.config import env_config


class EnvBase(Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, no_obstacle: bool,
                 end_on_collision: bool,
                 fixed_init_and_goal: bool):
        self.no_obstacle = no_obstacle
        self.end_on_collision = end_on_collision
        self.fixed_init_and_goal = fixed_init_and_goal
        self.subgoal = None
        self.subgoal_list = []
        self.traj = []
        self.roa_center = None
        self.roa_radius = 0.0

        self.obstacle_radius = 0.15
        self.robot_radius = 0.3

        self.render_config = {
            "traj_sample_freq": -1,
            "follow": False,
            "vertical": False,
            "scale": 4.0
        }

    def reset(self):
        self.subgoal = None
        self.subgoal_list = []
        self.traj = []
        self.roa_center = None
        self.roa_radius = 0.0

    @abstractmethod
    def goal_obs(self) -> np.ndarray:
        pass

    @abstractmethod
    def robot_obs(self) -> np.ndarray:
        pass

    @abstractmethod
    def obstacle_obs(self) -> np.ndarray:
        pass

    @abstractmethod
    def update_env_config(self, config: Dict):
        pass

    def set_subgoal(self, subgoal: np.ndarray, store=False):
        self.subgoal = subgoal
        if store:
            self.subgoal_list.append(subgoal)

    def get_obs(self):
        return np.concatenate([self.goal_obs(),
                               self.robot_obs(),
                               self.obstacle_obs()])

    def set_roa(self, roa_center: np.ndarray, roa_radius: float):
        self.roa_center = roa_center
        self.roa_radius = roa_radius

    def set_render_config(self, config: Dict):
        self.render_config.update(config)


class SafetyGymBase(EnvBase):

    def __init__(self,
                 env_name,
                 no_obstacle=False,
                 end_on_collision=False,
                 fixed_init_and_goal=False) -> None:
        super().__init__(no_obstacle=no_obstacle,
                         end_on_collision=end_on_collision,
                         fixed_init_and_goal=fixed_init_and_goal)

        if self.no_obstacle:
            env_name = env_name[:-4] + "0-v0"
        self.env = gym.make(env_name)
        self.env_name = env_name

        # Count robot relevant observations (ignore lidar)
        self.robot_obs_size = sum([np.prod(self.env.obs_space_dict[obs_name].shape)
                                   for obs_name in self.env.obs_space_dict
                                   if 'lidar' not in obs_name])

        self.obstacle_in_obs = 2
        self.num_relevant_dim = 2  # For x-y relevant observations ignoring z-axis

        # Reward config
        self.collision_penalty = -0.01
        self.arrive_reward = 20

        customized_config = env_config[self.robot_name].get("env_prop", None)
        if customized_config is not None:
            self.update_env_config(customized_config)

        self._build_space()
        self.env.toggle_observation_space()
        self.previous_goal_dist = None

    def _build_space(self):
        self.action_space = self.env.action_space

        max_observation = 10
        if self.no_obstacle:
            observation_high = max_observation * np.ones(
                self.num_relevant_dim + self.robot_obs_size,
                dtype=np.float32)
        else:
            observation_high = max_observation * np.ones(
                self.num_relevant_dim + self.robot_obs_size + self.obstacle_in_obs * self.num_relevant_dim,
                dtype=np.float32)
        observation_low = -observation_high
        self.observation_space = gym.spaces.Box(observation_low, observation_high, dtype=np.float32)

    def seed(self, seed=None):
        self.env.seed(seed)

    def update_env_config(self, config: Dict):
        self.__dict__.update(config)  # Lazy implementation: can introduce unnecessary binding
        self.env.__dict__.update(config)

        assert "robot_base" not in config.keys(), \
            "Do not change robot, this requires to rebuild observation and action space"
        self.env.build_placements_dict()

        self.env.viewer = None
        self.env.world = None
        self.env.clear()
        self.env.done = True

        self.env.clear()
        self._build_space()

    @property
    def robot_name(self):
        all_robot_names = ["Point", "Car", "Doggo"]
        for name in all_robot_names:
            if name in self.env_name:
                return name

    @property
    def robot_pos(self):
        return self.env.robot_pos[:2]

    @property
    def hazards_pos(self):
        return self.env.hazards_pos

    def reset(self):
        super(SafetyGymBase, self).reset()
        self.env.reset()
        self.env.num_steps = 10000

        if self.fixed_init_and_goal and (
                len(self.env.goal_locations) == 0
                or
                len(self.env.robot_locations) == 0
        ):
            self.env.goal_locations = self.goal_obs()[:2].tolist()
            self.env.robot_locations = self.robot_pos[:2].tolist()

        self.previous_goal_dist = None

        return self.get_obs()

    def goal_obs(self) -> np.ndarray:
        if self.subgoal is not None:
            goal_obs = self.subgoal[:self.num_relevant_dim] - self.env.robot_pos[:self.num_relevant_dim]
        else:
            goal_obs = (self.env.goal_pos - self.env.robot_pos)[:self.num_relevant_dim]
        return goal_obs

    def robot_obs(self) -> np.ndarray:
        # only gets observation dimensions relevant to robot from safety-gym
        obs = self.env.obs()
        flat_obs = np.zeros(self.robot_obs_size)
        offset = 0

        for k in sorted(self.env.obs_space_dict.keys()):
            if "lidar" in k:
                continue
            k_size = np.prod(obs[k].shape)
            flat_obs[offset:offset + k_size] = obs[k].flat
            offset += k_size
        return flat_obs

    def obstacle_obs(self) -> np.ndarray:
        if self.no_obstacle:
            return np.array([])

        # get distance to each obstacle upto self.obstacle_in_obs nearest obstacles
        vec_to_obs = (self.env.hazards_pos - self.env.robot_pos)[:, :self.num_relevant_dim]
        dist_to_obs = np.linalg.norm(vec_to_obs, ord=2, axis=-1)
        order = dist_to_obs.argsort()[:self.obstacle_in_obs]
        flattened_vec = vec_to_obs[order].flatten()
        # in case of that the obstacle number in environment is smaller than self.obstacle_in_obs
        output = np.zeros(self.obstacle_in_obs * self.num_relevant_dim)
        output[:flattened_vec.shape[0]] = flattened_vec
        return output

    def get_goal_reward(self):
        goal_dist = np.linalg.norm(self.goal_obs(), ord=2)
        if self.previous_goal_dist is None:
            goal_reward = 0.0
        else:
            goal_reward = (self.previous_goal_dist - goal_dist) * 10
        self.previous_goal_dist = goal_dist

        return goal_reward

    def step(self, action: np.ndarray):
        s, r, done, info = self.env.step(action)

        # As of now use safety gym info['cost'] to detect collisions
        collision = info.get('cost', 1.0) > 0
        info["collision"] = collision

        arrive = info.get("goal_met", False)

        reward = self.get_goal_reward() + collision * self.collision_penalty + arrive * self.arrive_reward

        if self.end_on_collision and collision:
            done = True
        else:
            done = arrive or done

        obs = self.get_obs()
        if arrive:
            # if the robot meets goal, the goal will be reset immediately
            # this can cause the goal observation has large jumps and affect Lyapunov function
            obs[:self.num_relevant_dim] = np.zeros(self.num_relevant_dim)

        self.traj.append(self.robot_pos)

        return obs, reward, done, info

    def render(self,
               mode="human",
               camera_id=1,
               width=2048,
               height=2048):
        # plot subgoal
        if self.env.viewer is not None:
            for subgoal in self.subgoal_list:
                self.env.render_area(subgoal, 0.1,
                                     np.array([0, 1, 0.0, 0.5]), 'subgoal', alpha=0.5)

            if self.render_config["traj_sample_freq"] > 0:
                for pos in self.traj[::self.render_config["traj_sample_freq"]]:
                    self.env.render_area(pos, 0.05,
                                         np.array([1, 0.5, 0, 0.5]), 'position', alpha=0.5)

            if self.roa_center is not None:
                self.env.render_area(self.roa_center, self.roa_radius,
                                     np.array([0.2, 1.0, 1.0, 0.5]), 'RoA approx', alpha=0.5)

        return self.env.render(mode, camera_id,
                               width=width, height=height,
                               follow=self.render_config["follow"],
                               vertical=self.render_config["vertical"],
                               scale=self.render_config["scale"])


class ObstacleMaskWrapper(gym.Wrapper):
    """
    ! This wrapper will not change the original observation space
    """

    def get_obs(self):
        return np.concatenate([self.goal_obs(),
                               self.robot_obs()])

    def reset(self, **kwargs):
        super(ObstacleMaskWrapper, self).reset(**kwargs)
        return self.get_obs()

    def step(self, action):
        _, r, d, info = super(ObstacleMaskWrapper, self).step(action)
        return self.get_obs(), r, d, info
