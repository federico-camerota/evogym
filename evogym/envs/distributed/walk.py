import numpy as np

from evogym.envs import WalkingFlat
from gym import spaces


class DistributedWalkingFlat(WalkingFlat):

    def __init__(self, body, connections=None):

        # make parent env
        super().__init__(body, connections)

        # override observation space

        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(num_robot_points + num_robot_points,), dtype=np.float)

    def step(self, action):

        # collect parent output
        obs, reward, done, info = super().step(action)

        # refine observation
        obs = np.concatenate((
            self.object_vel_at_time(self.get_time(), "robot").reshape(1, -1).ravel(),
            self.get_relative_pos_obs("robot"),
        ))

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, info

    def reset(self):

        super().reset()
        # refine observation
        obs = np.concatenate((
            self.object_vel_at_time(self.get_time(), "robot").reshape(1, -1).ravel(),
            self.get_relative_pos_obs("robot"),
        ))

        return obs
