import numpy as np

from evogym.envs import WalkingFlat
from gym import spaces

import abc_sr.evogym_utils as evoutils


class DistributedWalkingFlat(WalkingFlat):

    def __init__(self, body, connections=None):

        # make parent env
        super().__init__(body, connections)

        # override observation space

        self.robot_shape = body.shape

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(1,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(5*(4 + 1),), dtype=np.float)

    def step(self, action):

        # collect parent output
        obs, reward, done, info = super().step(action)

        # refine observation
        vel = self.object_vel_at_time(self.get_time(), "robot")
        pos = self.get_relative_pos_obs("robot").reshape((2, -1))
        obs = np.hstack([pos, vel])

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, info

    def reset(self):

        super().reset()
        # refine observation
        vel = self.object_vel_at_time(self.get_time(), "robot")
        pos = self.get_relative_pos_obs("robot").reshape((2, -1))
        obs = np.hstack([pos, vel])

        return obs
