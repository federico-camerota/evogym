import gym
import evogym.envs
from evogym import sample_robot
import numpy as np


def compute_voxel_obs(body):
    vertices = np.zeros((body.shape[0] + 1, body.shape[1] + 1))
    for row in range(body.shape[0]):
        for col in range(body.shape[1]):
            if body[row, col]:
                vertices[row, col] = 1
                vertices[row, col + 1] = 1
                vertices[row + 1, col] = 1
                vertices[row + 1, col + 1] = 1
    return vertices


def get_single_voxel_obs(body, obs, i, j):
    vertices = compute_voxel_obs(body)
    num = 0
    voxel_obs = []
    for row in range(vertices.shape[0]):
        for col in range(vertices.shape[1]):
            if (row, col) == (i, j) or (row, col) == (i + 1, j) or (row, col) == (i, j + 1) or (row, col) == (i + 1, j + 1):
                voxel_obs.append(obs[num])
            if vertices[row, col]:
                num += 1
    return np.array(voxel_obs)


if __name__ == '__main__':

    body, connections = sample_robot((5,5))
    env = gym.make('Walker-v0', body=body)
    env.reset()

    while True:
        action = env.action_space.sample()-1
        ob, reward, done, info = env.step(action)
        env.render()

        if done:
            env.reset()

    env.close()
