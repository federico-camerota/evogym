from distributed.walk import DistributedWalkingFlat
import numpy as np


def make_distributed_env(base_env, body):
    if base_env == "Walker-v0":
        return DistributedWalkingFlat(body)
    raise ValueError("Unsupported base env for distributed: {}".format(str(base_env)))


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
