import math

import numpy as np


def get_size(body):
    return np.count_nonzero(body)


def get_num_voxels_type(body, voxel_type):
    return np.count_nonzero(body[body == voxel_type])


def get_width(body):
    start = None
    for i in range(body.shape[0]):
        if np.any(body[i, :]) and start is None:
            start = i
        elif not np.any(body[i, :]) and start is not None:
            return i - start
    return body.shape[0] - start


def get_height(body):
    start = None
    for i in range(body.shape[1]):
        if np.any(body[:, i]) and start is None:
            start = i
        elif not np.any(body[:, i]) and start is not None:
            return i - start
    return body.shape[1] - start


def get_elongation(body):
    coordinates = list(filter(lambda x: body[x[0], x[1]] != 0, [(x, y) for x in range(body.shape[0]) for y in range(body.shape[1])]))
    diameters = []
    for i in range(8):
        theta = (2 * i * math.pi) / 8
        rotated = [(x * math.cos(theta) - y * math.sin(theta), x * math.sin(theta) + y * math.cos(theta))
                   for x, y in coordinates]
        min_x, max_x, min_y, max_y = min(rotated, key=lambda x: x[0])[0], max(rotated, key=lambda x: x[0])[0], \
                                     min(rotated, key=lambda x: x[1])[1], max(rotated, key=lambda x: x[1])[1]
        side_x = max_x - min_x + 1
        side_y = max_y - min_y + 1
        diameters.append(min(side_y, side_x) / max(side_y, side_x))
    print(diameters)
    return 1.0 - min(diameters)


def get_eccentricity(body):
    body2 = np.array([[1 if body[x, y] != 0 else 0 for y in range(body.shape[1])] for x in range(body.shape[0])])
    stop = False
    while not stop:
        stop = True
        for x in range(body2.shape[0]):
            for y in range(body2.shape[1]):
                if body2[x, y] == 0 and sum([c == 1 for c in get_neighborhood(body2, x, y)]) >= 3:
                    body2[x, y] = 1
                    stop = False
    return np.count_nonzero(np.array([[1 if body[x, y] != 0 else 0 for y in range(body.shape[1])]
                                      for x in range(body.shape[0])])) / np.count_nonzero(body2)


def get_neighborhood(body, x, y):
    out = []
    if x < body.shape[0] - 1:
        out.append(body[x + 1, y])
    if x > 0:
        out.append(body[x - 1, y])
    if y > 0:
        out.append(body[x, y - 1])
    if y < body.shape[1] - 1:
        out.append(body[x, y + 1])
    return out
