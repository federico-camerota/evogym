import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from skimage.measure import regionprops_table
from skimage.measure import label as label_skimage


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
    body2 = np.array([[1 if body[x, y] != 0 else 0 for y in range(body.shape[1])] for x in range(body.shape[0])])
    return regionprops_table(label_skimage(body2), properties=("centroid", "orientation", "eccentricity"))["eccentricity"][0]


def get_eccentricity(body, eps=1e-15):
    coords = []
    for x in range(body.shape[0]):
        for y in range(body.shape[1]):
            if body[x, y] != 0:
                coords.append((x, y))
    points = np.array([[x, y] for x, y in coords])
    try:
        hull = ConvexHull(points)
    except:
        return 1.0
    polygon = Polygon(points[hull.vertices])
    hull_mask = np.zeros(body.shape)
    for x in range(body.shape[0]):
        for y in range(body.shape[1]):
            if Point(x, y).distance(polygon) < eps:
                hull_mask[x, y] = 1
    return np.count_nonzero(body) / np.sum(hull_mask)
