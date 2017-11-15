import torch
import torch.nn as nn
import numpy as np
from math import sqrt

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../data"))


def get_bbox_of_anchor(anchor):
    coeffs = anchor[:-1] / 2.0

    x = torch.FloatTensor([1, 0, 0])
    y = torch.FloatTensor([0, 1, 0])
    z = torch.FloatTensor([0, 0, 1])

    t = anchor[3]
    cosval = np.cos(t)
    sinval = np.sin(t)

    R = torch.FloatTensor([[cosval, sinval, 0], [-sinval, cosval, 0],
                           [0, 0, 1]])
    x = x @ R
    y = y @ R

    centroid = x + y + z
    centroid /= np.sqrt(centroid.pow(2).sum())
    centroid *= np.sqrt(coeffs.pow(2).sum())

    return torch.stack([x, y, z, coeffs, centroid]).numpy()


class _BoundingBox():
    def __init__(self, box, z_min, z_max):
        self.box, self.z_min, self.z_max = box, z_min, z_max

    def __str__(self):
        return "{}\n{}\n{}".format(self.box, self.z_min, self.z_max)

    def __format__(self, code):
        return format(str(self), code)


def _convert_corners_to_bb(c):
    box = c[:, :4, :2]
    z_min, _ = c[..., -1].min(dim=-1)
    z_max, _ = c[..., -1].max(dim=-1)

    return _BoundingBox(box, z_min, z_max)


def _get_dss_anchor_set():
    pass


def _get_mini_anchor_set():
    w_scales = [0.25, 0.5, 1.0]
    d_scales = [0.25, 0.5, 1.0]
    h_scales = [0.25, 0.5, 0.75]
    thetas = [0.0]

    anchor_set = torch.FloatTensor(
        len(w_scales) * len(d_scales) * len(h_scales) * len(thetas), 4)

    counter = 0
    for t in thetas:
        for h in h_scales:
            for d in d_scales:
                for w in w_scales:
                    anchor_set[counter] = torch.FloatTensor(
                        [w / 2.0, d, h / 2.0, t])
                    counter += 1

    return anchor_set


def _get_anchor_set():
    w_scales = [0.25, 0.5, 1.0, 1.75]
    h_scales = [0.25, 0.5, 1.0, 1.75]
    z_scales = [0.25, 0.5, 0.75]
    thetas = [np.pi * i / 3.0 for i in range(3)]

    anchor_set = torch.FloatTensor(
        len(w_scales) * len(h_scales) * len(z_scales) * len(thetas), 4)

    counter = 0
    for t in thetas:
        for z in z_scales:
            for h in h_scales:
                for w in w_scales:
                    anchor_set[counter] = torch.FloatTensor(
                        [w / 2.0, h, z / 2.0, t])
                    counter += 1

    return anchor_set


def _get_kmeans_anchors():
    return torch.from_numpy(
        np.load(os.path.join(BASE_DIR, "../data", "bbox_clusters.npy"))).type(
            torch.FloatTensor)


anchor_set = _get_kmeans_anchors()
NUM_ANCHORS = anchor_set.size(0)

if __name__ == "__main__":
    print(anchor_set)
