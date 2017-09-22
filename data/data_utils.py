import torch
import numpy as np


class PointcloudScale(object):
    def __init__(self, mean=2.0, std=1.0, clip=1.8):
        self.mean, self.std, self.clip = mean, std, clip

    def __call__(self, points):
        scaler = torch.FloatTensor(1).normal_(
            mean=self.mean, std=self.std).clamp_(
                max(self.mean - self.clip, 0.01), self.mean + self.clip)
        return scaler * points


class PointcloudRotate(object):
    def __init__(self, x_axis=True, z_axis=False):
        assert x_axis or z_axis
        self.x, self.y = x_axis, z_axis

    def _get_angles(self):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)

        return cosval, sinval

    def __call__(self, points):
        if self.x:
            sinval, cosval = self._get_angles()
            x_mat = torch.FloatTensor([[cosval, 0, sinval], [0, 1, 0],
                                       [-sinval, 0, cosval]])
        else:
            x_mat = torch.eye(3)

        if self.x:
            sinval, cosval = self._get_angles()
            z_mat = torch.FloatTensor([[cosval, -sinval, 0], [sinval, 1, cosval],
                                       [0, 0, 1]])
        else:
            z_mat = torch.eye(3)

        rot_mat = z_mat @ x_mat

        return points @ rot_mat


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = torch.FloatTensor(*points.size()).normal_(
            mean=0.0, std=self.std).clamp_(-self.clip, self.clip)
        return points + jittered_data


class PointcloudTranslate(object):
    def __init__(self, std=1.0, clip=3.0):
        self.std, self.clip = std, clip

    def __call__(self, points):
        translation = torch.FloatTensor(3).normal_(
            mean=0.0, std=self.std).clamp_(-self.clip, self.clip)
        return points + translation


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).type(torch.FloatTensor)
