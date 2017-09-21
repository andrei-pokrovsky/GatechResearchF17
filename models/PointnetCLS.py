import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import pytorch_utils as pt_utils
from TransformNets import TransformNet, TranslationNet


class Pointnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.translation_net = TranslationNet()
        self.t_net = TransformNet(1, 3, 3, scale=False)
        self.f_net = TransformNet(64, 1, 64, scale=False)

        self.input_mlp = nn.Sequential(
            pt_utils.Conv2d(1, 64, [1, 3], bn=True),
            pt_utils.Conv2d(64, 64, bn=True))

        self.second_mlp = pt_utils.SharedMLP([64, 64, 128, 1024], bn=True)

        self.final_mlp = nn.Sequential(
            pt_utils.FC(1024, 512, bn=True),
            pt_utils.FC(512, 256, bn=True),
            nn.Dropout(0.3), pt_utils.FC(256, 40, activation=None))

    def forward(self, points: torch.Tensor):
        batch_size, n_points, _ = points.size()
        end_points = {}

        points = points + self.translation_net(points).unsqueeze(1)
        points, transform = self.apply_transform(
            points, *self.t_net(points.unsqueeze(1)))

        points = self.input_mlp(points.unsqueeze(1))

        points, transform = self.apply_transform(points.squeeze().transpose(
            1, 2), *self.f_net(points))
        end_points['trans2'] = transform

        points = F.max_pool2d(
            self.second_mlp(points.transpose(1, 2).unsqueeze(-1)),
            kernel_size=[n_points, 1])
        return self.final_mlp(points.view(-1, 1024)), end_points

    def set_bn_momentum(self, bn_momentum):
        def fn(m):
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.momentum = bn_momentum

        self.apply(fn)

    def apply_transform(self, points, rotation, scale=None):
        points = points @ rotation
        if scale is not None:
            points = points * scale.contiguous().view(-1, 1, 1).repeat(
                1, points.size(1), points.size(2))

        return points, rotation


if __name__ == "__main__":
    from torch.autograd import Variable
    model = Pointnet()
    data = Variable(torch.randn(2, 10, 3))
    print(model(data))
