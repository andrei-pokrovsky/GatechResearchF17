import torch
import torch.nn as nn

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import pytorch_utils as pt_utils
from TransformNets import TransformNet


class Pointnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_dropout = nn.Dropout2d()

        self.t_net = TransformNet(1, 3, 3)
        self.f_net = TransformNet(64, 1, 64)

        self.input_mlp = nn.Sequential(
            pt_utils.Conv2d(1, 64, [1, 3], bn=True),
            pt_utils.Conv2d(64, 64, bn=True))

        self.second_mlp = nn.Sequential(
            pt_utils.SharedMLP([64, 64, 128, 1024], bn=True),
            nn.AdaptiveMaxPool2d((1, 1)))

        self.final_mlp = nn.Sequential(
            pt_utils.FC(1024, 512, bn=True),
            pt_utils.FC(512, 256, bn=True),
            nn.Dropout(0.7), pt_utils.FC(256, 40, activation=None))

    def forward(self, points: torch.Tensor):
        n_points = points.size()[1]
        points = self.init_dropout(points)

        transform = self.t_net(torch.unsqueeze(points, 1))
        points = torch.bmm(points, transform)

        points = self.input_mlp(torch.unsqueeze(points, 1))
        transform = self.f_net(points)
        points = torch.bmm(points.view(-1, n_points, 64), transform).view(
            -1, 64, n_points, 1)

        points = self.second_mlp(points)
        return self.final_mlp(points.view(-1, 1024)), transform


if __name__ == "__main__":
    from torch.autograd import Variable
    model = Pointnet()
    data = Variable(torch.randn(2, 10, 3))
    print(model(data))
