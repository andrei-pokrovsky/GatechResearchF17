import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import pytorch_utils as pt_utils
from TransformNets import TransformNet


def model_fn_decorator(criterion):
    def model_fn(model, inputs, labels):
        preds = model(inputs)
        loss = criterion(preds.view(labels.numel(), -1), labels.view(-1))

        _, classes = torch.max(preds.data, 2)
        acc = (classes == labels.data).sum()

        return preds, loss, acc

    return model_fn


class Pointnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_mlp = nn.Sequential(
            pt_utils.Conv2d(1, 64, [1, 9], bn=True),
            pt_utils.SharedMLP([64, 64, 128, 1024], bn=True))

        self.feat1_fc = nn.Sequential(
            pt_utils.FC(1024, 256, bn=True), pt_utils.FC(256, 128, bn=True))

        self.final_convs = nn.Sequential(
            pt_utils.SharedMLP([1024 + 128, 512, 256], bn=True),
            nn.Dropout2d(p=0.3),
            pt_utils.Conv2d(256, 13, activation=None))

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal(
                    param, gain=nn.init.calculate_gain('relu'))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        batch_size, n_points, _ = points.size()

        points = torch.unsqueeze(points, 1)
        points = self.input_mlp(points)

        pc_feat1 = self.feat1_fc(
            F.adaptive_max_pool2d(points, [1, 1]).view(batch_size, -1))

        pc_feat1_expand = pc_feat1.view(batch_size, -1, 1, 1).repeat(
            1, 1, n_points, 1)

        points_feat1_concat = torch.cat((points, pc_feat1_expand), dim=1)

        return self.final_convs(points_feat1_concat).squeeze(-1).transpose(
            1, 2).contiguous()

    def set_bn_momentum(self, bn_momentum):
        def fn(m):
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.momentum = bn_momentum

        self.apply(fn)


if __name__ == "__main__":
    from torch.autograd import Variable
    model = Pointnet()
    data = Variable(torch.randn(2, 3, 9))
    print(model(data).view(-1, 13))
