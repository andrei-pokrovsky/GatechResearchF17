import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))

import torch
import torch.nn as nn
import pytorch_utils as pt_utils
from pointnet2_utils import PointnetSAModule, PointnetFPModule
from collections import namedtuple


def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ['preds', 'loss', 'acc'])

    def model_fn(model, inputs, labels):
        xyz = inputs[..., :3]
        if inputs.size(2) > 3:
            points = inputs[..., 3:]
        else:
            points = None

        preds = model(xyz, points)
        loss = criterion(preds.view(labels.numel(), -1), labels.view(-1))

        _, classes = torch.max(preds.data, 2)
        acc = (classes == labels.data).sum()

        return ModelReturn(preds, loss, acc)

    return model_fn


class Pointnet2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.SA_module0 = PointnetSAModule(1024, 0.1, 32, mlp=[9, 32, 32, 64])
        self.SA_module1 = PointnetSAModule(
            256, 0.2, 32, mlp=[64 + 3, 64, 64, 128])
        self.SA_module2 = PointnetSAModule(
            64, 0.4, 32, mlp=[128 + 3, 128, 128, 256])
        self.SA_module3 = PointnetSAModule(
            16, 0.8, 32, mlp=[256 + 3, 256, 256, 512])

        self.FP_module0 = PointnetFPModule(mlp=[512 + 256, 256, 256])
        self.FP_module1 = PointnetFPModule(mlp=[256 + 128, 256, 256])
        self.FP_module2 = PointnetFPModule(mlp=[256 + 64, 256, 128])
        self.FP_module3 = PointnetFPModule(mlp=[128 + 6, 128, 128, 128])

        self.FC_1 = pt_utils.Conv1d(128, 128, bn=True)
        self.dropout = nn.Dropout()
        self.FC_2 = pt_utils.Conv1d(128, num_classes, activation=None)
        self.FC_layer = nn.Sequential(
            pt_utils.Conv1d(128, 128, bn=True),
            nn.Dropout(), pt_utils.Conv1d(128, num_classes, activation=None))

    def forward(self, xyz, points=None):
        l0_xyz = xyz
        l0_points = points

        l1_xyz, l1_points = self.SA_module0(l0_xyz, l0_points)
        l2_xyz, l2_points = self.SA_module1(l1_xyz, l1_points)
        l3_xyz, l3_points = self.SA_module2(l2_xyz, l2_points)
        l4_xyz, l4_points = self.SA_module3(l3_xyz, l3_points)

        l3_points = self.FP_module0(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.FP_module1(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.FP_module2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.FP_module3(l0_xyz, l1_xyz, l0_points,
                                    l1_points).transpose(1, 2)

        return self.FC_layer(l0_points).transpose(1, 2).contiguous()

    def set_bn_momentum(self, bn_momentum):
        def fn(m):
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.momentum = bn_momentum

        self.apply(fn)


if __name__ == "__main__":
    from torch.autograd import Variable
    import numpy as np
    import torch.optim as optim
    B = 64
    N = 1024
    inputs = Variable(torch.randn(B, N, 9).cuda())
    labels = Variable(
        torch.from_numpy(np.random.randint(0, 3, size=B * N)).view(B, N)
        .cuda())
    model = Pointnet2(3)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-1)

    model_fn = model_fn_decorator(nn.CrossEntropyLoss())
    for _ in range(20):
        optimizer.zero_grad()
        _, loss, _ = model_fn(model, inputs, labels)
        loss.backward()
        print(loss)
        optimizer.step()
