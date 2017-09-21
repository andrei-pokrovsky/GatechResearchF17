import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import pytorch_utils as pt_utils
from AnchorSet import NUM_ANCHORS


class BoxProposalHead(nn.Sequential):
    def __init__(self, k=9):
        super().__init__()
        self.k = k

        self.add_module('MLP',
                        pt_utils.SharedMLP(
                            [1024 + 256 + 64, 1024, 512], bn=True))
        self.add_module('conv', pt_utils.Conv2d(512, 8 * k, activation=None))

        nn.init.constant(self[1][0].weight, 0)
        nn.init.normal(self[1][0].bias, std=1e-3)

    def forward(self, X):
        batch_size, _, n_points, _ = X.size()
        X = super().forward(X)
        return X.squeeze(-1).transpose(1, 2).contiguous().view(batch_size, n_points, self.k, 8)


class ObjectConfidenceHead(nn.Sequential):
    def __init__(self, k=9):
        super().__init__()
        self.k = k

        self.add_module('MLP',
                        pt_utils.SharedMLP(
                            [1024 + 256 + 64, 1024, 256], bn=True))
        self.add_module('conv',
                        pt_utils.Conv2d(256, 2 * k, activation=None))

    def forward(self, X):
        batch_size, _, n_points, _ = X.size()
        X = super().forward(X)
        X = X.squeeze(-1).transpose(1, 2).contiguous().view(batch_size, n_points, self.k, 2)

        X -= X.max()
        X.exp_()
        max_X, _ = X.max(dim=-1)

        return X / max_X.unsqueeze(-1)


class BoxProposalNetwork(nn.Module):
    def __init__(self, k=9):
        super().__init__()

        self.input_mlp = nn.Sequential(
            pt_utils.Conv2d(1, 64, kernel_size=[1, 6], bn=True),
            pt_utils.Conv2d(64, 64, bn=True))

        self.second_mlp = pt_utils.SharedMLP([64, 128, 256], bn=True)

        self.final_mlp = pt_utils.SharedMLP([256, 512, 1024], bn=True)

        self.box_head = BoxProposalHead(NUM_ANCHORS)
        self.confidence_head = ObjectConfidenceHead(NUM_ANCHORS)

    def forward(self, points: torch.Tensor):
        batch_size, n_points, _ = points.size()

        feats1 = self.input_mlp(points.unsqueeze(1))
        feats2 = self.second_mlp(feats1)
        feats3 = F.max_pool2d(
            self.final_mlp(feats2), kernel_size=[n_points, 1])

        stacked_feats = torch.cat(
            [
                feats1, feats2,
                feats3.view(batch_size, -1, 1, 1).repeat(1, 1, n_points, 1)
            ],
            dim=1)
        return self.box_head(stacked_feats), self.confidence_head(stacked_feats)

    def set_bn_momentum(self, bn_momentum):
        def fn(m):
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.momentum = bn_momentum

        self.apply(fn)


if __name__ == "__main__":
    from torch.autograd import Variable
    model = BoxProposalNetwork()
    data = Variable(torch.randn(2, 10, 6))
    print(model(data))
