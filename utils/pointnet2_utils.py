import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
from linalg_utils import pdist2, PDist2Order
from collections import namedtuple
import pointnet2
import pytorch_utils as pt_utils


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        r"""
        Uses iterative furthest point sampling to select a set of npoint points that have the largest
        minimum distance

        Parameters
        ---------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of points in the sampled set

        Returns
        torch.Tensor
            (B, npoint, 3) tensor containing the set
        ------
        """
        B, N, _ = xyz.size()

        output = torch.cuda.FloatTensor(npoint, B, 3)
        output[0].copy_(xyz[:, 0])

        idx = torch.cuda.LongTensor(B)
        scratch = torch.cuda.FloatTensor(B)

        distance_to_closet_in_set = torch.cuda.FloatTensor(B, N).fill_(1e38)
        scratch2 = torch.cuda.FloatTensor(B, N)

        all_b = torch.cuda.LongTensor([b for b in range(B)])

        for current_n in range(1, npoint):
            torch.min(
                distance_to_closet_in_set,
                pdist2(xyz, output[current_n - 1]),
                out=scratch2)
            distance_to_closet_in_set.copy_(scratch2)
            torch.max(distance_to_closet_in_set, dim=1, out=(scratch, idx))
            output[current_n].copy_(xyz[all_b, idx])

        return output.transpose(0, 1).contiguous()

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown: torch.Tensor,
                known: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known points
        known : torch.Tensor
            (B, m, 3) tensor of unknown points

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        B, N, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        unknown = unknown.contiguous()
        known = known.contiguous()
        dist2 = dist2.contiguous()
        idx = idx.contiguous()
        pointnet2.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, points: torch.Tensor, idx: torch.Tensor,
                weight: torch.Tensor) -> torch.Tensor:
        r"""
            Performs weight linear interpolation on 3 points
        Parameters
        ----------
        points : torch.Tensor
            (B, m, c)  Points to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target points in points
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, n, c) tensor of the interpolated points
        """

        B, m, c = points.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        output = torch.cuda.FloatTensor(B, n, c)

        points = points.contiguous()
        idx = idx.contiguous()
        weight = weight.contiguous()
        output = output.contiguous()
        pointnet2.three_interpolate_wrapper(B, m, c, n, points, idx, weight,
                                            output)

        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor
                 ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, n, c) tensor with gradients of ouputs

        Returns
        -------
        grad_points : torch.Tensor
            (B, m, c) tensor with gradients of points
        None

        None
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, n, c = grad_out.size()

        grad_points = Variable(torch.cuda.FloatTensor(B, m, c).zero_())

        grad_out = grad_out.contiguous()
        idx = idx.contiguous()
        weight = weight.contiguous()
        grad_points = grad_points.contiguous()
        pointnet2.three_interpolate_grad_wrapper(B, n, c, m, grad_out.data,
                                                 idx, weight, grad_points.data)

        return grad_points, None, None


three_interpolate = ThreeInterpolate.apply


class GroupPoints(Function):
    @staticmethod
    def forward(ctx, points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        r"""

        Parameters
        ----------
        points : torch.Tensor
            (B, N, C) tensor of points to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of points to group with

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample, C) tensor
        """
        B, npoints, nsample = idx.size()
        _, N, C = points.size()

        output = torch.cuda.FloatTensor(B, npoints, nsample, C)

        points = points.contiguous()
        idx = idx.contiguous()
        output = output.contiguous()
        pointnet2.group_points_wrapper(B, N, C, npoints, nsample, points, idx,
                                       output)

        ctx.idx_N_C_for_backward = (idx, N, C)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, npoint, nsample, C) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, N, C) gradient of the points
        None
        """
        idx, N, C = ctx.idx_N_C_for_backward

        B, npoint, nsample, _ = grad_out.size()
        grad_points = Variable(torch.cuda.FloatTensor(B, N, C).zero_())

        grad_out = grad_out.contiguous()
        grad_points = grad_points.contiguous()
        pointnet2.group_points_grad_wrapper(
            B, N, C, npoint, nsample, grad_out.data, idx, grad_points.data)

        return grad_points, None


group_points = GroupPoints.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor,
                new_xyz: torch.Tensor) -> torch.Tensor:
        r"""

        Parameters
        ---------
        radius : float
            radius of the balls
        nsample : int
            maximum number of points in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the points
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        ------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the points that form the query balls
        """

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

        new_xyz = new_xyz.contiguous()
        xyz = xyz.contiguous()
        idx = idx.contiguous()
        pointnet2.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz,
                                     xyz, idx)

        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class SampleAndGroup(nn.Module):
    r"""
    Samples points using FPS, groups with a ball query of radius

    Parameters
    ---------
    npoint : int32
        Number of points to sample during the FPS sampling
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of points to gather in the ball
    """

    def __init__(self,
                 npoint: int,
                 radius: float,
                 nsample: int,
                 use_xyz: bool = True):
        super().__init__()
        self.npoint, self.radius, self.nsample, self.use_xyz = npoint, radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor,
                points: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ---------
        xyz : torch.Tensor
            xyz coordinates of the points (B, N, 3)
        points : torch.Tensor
            Descriptors of the points (B, N, C)

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor
        new_points : torch.Tensor
            (B, npoint, nsample, 3 + C) tensor
        """
        new_xyz = furthest_point_sample(xyz, self.npoint)  # (B, npoint, 3)

        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = group_points(xyz, idx)  # (B, npoint, nsample, 3)
        grouped_xyz -= new_xyz.unsqueeze(2)

        if points is not None:
            grouped_points = group_points(points, idx)
            if self.use_xyz:
                new_points = torch.cat(
                    [grouped_xyz, grouped_points],
                    dim=-1)  # (B, npoint, nsample, 3 + C)
            else:
                new_points = group_points
        else:
            new_points = grouped_xyz

        return new_xyz, new_points


class PointnetSAModule(nn.Module):
    r"""
    Pointnet set abstrction layer
    Parameters
    ----------
    npoint : int
        Number of points
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(self,
                 npoint: int,
                 radius: float,
                 nsample: int,
                 mlp: list,
                 bn: bool = True):
        super().__init__()

        self.grouper = SampleAndGroup(npoint, radius, nsample)
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, xyz: torch.Tensor,
                points: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the points
        point : torch.Tensor
            (B, N, C) tensor of the descriptors of the the points

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new points' xyz
        new_points : torch.Tensor
            (B, npoint, mlp[-1]) tensor of the new_points descriptors
        """

        new_xyz, new_points = self.grouper(
            xyz, points)  # (B, npoint, 3), (B, npoint, nsample, 3 + C)
        new_points = self.mlp(new_points.permute(
            0, 3, 1, 2))  # (B, mlp[-1], npoint, nsample)
        new_points = F.max_pool2d(
            new_points,
            kernel_size=[1, new_points.size(3)])  # (B, mlp[-1], npoint, 1)
        new_points = new_points.squeeze(-1)  # (B, mlp[-1], npoint)
        new_points = new_points.transpose(
            1, 2).contiguous()  # (B, npoint, mlp[-1])

        return new_xyz, new_points


class PointnetFPModule(nn.Module):
    r"""
        Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp: list, bn: bool = True):
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor,
                unknow_feats: torch.Tensor,
                known_feats: torch.Tensor) -> torch.Tensor:
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown points
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known points
        unknow_feats : torch.Tensor
            (B, n, C1) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, m, C2) tensor of features to be propigated

        Returns
        -------
        new_points : torch.Tensor
            (B, n, mlp[-1]) tensor of the features of the unknown points
        """

        dist, idx = three_nn(unknown, known)
        dist += 1e-10
        dist_recip = 1.0 / dist
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm

        interpolated_feats = three_interpolate(known_feats, idx, weight)
        if unknow_feats is not None:
            new_points = torch.cat(
                [interpolated_feats, unknow_feats], dim=-1)  #(B, n, C2 + C1)
        else:
            new_points = interpolated_feats

        new_points = new_points.unsqueeze(-1).transpose(1,
                                                        2)  #(B, C2 + C1, n, 1)
        new_points = self.mlp(new_points)

        return new_points.squeeze(-1).transpose(
            1, 2).contiguous()  #(B, n, mlp[-1])


if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = Variable(torch.randn(2, 10, 3).cuda(), requires_grad=True)
    xyz_feats = Variable(torch.randn(2, 2, 6).cuda(), requires_grad=True)

    test_module = PointnetSAModule(1, radius=10.0, nsample=3, mlp=[3, 3])
    test_module.cuda()
    print(test_module(xyz))

    #  test_module = PointnetFPModule(mlp=[6, 6])
    #  test_module.cuda()
    #  from torch.autograd import gradcheck
    #  inputs = (xyz, xyz, None, xyz_feats)
    #  test = gradcheck(test_module, inputs, eps=1e-6, atol=1e-4)
    #  print(test)

    for _ in range(1):
        _, new_points = test_module(xyz)
        new_points.backward(
            torch.cuda.FloatTensor(*new_points.size()).fill_(1))
        print(new_points)
        print(xyz.grad)
