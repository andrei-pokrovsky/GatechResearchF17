import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
from linalg_utils import pdist2
from collections import namedtuple
import pointnet2
import pytorch_utils as pt_utils


def furthest_point_sample(xyz: torch.Tensor,
                          npoint: int,
                          dmat: torch.Tensor = None) -> torch.Tensor:
    r"""
    Uses iterative furthest point sampling to select a set of npoint points that have the largest
    minimum distance

    Parameters
    ---------
    xyz : torch.Tensor
        (B, N, 3) tensor where N > npoint
    npoint : int32
        number of points in the sampled set
    dmat : torch.Tensor
        (B, N, N) tensor where dmat[b, i, j] = torch.dist(xyz[b, i], xyz[b, j], p=2)
        If it is none, it will be calculated

    Returns
    torch.Tensor
        (B, npoint) tensor containing the set
    ------
    """
    B, N, _ = xyz.size()
    if dmat is None: dmat = pdist2(xyz.data)

    _, furthest = torch.max(dmat.view(B, -1), dim=1)

    j_0 = (furthest / N)
    i_0 = (furthest % N)

    output = torch.LongTensor(npoint, B).type_as(j_0)
    all_b = torch.LongTensor([b for b in range(B)]).type_as(j_0)

    output[0] = j_0
    output[1] = i_0

    distance_to_closet_in_set = torch.min(dmat[all_b, i_0], dmat[all_b, j_0])
    for current_n in range(2, npoint):
        _, i = torch.max(distance_to_closet_in_set, dim=1)
        output[current_n] = i
        distance_to_closet_in_set = torch.min(distance_to_closet_in_set,
                                              dmat[all_b, i])

    return output.transpose_(0, 1)


def three_nn(unknown: torch.Tensor,
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
    dmat = pdist2(unknown, known)

    dist, idx = torch.sort(dmat, dim=2)

    return dist[..., :3], idx[..., :3]


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

        ctx.idx_N_C_for_backward = (idx, N, C)
        output = torch.cuda.FloatTensor(B, npoints, nsample, C)

        pointnet2.group_points_wrapper(B, N, C, npoints, nsample, points, idx,
                                       output)

        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
        idx, N, C = ctx.idx_N_C_for_backward

        B, npoint, nsample, _ = grad_out.size()
        grad_points = Variable(torch.cuda.FloatTensor(B, N, C).zero_())

        pointnet2.group_points_grad_wrapper(B, N, C, npoint, nsample,
                                            grad_out.data, idx,
                                            grad_points.data)

        return grad_points, None


group_points = GroupPoints.apply


def ball_query(radius: float,
               nsample: int,
               xyz: torch.Tensor,
               new_xyz: torch.Tensor,
               dmat: torch.Tensor = None) -> torch.Tensor:
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
    dmat : torch.Tensor
        (B, npoint, N) tensor where dmat[b, i, j] = torch.dist(new_xyz[b, i], xyz[b, j], p=2)
        If it is none, it will be calculated

    Returns
    ------
    torch.Tensor
        (B, npoint, nsample) tensor with the indicies of the points that form the query balls
    """

    if dmat is None: dmat = pdist2(new_xyz, xyz)

    B, N, _ = xyz.size()
    npoint = new_xyz.size(1)
    idx = torch.cuda.LongTensor(B, npoint, nsample)
    pointnet2.ball_query_wrapper(B, N, npoint, radius, nsample, dmat, idx)

    return idx


def gather_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    r"""
    Parameters
    ----------
    points : torch.Tensor
        (B, N, C) tensor to gather from
    idx : torch.Tensor
        (B, npoint) tensor of the indicies to gather from points

    Returns
    -------
    torch.Tensor
        (B, npoint, C) tensor
    """
    idx = idx.unsqueeze(-1).repeat(1, 1, points.size(2))
    return torch.gather(points, dim=1, index=idx)


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
        new_xyz = gather_points(xyz, furthest_point_sample(
            xyz, self.npoint))  # (B, npoint, 3)

        idx = ball_query(self.radius, self.nsample, xyz.data, new_xyz.data)
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
        new_points = self.mlp(new_points.permute(0, 3, 1, 2))  # (B, mlp[-1], npoint, nsample)
        new_points = F.max_pool2d(
            new_points,
            kernel_size=[1, new_points.size(3)])  # (B, mlp[-1], npoint, 1)
        new_points = new_points.squeeze(-1)  # (B, mlp[-1], npoint)
        new_points = new_points.transpose(1, 2)  # (B, npoint, mlp[-1])

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
        norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
        weight = (1.0 / dist) / norm

        nn_1 = gather_points(known_feats, idx[..., 0])
        nn_2 = gather_points(known_feats, idx[..., 1])
        nn_3 = gather_points(known_feats, idx[..., 2])

        interpolated_feats = nn_1 * weight[..., 0].unsqueeze(-1) + nn_2 * weight[..., 1].unsqueeze(-1) + nn_3 * weight[..., 2].unsqueeze(-1) # (B, n, C2)

        if unknow_feats is not None:
            new_points = torch.cat([interpolated_feats, unknow_feats], dim=-1) #(B, n, C2 + C1)
        else:
            new_points = interpolated_feats

        new_points = new_points.unsqueeze(-1).transpose(1, 2) #(B, C2 + C1, n, 1)
        new_points = self.mlp(new_points)

        return new_points.squeeze(-1).transpose(1, 2) #(B, n, mlp[-1])

if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = Variable(torch.randn(3, 5, 3).cuda())

    test_module = PointnetFPModule(mlp=[6, 4, 5])
    test_module.cuda()
    print(test_module(xyz, xyz, xyz, xyz))
    #  test_module = PointnetSAModule(3, 1.5, 2, mlp=[6, 4, 5, 6])
    #  test_module.cuda()
    #  print(test_module(xyz, xyz))
