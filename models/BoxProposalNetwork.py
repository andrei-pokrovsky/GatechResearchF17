import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_utils as pt_utils
from pointnet2_modules import (PointnetSAModule, PointnetFPModule,
                               PointnetSAModuleMSG)
from pointnet2_utils import RandomDropout
from collections import namedtuple
from compute_i_obj import compute_i_obj, compute_iou
import numpy
from TransformNets import TransformNet, TranslationNet


def model_fn_decorator(anchor_set, lmbd=10, n_pretrain_epochs=5,
                       max_trues=256):
    ModelReturn = namedtuple("ModelReturn", ['preds', 'loss', 'eval_dict'])

    l_cls = nn.CrossEntropyLoss(size_average=False)
    l_smooth = nn.SmoothL1Loss(size_average=False)

    k = anchor_set.size(0)
    anchor_set_var = Variable(
        anchor_set, requires_grad=False).cuda().view(1, 1, k,
                                                     anchor_set.size(1))

    def subsample(tensor, n, dim=1):
        if tensor.size(dim) <= n:
            return tensor
        idx = torch.randperm(tensor.size(dim)).type_as(tensor)
        return torch.index_select(tensor, dim=dim, index=idx[0:n])

    def _get_const_tgt(t, fill_val=0):
        return Variable(
            t.data.new(* [1 for _ in range(t.dim())]).fill_(fill_val)
            .expand_as(t))

    def _get_zero_tgt(t):
        return Variable(
            t.data.new(* [1 for _ in range(t.dim())]).zero_().expand_as(t))

    def calc_reg_loss(preds, anchors, targets, trues):
        preds = preds[trues[0], trues[1], trues[2]]
        anchors = anchors[trues[0], trues[1], trues[2]]

        coords_diff = (preds[..., 4:] - targets[..., 4:]) / anchors[..., 0:3]
        scalers_diff = (
            preds[..., 0:3] - targets[..., 0:3]) / anchors[..., 0:3]
        theta_diff = torch.cat(
            [
                preds[..., 3].sin().abs() - targets[..., 3].sin().abs(),
                preds[..., 3].cos().abs() - targets[..., 3].cos().abs(),
            ],
            dim=-1)

        coords_loss = l_smooth(coords_diff, _get_zero_tgt(coords_diff))
        scalers_loss = l_smooth(scalers_diff, _get_zero_tgt(scalers_diff))
        theta_loss = l_smooth(theta_diff, _get_zero_tgt(theta_diff))

        return (coords_loss + scalers_loss + theta_loss) / (
            coords_diff.numel() + scalers_diff.numel() + theta_diff.numel())

    def model_fn(model, data, eval=False, epoch=1):
        xyz, rgb, boxes_truth, _ = data

        B, N, _ = xyz.size()
        number_of_ground_truth_boxes = boxes_truth.size(1)

        xyz = Variable(xyz.cuda(async=True), volatile=eval)
        rgb = Variable(rgb.cuda(async=True), volatile=eval)
        boxes_truth = Variable(
            boxes_truth.cuda(async=True), volatile=eval)  # (B, nb, 7)

        l_xyz, i_obj_preds, conf_preds, boxes_preds = model(
            xyz, rgb)  # (B, np, 3) (B, np, k), (B, np, k, 7)

        number_of_predictions = l_xyz.size(1)
        boxes_preds[..., :4] = boxes_preds[..., :4] + anchor_set_var
        boxes_preds[..., 4:] = boxes_preds[..., 4:] + l_xyz.unsqueeze(2)

        full_anchors = anchor_set_var.clone().expand(B, number_of_predictions,
                                                     k, 4)
        anchor_predictions = torch.cat(
            [
                full_anchors,
                l_xyz.unsqueeze(2).expand(B, number_of_predictions, k, 3)
            ],
            dim=-1)

        closest_box = torch.min(
            (boxes_truth[..., 4:].unsqueeze(2).unsqueeze(2) -
             anchor_predictions[..., 4:].unsqueeze(1)).pow(2).sum(dim=-1),
            dim=1)[1]
        what_t_should_be = torch.gather(
            boxes_truth[..., 4:].unsqueeze(2).unsqueeze(2).expand(
                B, number_of_ground_truth_boxes, number_of_predictions, k, 3),
            index=closest_box.unsqueeze(1).unsqueeze(-1).expand(
                B, 1, number_of_predictions, k, 3),
            dim=1).squeeze(1) - anchor_predictions[..., 4:]
        what_t_should_be = what_t_should_be.max(
            -anchor_predictions[..., 0:3]).min(anchor_predictions[..., 0:3])
        anchor_predictions[
            ..., 4:] = anchor_predictions[..., 4:] + what_t_should_be

        full_trues, full_falses, target_box, target_vol = compute_i_obj(
            anchor_predictions.data,
            boxes_truth.data,
            max_trues=int(
                1.1 * max_trues / B))  # (B, np, k), (B, np, k), (B, np, k)

        if False:
            print((full_trues[0] == 0).sum())

            output_rgb = l_xyz.data.clone().fill_(255)
            output_rgb[full_trues[0], full_trues[1]] = l_xyz.data.new(
                [255, 0, 0]).unsqueeze(0).expand(full_trues.size(1), 3)
            with open("tmp.ptx", "w+") as f:
                num_boxes = 0
                print(boxes_truth[0])
                for i in range(boxes_truth.size(1)):
                    if (boxes_truth[0, i].data.cpu().numpy() != 0).all():
                        num_boxes += 1

                #  num_boxes = min(num_boxes, 3)

                f.write("{}\n1\n".format(l_xyz.size(1) + num_boxes * 8))

                for i in range(l_xyz.size(1)):
                    f.write("{} {} {} {} {} {}\n".format(
                        *l_xyz[0, i].data, *output_rgb[0, i]))

                #  for i in range(xyz.size(1)):
                #  f.write("{} {} {} {} {} {}\n".format(
                #  *xyz[0, i].data, *(rgb[0, i].data * 255.0)))

                for i in range(num_boxes):
                    tmp = boxes_truth[0, i].data.cpu()
                    centroid = tmp[4:]
                    wdh = tmp[0:3]
                    theta = tmp[3]

                    sinval = numpy.sin(theta)
                    cosval = numpy.cos(theta)
                    x_axis = torch.FloatTensor([cosval, -sinval, 0.0])
                    y_axis = torch.FloatTensor([sinval, cosval, 0.0])
                    z_axis = torch.FloatTensor([0.0, 0.0, 1.0])
                    corners = torch.FloatTensor(8, 3)

                    corners[0] = -wdh * x_axis - wdh * y_axis - wdh * z_axis
                    corners[1] = wdh * x_axis - wdh * y_axis - wdh * z_axis
                    corners[2] = -wdh * x_axis + wdh * y_axis - wdh * z_axis
                    corners[3] = -wdh * x_axis + wdh * y_axis - wdh * z_axis

                    corners[4] = -wdh * x_axis - wdh * y_axis + wdh * z_axis
                    corners[5] = wdh * x_axis - wdh * y_axis + wdh * z_axis
                    corners[6] = -wdh * x_axis + wdh * y_axis + wdh * z_axis
                    corners[7] = wdh * x_axis + wdh * y_axis + wdh * z_axis

                    corners += centroid.unsqueeze(0)

                    for j in range(8):
                        f.write("{} {} {} 0 255 0\n".format(*corners[j]))

            sys.exit(1)

        trues = subsample(full_trues, max_trues)
        falses = subsample(full_falses, max_trues * 2 - trues.size(1))

        i_obj_true_loss = l_cls(
            i_obj_preds[trues[0], trues[1]],
            Variable(trues.new(1).fill_(1).expand(trues.size(1))))

        i_obj_false_loss = l_cls(
            i_obj_preds[falses[0], falses[1]],
            Variable(falses.new(1).zero_().expand(falses.size(1))))

        conf_loss = l_cls(conf_preds[trues[0], trues[1]],
                          Variable(trues[2])) / trues.size(1)

        prediction_loss = (i_obj_true_loss + i_obj_false_loss) / (
            trues.size(1) + falses.size(1)) + conf_loss

        trues_targets = boxes_truth[trues[0], target_box[trues[0], trues[1]]]
        reg_loss = calc_reg_loss(boxes_preds, full_anchors, trues_targets,
                                 trues)
        loss = lmbd * reg_loss + prediction_loss

        i_obj = i_obj_preds.data.max(dim=-1)[1]
        target = i_obj.clone().zero_()
        target[trues[0], trues[1]] = 1
        precision = (i_obj == target).sum() / target.numel()
        recall = (i_obj[trues[0], trues[1]] == 1).sum() / trues.size(1)
        i_obj_f1 = 2.0 * (precision * recall) / (precision + recall)

        predicted_best_anchor = torch.max(conf_preds.data, dim=-1)[1]
        best_anchor_acc = 1.0 - torch.sum(
            torch.abs(1.0 - (
                target_vol[trues[0], trues[1],
                           predicted_best_anchor[trues[0], trues[1]]] /
                target_vol[trues[0], trues[1], trues[2]]))) / trues.size(1)

        boxes_preds_data = boxes_preds.clone().detach().data
        boxes_truth_data = boxes_truth.clone().detach().data
        iou_vol = compute_iou(boxes_preds_data, boxes_truth_data)
        tmp = torch.max(
            iou_vol.view(B, number_of_ground_truth_boxes, -1), dim=-1)[0]
        mask = tmp > 1e-5
        tmp = tmp[mask]
        orcale_iou = tmp.sum() / tmp.numel()

        predicted_trues = torch.nonzero(i_obj)
        if predicted_trues.dim() != 0:
            predicted_trues = predicted_trues.t()
            model_iou = torch.sum(target_vol[predicted_trues[
                0], predicted_trues[1], predicted_best_anchor[predicted_trues[
                    0], predicted_trues[1]]]) / predicted_trues.size(1)
        else:
            model_iou = 0.0

        eval_dict = {
            "i_obj_f1": i_obj_f1,
            "anchor_acc": best_anchor_acc,
            "number_of_trues": full_trues.size(1),
            "orcale_iou": orcale_iou,
            "model_iou": model_iou
        }
        return ModelReturn(None, loss, eval_dict)

    return model_fn


class BPNHead(nn.Module):
    def __init__(self, in_c, k):
        super().__init__()

        int_channels = 256
        self.int_layer = nn.Sequential(
            pt_utils.Conv1d(in_c, int_channels, bn=True), nn.Dropout(p=0.0))

        init_lmbd = lambda x: torch.nn.init.normal(x, mean=0.0, std=0.01)
        self.i_obj_head = nn.Sequential(
            pt_utils.Conv1d(
                int_channels, 2, activation=None, bias=True, init=init_lmbd))
        self.conf_head = nn.Sequential(
            pt_utils.Conv1d(
                int_channels, k, activation=None, bias=True, init=init_lmbd))
        self.coordinate_head = nn.Sequential(
            pt_utils.Conv1d(
                int_channels,
                k * 7,
                activation=None,
                bias=True,
                init=init_lmbd))

    def forward(self, X):
        r"""
        Parameters
        ----------
            X : torch.Tensor
                (B, N, in_c) tensor of the descriptors for every point to predict a box of
        Returns
        -------
            i_obj : torch.Tensor
                (B, N, 2) tensor of wether or not that anchor corresponds to an object
            confs : torch.Tensor
                (B, N, k) which anchor is best
            coords : torch.Tensor
                (B, N, k, 7) tensor
        """
        B, N, _ = X.size()

        X = self.int_layer(X.transpose(1, 2))  #(B, int_channels, N)
        confs = self.conf_head(X).view(B, -1, N).permute(0, 2, 1).contiguous()
        i_obj = self.i_obj_head(X).permute(0, 2, 1).contiguous()
        coords = self.coordinate_head(X).view(B, -1, 7, N).permute(
            0, 3, 1, 2).contiguous()

        return i_obj, confs, coords


class BPNBody(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        #  self.initial_dropout = RandomDropout(0.5)
        self.initial_dropout = None

        c_in = input_channels + 3
        self.SA_module0 = PointnetSAModuleMSG(
            npoint=2048,
            radii=[0.1, 0.2],
            nsamples=[64, 128],
            mlps=[[c_in, 32, 64, 128], [c_in, 32, 64, 128]])
        c_out_0 = 128 + 128

        c_in = c_out_0 + 3
        self.SA_module1 = PointnetSAModuleMSG(
            npoint=256,
            radii=[0.2, 0.4],
            nsamples=[16, 32],
            mlps=[[c_in, 64, 64, 128], [c_in, 128, 128, 256]])
        c_out_1 = 128 + 256

        c_in = c_out_1 + 3
        self.SA_module2 = PointnetSAModuleMSG(
            npoint=128,
            radii=[0.4, 0.8],
            nsamples=[16, 32],
            mlps=[[c_in, 128, 128, 256], [c_in, 128, 128, 256]])
        c_out_2 = 256 + 256

        c_in = c_out_2 + 3
        self.SA_module3 = PointnetSAModuleMSG(
            npoint=64,
            radii=[0.8, 1.6],
            nsamples=[16, 32],
            mlps=[[c_in, 256, 256, 512], [c_in, 256, 256, 512]])
        c_out_3 = 512 + 512

        self.FP_module3 = PointnetFPModule(mlp=[c_out_3 + c_out_2, 1024, 1024])
        self.FP_module2 = PointnetFPModule(mlp=[1024 + c_out_1, 1024, 1024])
        self.FP_module1 = PointnetFPModule(
            mlp=[1024 + c_out_0, output_channels])
        #  self.FP_module0 = PointnetFPModule(
        #  mlp=[256 + input_channels, 128, output_channels])

    def forward(self, xyz, points=None):
        """
        Parameters
        ----------
            xyz : torch.Tensor
                (B, N, 3) tensor of the xyz coordinates of 3D points
            points : torch.Tensor
                (B, N, C) tensor of the descriptors of the 3D points

        Returns
        -------
            l0_points : torch.Tensor
                (B, N, output_channels) tensor of the calculated descriptors for each point
        """
        if points is not None and self.initial_dropout is not None:
            tmp = self.initial_dropout(torch.cat([points, xyz], dim=-1))
            points, xyz = tmp.split(points.size(-1), dim=-1)
        elif self.initial_dropout is not None:
            xyz = self.initial_dropout(xyz)

        l0_xyz, l0_points = xyz, points

        l1_xyz, l1_points = self.SA_module0(l0_xyz, l0_points)
        l2_xyz, l2_points = self.SA_module1(l1_xyz, l1_points)
        l3_xyz, l3_points = self.SA_module2(l2_xyz, l2_points)
        l4_xyz, l4_points = self.SA_module3(l3_xyz, l3_points)

        l3_points = self.FP_module3(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.FP_module2(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.FP_module1(l1_xyz, l2_xyz, l1_points, l2_points)
        #  l0_points = self.FP_module0(l0_xyz, l1_xyz, l0_points, l1_points)

        return l1_xyz, l1_points


class BPN(nn.Module):
    def __init__(self, input_channels, k, int_channels=1024):
        super().__init__()

        self.translation_net = TranslationNet()
        self.transform_net = TransformNet(1, 6, 2, scale=True)

        self.body = BPNBody(input_channels, int_channels)
        self.head = BPNHead(int_channels + 3, k)

    def forward(self, xyz, points=None):
        r"""

        Returns
        ------
            l_xyz : torch.Tensor
                (B, N, 3)
            conf : torch.Tensor
                (B, N, k) tensor of the confidence prediction
            coords : torch.Tensor
                (B, N, k, 7) tensor for the coordinate prediction of the bbox
        """
        B, N, _ = xyz.size()

        xyz, points = self.body(xyz, points)
        i_obj, conf, coords = self.head(torch.cat([xyz, points], dim=-1))

        return xyz, i_obj, conf, coords


if __name__ == "__main__":
    from torch.autograd import Variable
    import numpy as np
    import torch.optim as optim
    from AnchorSet import anchor_set

    B = 1
    N = 2048
    k = anchor_set.size(0)
    xyz = torch.randn(B, N, 3)
    xyz[:, 0] = torch.FloatTensor([[0.0, 0.0, 0.0]]).expand(B, 3)
    rgb = torch.randn(B, N, 3)
    boxes_truth = torch.FloatTensor([[[0.5, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0],
                                      [0.0 for _ in range(7)]]])

    data = (xyz, rgb, boxes_truth, None)

    model = BPN(3, anchor_set.size(0))
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model_fn = model_fn_decorator(anchor_set, lmbd=10, n_pretrain_epochs=200)
    for i in range(10000):
        optimizer.zero_grad()
        _, loss, eval_dict = model_fn(model, data, epoch=i)
        loss.backward()
        print(loss.data[0], eval_dict)
        optimizer.step()
