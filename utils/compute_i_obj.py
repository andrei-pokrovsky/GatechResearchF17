import torch
from cuda_bridge import pia_wrapper


def compute_iou(preds, boxes) -> torch.Tensor:
    B, N, bpp, _ = preds.size()
    nb = boxes.size(1)

    iou_out = torch.cuda.FloatTensor(B, nb, N, bpp).zero_()
    pia_wrapper(B, N, bpp, nb, preds, boxes, iou_out)

    return iou_out


def compute_i_obj(preds: torch.Tensor, boxes: torch.Tensor,
                  max_trues=256) -> (torch.Tensor, torch.Tensor):
    r"""
    Parameters
    ----------
        preds : torch.Tensor
            (B, N, bpp, 7) Tensor of the predictions for each point in points.  It has bpp boxes per points and each box is parametized
            by [h, w, z, theta, cx, cy, cz]
        boxes : torch.Tensor
            (B, nb, 7) ground truth boxes parametized by [h, w, z, theta, cx, cy, cz]

    Returns
    -------
        i_obj : torch.Tensor
            (B, N, bpp) tensor where 1 denotes that a given box is "responsible" for predicting that ground truth

        int_vol : torch.Tensor
            (B, N, bpp, nb) normalized intersection volume between every predicted box and the ground truth
    """

    B, N, bpp, _ = preds.size()
    nb = boxes.size(1)

    preds = preds.contiguous()
    boxes = boxes.contiguous()

    iou_all = compute_iou(preds, boxes)  # (B, nb, N, bpp)
    iou_out, best_anchor = iou_all.max(dim=-1)  # (B, nb, N)

    positives = torch.cuda.ByteTensor(B, N).zero_()

    vals, best_box = torch.topk(iou_out, k=2, dim=-1)  # (B, nb)
    mask = vals > 1e-3
    best_box = best_box[mask]
    all_b = best_box.new([b for b in range(B)]).view(B, 1, 1).expand(B, nb, 2)
    all_b = all_b[mask]
    positives[all_b, best_box] = 1

    vol, target_box = iou_out.max(dim=1)  # (B, N)
    value, topk_idx = torch.topk(vol, k=max_trues, dim=-1)
    all_b = topk_idx.new([b for b in range(B)]).view(B, 1).expand(B, max_trues)
    mask = value >= 0.7
    topk_idx = topk_idx[mask]
    all_b = all_b[mask]
    if topk_idx.dim() != 0:
        positives[all_b, topk_idx] = 1

    non_positives = positives == 0
    negatives = (vol < 0.35) & non_positives

    best_anchor = torch.gather(
        best_anchor, dim=1, index=target_box.unsqueeze(1)).squeeze(1)

    trues = torch.nonzero(positives)
    if trues.dim() == 0:
        trues = None
    else:
        trues = trues.t()
        trues = torch.cat(
            [trues, best_anchor[trues[0], trues[1]].unsqueeze(0)],
            dim=0).contiguous()

    falses = torch.nonzero(negatives)
    if falses.dim() == 0:
        falses = None
    else:
        falses = falses.t().contiguous()

    return trues, falses, target_box, iou_all.max(dim=1)[0]


import shapely.geometry as geo
import numpy as np


def _bbox_to_rect(bbox):
    pts = torch.zeros(4, 2)
    bx = torch.FloatTensor([np.cos(bbox[3]), -np.sin(bbox[3])])
    by = torch.FloatTensor([np.sin(bbox[3]), np.cos(bbox[3])])
    xy = bbox[4:6]
    wd = bbox[0:2]

    pts[0] = xy + wd * bx + wd * by
    pts[1] = xy - wd * bx + wd * by
    pts[2] = xy - wd * bx - wd * by
    pts[3] = xy + wd * bx - wd * by

    p = geo.Polygon([(pts[i, 0], pts[i, 1]) for i in range(4)])

    return p


if __name__ == "__main__":
    N = 200
    b1 = torch.FloatTensor(N, 7).normal_(2.0, 0.5)
    b2 = b1.clone().normal_(2.0, 0.5)

    iou = compute_iou(b1.view(1, 1, N, 7).cuda(),
                      b2.view(1, N, 7).cuda()).view(N, N).cpu()

    iou_shapely = torch.FloatTensor(N, N)
    for i in range(N):
        box = b2[i]
        box_vol = box[0] * box[1] * box[2] * 2 * 2 * 2
        for j in range(N):
            pred = b1[j]
            pred_vol = pred[0] * pred[1] * pred[2] * 2 * 2 * 2
            z_overlap = min(pred[6] + pred[2], box[6] + box[2]) - max(
                pred[6] - pred[2], box[6] - box[2])

            b_poly = _bbox_to_rect(box)
            p_poly = _bbox_to_rect(pred)
            intersection = b_poly.intersection(p_poly)

            int_vol = z_overlap * intersection.area
            union = box_vol + pred_vol - int_vol

            iou_shapely[i, j] = int_vol / union

    print(((iou_shapely - iou) > 1e-5).sum())
