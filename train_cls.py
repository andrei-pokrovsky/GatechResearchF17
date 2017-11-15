import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import os
import tensorboard_logger as tb_log

from models import PointnetCls as Pointnet
from models.PointnetCLS import model_fn_decorator
from data import ModelNet40Cls
import utils.pytorch_utils as pt_utils
import utils.data_utils as d_utils
import argparse

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-batch_size", type=int, default=128, help="Batch size [default: 128]")
parser.add_argument(
    "-num_points",
    type=int,
    default=1024,
    help="Number of points to train with [default: 1024]")
parser.add_argument(
    "-weight_decay", type=float, default=1e-5, help="L2 regularization coeff")
parser.add_argument(
    "-lr",
    type=float,
    default=1e-2,
    help="Initial learning rate [default: 1e-2]")
parser.add_argument(
    "-lr_decay",
    type=float,
    default=0.7,
    help="Learning rate decay gamma [default: 0.7]")
parser.add_argument(
    "-decay_step",
    type=int,
    default=20,
    help="Learning rate decay step [default: 20]")
parser.add_argument(
    "-bn_momentum",
    type=float,
    default=0.5,
    help="Initial batch norm momentum [default: 0.5]")
parser.add_argument(
    "-bnm_decay",
    type=float,
    default=0.5,
    help="Batch norm momentum decay gamma [default: 0.5]")
parser.add_argument(
    "-checkpoint", type=str, default=None, help="Checkpoint to start from")
parser.add_argument(
    "-epochs", type=int, default=200, help="Number of epochs to train for")
parser.add_argument(
    "-run_name",
    type=str,
    default="cls_run_1",
    help="Name for run in tensorboard_logger")

lr_clip = 1e-5
bnm_clip = 1e-2

if __name__ == "__main__":
    args = parser.parse_args()

    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    transforms = transforms.Compose([
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudRotate(x_axis=True),
        d_utils.PointcloudScale(),
        d_utils.PointcloudTranslate(),
        d_utils.PointcloudJitter()
    ])

    test_set = ModelNet40Cls(
        args.num_points, BASE_DIR, transforms=transforms, train=False)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True)

    train_set = ModelNet40Cls(args.num_points, BASE_DIR, transforms=transforms)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True)

    tb_log.configure('runs/{}'.format(args.run_name))

    model = Pointnet()
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), lr_clip / args.lr)
    bn_lbmd = lambda e: max(args.bn_momentum * args.bnm_decay**(e // args.decay_step), bnm_clip)

    if args.checkpoint is not None:
        start_epoch, best_prec = pt_utils.load_checkpoint(
            model, optimizer, filename=args.checkpoint.split(".")[0])

        lr_scheduler = lr_sched.LambdaLR(
            optimizer, lr_lambda=lr_lbmd, last_epoch=start_epoch)
        bnm_scheduler = pt_utils.BNMomentumScheduler(
            model, bn_lambda=bn_lbmd, last_epoch=start_epoch)
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = pt_utils.BNMomentumScheduler(model, bn_lambda=bn_lbmd)

        best_prec = 0.0
        start_epoch = 1

    model_fn = model_fn_decorator(nn.CrossEntropyLoss())

    trainer = pt_utils.Trainer(
        model,
        model_fn,
        optimizer,
        checkpoint_name="cls_checkpoint",
        best_name="cls_best",
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler)

    trainer.train(
        start_epoch,
        args.epochs,
        train_loader,
        test_loader,
        best_prec=best_prec)

    if start_epoch == args.epochs:
        _ = trainer.eval_epoch(start_epoch, test_loader)
