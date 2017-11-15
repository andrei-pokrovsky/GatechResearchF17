import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import numpy as np
import os

from models import BPN
from models.BoxProposalNetwork import model_fn_decorator
from AnchorSet import anchor_set, NUM_ANCHORS
from data import SUNRGBD3DBBox
import utils.pytorch_utils as pt_utils

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for bpn training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "-num_points",
        type=int,
        default=8192 * 8,
        help="Number of points to train with")
    parser.add_argument(
        "-weight_decay",
        type=float,
        default=1e-4,
        help="L2 regularization coeff")
    parser.add_argument(
        "-lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "-lr_gamma", type=float, default=0.8, help="Learning rate decay gamma")
    parser.add_argument(
        "-decay_step",
        type=int,
        default=10,
        help="Decay step size of LR and BNM")
    parser.add_argument(
        "-bnm", type=float, default=0.9, help="Initial batch norm momentum")
    parser.add_argument(
        "-bnm_gamma",
        type=float,
        default=0.7,
        help="Batch norm momentum decay gamma")
    parser.add_argument(
        "-checkpoint", type=str, default=None, help="Checkpoint to start from")
    parser.add_argument(
        "-epochs", type=int, default=300, help="Number of epochs to train for")
    parser.add_argument(
        "-run_name",
        type=str,
        default="bpn_run_1",
        help="Name for run in tensorboard_logger")

    preload_group = parser.add_mutually_exclusive_group()
    preload_group.add_argument(
        "-preload",
        dest='preload',
        help="Whether or not to preload data",
        action='store_true')
    preload_group.add_argument(
        "-nopreload",
        dest='preload',
        help="Whether or not to preload data",
        action='store_false')
    parser.set_defaults(preload=True)

    return parser.parse_args()


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

lr_clip = 5e-4
bnm_clip = 1e-2


def main():
    args = parse_args()

    dataset = SUNRGBD3DBBox(
        BASE_DIR, args.num_points, data_precent=1.0, preload=args.preload)
    all_idx = [i for i in range(len(dataset))]
    train_sampler = SubsetRandomSampler(all_idx[0:int(0.7 * len(dataset))])
    val_sampler = SubsetRandomSampler(all_idx[int(0.7 * len(dataset)):-1])

    B: int = max(args.batch_size, torch.cuda.device_count())
    n_workers: int = 4
    train_loader = DataLoader(
        dataset,
        batch_size=B,
        pin_memory=True,
        num_workers=n_workers,
        shuffle=False,
        sampler=train_sampler)
    val_loader = DataLoader(
        dataset,
        batch_size=B,
        pin_memory=True,
        num_workers=n_workers,
        shuffle=False,
        sampler=val_sampler)

    model = BPN(3, NUM_ANCHORS)
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_lbmd = lambda e: max(args.lr_gamma**(e // args.decay_step), lr_clip / args.lr)
    bnm_lmbd = lambda e: max(args.bnm * args.bnm_gamma**(e // args.decay_step), bnm_clip)

    if args.checkpoint is None:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
        bnm_scheduler = pt_utils.BNMomentumScheduler(model, bnm_lmbd)
        start_epoch = 1
        best_loss = 1e40
    else:
        start_epoch, best_prec = pt_utils.load_checkpoint(
            model, optimizer, filename=args.checkpoint.split(".")[0])

        lr_scheduler = lr_sched.LambdaLR(
            optimizer, lr_lbmd, last_epoch=start_epoch)
        bnm_scheduler = pt_utils.BNMomentumScheduler(
            model, bnm_lmbd, last_epoch=start_epoch)

    model_fn = model_fn_decorator(anchor_set, max_trues=B*64)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(
            model, device_ids=[i for i in range(torch.cuda.device_count())])
    else:
        net = model

    trainer = pt_utils.Trainer(
        net,
        model_fn,
        optimizer,
        checkpoint_name="sem_seg_checkpoint",
        best_name="sem_seg_best",
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler,
        eval_frequency=10,
        log_name="runs/{}".format(args.run_name))

    trainer.train(
        start_epoch,
        args.epochs,
        train_loader,
        val_loader,
        best_loss=best_loss)

    if start_epoch == args.epochs:
        test_loader.dataset.data_precent = 1.0
        _ = trainer.eval_epoch(start_epoch, test_loader)


if __name__ == "__main__":
    main()
