import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import tensorboard_logger as tb_log

from models.ObjectProposalNetwork import ObjectProposalNetwork
from data.MultiObjectLoader import MultiObjectLoader
import models.pytorch_utils as pt_utils

fresh = True

BATCH_SIZE = 32
NUM_POINT = 512

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

test_set = MultiObjectLoader(NUM_POINT, BASE_DIR, train=False)
test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True)

tb_log.configure('runs/opn-run-1')

best_prec = 0.0
start_epoch = 1

DECAY_STEP = 20
lr_init = 1e-2
lr_clip = 1e-5
lr_decay = 0.7

model = ObjectProposalNetwork()
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=1e-9)
lr_scheduler = lr_sched.LambdaLR(
    optimizer, lambda e: max(lr_decay**(e // DECAY_STEP), lr_clip))

if not fresh:
    start_epoch, best_prec = pt_utils.load_checkpoint(
        model, optimizer, filename='cls_best')
    lr_scheduler.step(start_epoch)

bn_momentum_clip = 0.01
bn_momentum_decay = 0.5


def get_bn_momentum(epoch):
    bn_init = 0.5
    bn_momentum = bn_init * (bn_momentum_decay**((epoch - 1) // DECAY_STEP))
    return max(bn_momentum, bn_momentum_clip)

def model_fn_decorator(criterion):
    def wrapped(model, inputs, labels):
        preds = model(inputs)
        preds = preds.contiguous()

        loss = criterion(preds.view(-1, 10), labels.view(-1))

        return loss, preds

    return wrapped

model_fn = model_fn_decorator(nn.CrossEntropyLoss())


NUM_EPOCHS = 300


def train():
    global best_prec
    if train:

        train_set = MultiObjectLoader(NUM_POINT, BASE_DIR)
        train_loader = DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True)

        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            lr_scheduler.step()

            print("{0}TRAIN{0}".format("-" * 5))
            train_epoch(epoch, optimizer, model_fn, train_loader)

            print("{0}EVAL{0}".format("-" * 5))
            val_prec, val_loss = eval_epoch(epoch, model_fn, test_loader)
            tb_log.log_value('validation error', 1.0 - val_prec, epoch - 1)
            tb_log.log_value("validation loss", val_loss, epoch - 1)

            is_best = val_prec > best_prec
            best_prec = max(val_prec, best_prec)
            pt_utils.save_checkpoint(
                pt_utils.checkpoint_state(model, optimizer, best_prec, epoch),
                is_best,
                filename='cls_checkpoint',
                bestname='cls_best')



def train_epoch(epoch, optimizer, model_fn, train_loader):
    model.train()
    model.set_bn_momentum(get_bn_momentum(epoch))
    total_loss, total_correct, total_seen, count = (0.0, ) * 4

    next_post = 20.0

    for i, data in enumerate(train_loader, 0):
        train_loader.dataset.randomize()

        inputs, labels, _ = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        optimizer.zero_grad()
        loss, preds = model_fn(model, inputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.data[0]
        count += 1.0

        _, classes = torch.max(preds.data, 2)
        num_correct = (classes == labels.data).sum()
        total_correct += num_correct
        total_seen += classes.numel()

        log_idx = (epoch - 1) * len(train_loader) + i
        tb_log.log_value('training loss', loss.data[0], log_idx)
        tb_log.log_value('training error',
                         1.0 - num_correct / float(inputs.numel()), log_idx)

        progress = float(i + 1) / len(train_loader) * 1e2
        if progress >= next_post:
            print("Epoch {} progress: [{:<5d} / {:<5d} ({:3.0f}%)]".format(
                epoch, i, len(train_loader), progress))
            next_post = next_post + 20.0

    print("[{}, {}] Mean loss: {:2.4f}  Mean Acc: {:2.3f}%".format(
        epoch, i, total_loss / count, total_correct / total_seen * 1e2))


def eval_epoch(epoch, model_fn, d_loader):
    model.eval()
    num_points = 2**7
    losses, accuracies = [], []
    while num_points <= 2**11:
        d_loader.dataset.set_num_points(num_points)
        total_correct = 0.0
        total_seen = 0.0
        total_loss = 0.0
        for i, data in enumerate(d_loader, 0):
            d_loader.dataset.randomize()
            inputs, labels, _ = data
            inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)

            model.zero_grad()
            loss, preds = model_fn(model, inputs, labels)

            _, classes = torch.max(preds.data, 2)
            total_correct += (classes == labels.data).sum()
            total_seen += classes.numel()
            total_loss += loss.data[0]

        print("[{}, {}] Eval Loss: {:2.4f}\tEval Accuracy: {:2.3f}%".format(
            epoch, num_points, total_loss / len(d_loader), total_correct / total_seen * 1e2))
        num_points *= 2

        accuracies.append(total_correct / total_seen)
        losses.append(total_loss / len(d_loader))

    return np.mean(accuracies), np.mean(losses)


if __name__ == "__main__":
    train()
    eval_epoch(-1, model_fn, test_loader)
