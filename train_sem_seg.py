import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import numpy as np
import tensorboard_logger as tb_log
import os

from models.PointnetSEMSEG import Pointnet
from data.Indoor3DSemSegLoader import Indoor3DSemSeg
import models.pytorch_utils as pt_utils

fresh = True

BATCH_SIZE = 32
NUM_POINT = 2048

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

test_set = Indoor3DSemSeg(NUM_POINT, BASE_DIR, train=False)
test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=1)

DECAY_STEP = 20
lr_init = 1e-2
lr_clip = 1e-5
lr_decay = 0.5

model = Pointnet()
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=1e-4)
lr_scheduler = lr_sched.LambdaLR(
    optimizer,
    lambda e: max(lr_decay**(e // DECAY_STEP), lr_clip))
epoch_start = 1
best_prec = 0

tb_log.configure('runs/run-1')

if not fresh:
    epoch_start, best_prec = pt_utils.load_checkpoint(
        model, optimizer, filename='sem_seg_best')
    lr_scheduler.step(epoch_start)



def get_bn_momentum(epoch):
    bn_init = 0.5
    bn_momentum = bn_init * (0.5**((epoch - 1) // DECAY_STEP))
    return max(bn_momentum, 0.01)


NUM_EPOCHS = 200


def train():
    global best_prec
    if train:
        criterion = nn.CrossEntropyLoss()

        train_set = Indoor3DSemSeg(NUM_POINT, BASE_DIR)
        train_loader = DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            pin_memory=True,
            num_workers=1,
            shuffle=True)

        for epoch in range(epoch_start, NUM_EPOCHS + 1):
            print("{0}TRAIN{0}".format("-" * 5))
            train_prec, train_loss = train_epoch(epoch, optimizer, criterion,
                                                 train_loader)
            tb_log.log_value('training accuracy', train_prec, epoch)
            tb_log.log_value('training loss', train_loss, epoch)

            print("{0}EVAL{0}".format("-" * 5))
            val_prec, val_loss = eval_epoch(epoch, test_loader, criterion)
            tb_log.log_value('validation accuracy', val_prec, epoch)
            tb_log.log_value('validation loss', val_loss, epoch)

            is_best = val_prec > best_prec
            best_prec = max(val_prec, best_prec)
            pt_utils.save_checkpoint(
                pt_utils.checkpoint_state(model, optimizer, best_prec, epoch),
                is_best,
                filename='sem_seg_checkpoint',
                bestname='sem_seg_best')

            lr_scheduler.step()


def train_epoch(epoch, optimizer, criterion, train_loader):
    model.train()
    model.set_bn_momentum(get_bn_momentum(epoch))
    total_loss, total_correct, total_seen, count = (0.0, ) * 4

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        labels = Variable(labels.cuda())

        optimizer.zero_grad()
        preds = model(Variable(inputs.cuda()))

        long_labels = labels.view(-1)
        loss = criterion(preds.view(long_labels.size(0), -1), long_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.data[0]
        count += 1.0

        _, classes = torch.max(preds.data, 2)
        total_correct += (classes.view(-1) == long_labels.data).sum()
        total_seen += long_labels.size()[0]

        if (i + 1) % (len(train_loader) // 5) == 0:
            print("Epoch {} progress: [{:<5d} / {:<5d} ({:3.0f})%]".format(
                epoch, i, len(train_loader), float(i) / len(train_loader) *
                1e2))

    print("[{}] Mean loss: {:2.4f}  Mean Acc: {:2.3f}%".format(
        epoch, total_loss / count, total_correct / total_seen * 1e2))

    return total_correct / total_seen, total_loss / count


def eval_epoch(epoch, d_loader, criterion=None):
    model.eval()
    num_points = 2**7
    accuracy = []
    losses = []
    while num_points <= 2**12:
        d_loader.dataset.set_num_points(num_points)
        total_correct = 0.0
        total_seen = 0.0
        total_loss = 0.0
        for i, data in enumerate(d_loader, 0):
            inputs, labels = data
            labels = labels.cuda()

            model.zero_grad()
            preds = model(Variable(inputs.cuda(), requires_grad=False))
            preds.detach_()

            if criterion is not None:
                long_labels = labels.view(-1)
                total_loss += criterion(
                    preds.view(long_labels.size(0), -1),
                    Variable(long_labels, requires_grad=False)).data[0]

            _, classes = torch.max(preds.data, 2)
            total_correct += (classes.view(-1) == labels.view(-1)).sum()
            total_seen += inputs.size()[0] * inputs.size()[1]

        if criterion is None:
            print("[{}, {}] Eval Accuracy: {:2.3f}%".format(
                epoch, num_points, total_correct / total_seen * 1e2))
        else:
            print("[{}, {}] Eval Accuracy: {:2.3f}%\t\tEval Loss: {:1.3f}".
                  format(epoch, num_points, total_correct / total_seen * 1e2,
                         total_loss / len(d_loader)))
        num_points *= 2

        accuracy.append(total_correct / total_seen)
        losses.append(total_loss / len(d_loader))

    if criterion is None:
        return np.mean(accuracy)
    else:
        return np.mean(accuracy), np.mean(losses)


if __name__ == "__main__":
    train()
    _ = eval_epoch(-1, test_loader)
