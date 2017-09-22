import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import tensorboard_logger as tb_log
import shutil, os


class SharedMLP(nn.Sequential):
    def __init__(self, args, bn=False, activation=nn.ReLU()):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module('layer{}'.format(i),
                            Conv2d(
                                args[i],
                                args[i + 1],
                                bn=bn,
                                activation=activation))


class _ConvBase(nn.Sequential):
    def __init__(self,
                 in_size,
                 out_size,
                 kernel_size,
                 stride,
                 padding,
                 activation,
                 bn,
                 init,
                 conv=None,
                 batch_norm=None):
        super().__init__()

        self.add_module('conv',
                        conv(
                            in_size,
                            out_size,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=not bn))
        init(self[0].weight)

        if not bn:
            nn.init.constant(self[0].bias, 0)

        if bn:
            self.add_module('bn', batch_norm(out_size))
            nn.init.constant(self[1].weight, 1)
            nn.init.constant(self[1].bias, 0)

        if activation is not None:
            self.add_module('activation', activation)


class Conv1d(_ConvBase):
    def __init__(self,
                 in_size,
                 out_size,
                 kernel_size=(1, 1),
                 stride=(1, 1),
                 padding=(0, 0),
                 activation=nn.ReLU(),
                 bn=False,
                 init=nn.init.kaiming_normal):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            batch_norm=nn.BatchNorm1d)


class Conv2d(_ConvBase):
    def __init__(self,
                 in_size,
                 out_size,
                 kernel_size=(1, 1),
                 stride=(1, 1),
                 padding=(0, 0),
                 activation=nn.ReLU(),
                 bn=False,
                 init=nn.init.kaiming_normal):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=nn.BatchNorm2d)


class Conv3d(_ConvBase):
    def __init__(self,
                 in_size,
                 out_size,
                 kernel_size=(1, 1),
                 stride=(1, 1),
                 padding=(0, 0),
                 activation=nn.ReLU(),
                 bn=False,
                 init=nn.init.kaiming_normal):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv3d,
            batch_norm=nn.BatchNorm3d)


class FC(nn.Sequential):
    def __init__(self,
                 in_size,
                 out_size,
                 activation=nn.ReLU(),
                 bn=False,
                 init=nn.init.kaiming_normal):
        super().__init__()
        self.add_module('fc', nn.Linear(in_size, out_size, bias=not bn))
        init(self[0].weight)

        if not bn:
            nn.init.constant(self[0].bias, 0)

        if bn:
            self.add_module('bn', nn.BatchNorm1d(out_size))
            nn.init.constant(self[1].weight, 1)
            nn.init.constant(self[1].bias, 0)

        if activation is not None:
            self.add_module('activation', activation)


def checkpoint_state(model, optimizer, best_prec, epoch):
    return {
        'epoch': epoch,
        'best_prec': best_prec,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }


def save_checkpoint(state,
                    is_best,
                    filename='checkpoint',
                    bestname='model_best'):
    filename = '{}.pth.tar'.format(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}.pth.tar'.format(bestname))


def load_checkpoint(model, optimizer, filename='checkpoint'):
    filename = "{}.pth.tar".format(filename)
    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        best_prec = checkpoint['best_prec']
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        print("==> Done")
    else:
        print("==> Checkpoint '{}' not found".format(filename))

    return epoch, best_prec


class BNMomentumScheduler(object):
    def __init__(self, model, bn_lambda, last_epoch=-1):
        self.model = model

        if not callable(getattr(self.model, "set_bn_momentum")):
            raise KeyError(
                "Class '{}' has no method 'set_bn_momentum(bn_momentum)'".
                format(type(model).__name__))

        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.set_bn_momentum(self.lmbd(epoch))


class Trainer(object):
    def __init__(self,
                 model,
                 model_fn,
                 optimizer,
                 checkpoint_name="ckpt",
                 best_name="best",
                 lr_scheduler=None,
                 bnm_scheduler=None):
        self.model, self.model_fn, self.optimizer, self.lr_scheduler, self.bnm_scheduler = (
            model, model_fn, optimizer, lr_scheduler, bnm_scheduler)

        self.checkpoint_name, self.best_name = checkpoint_name, best_name

    def _train_epoch(self, epoch, d_loader):
        self.model.train()
        total_loss, total_correct, total_seen, count = (0.0, ) * 4

        next_progress_post = 20.0
        for i, data in enumerate(d_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            self.optimizer.zero_grad()
            _, loss, num_correct = self.model_fn(self.model, inputs, labels)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.data[0]
            count += 1.0

            total_correct += num_correct
            total_seen += labels.numel()

            idx = (epoch - 1) * len(d_loader) + i
            tb_log.log_value("Training loss", loss.data[0], step=idx)
            tb_log.log_value(
                "Training error",
                1.0 - (num_correct / labels.numel()),
                step=idx)

            progress = float(i) / len(d_loader) * 1e2
            if progress > next_progress_post:
                print("Epoch {} progress: [{:<5d} / {:<5d} ({:3.0f})%]".format(
                    epoch, i, len(d_loader), progress))
                next_progress_post += 20.0

            d_loader.dataset.randomize()

        print("[{}] Mean loss: {:2.4f}  Mean Acc: {:2.3f}%".format(
            epoch, total_loss / count, total_correct / total_seen * 1e2))

    def eval_epoch(self, epoch, d_loader):
        self.model.eval()
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
                inputs, labels = Variable(
                    inputs.cuda(), volatile=True), Variable(
                        labels.cuda(), volatile=True)

                _, loss, num_correct = self.model_fn(self.model, inputs,
                                                     labels)

                total_loss += loss.data[0]
                total_correct += num_correct
                total_seen += labels.numel()
                d_loader.dataset.randomize()

            print("[{}, {}] Eval Accuracy: {:2.3f}%\t\tEval Loss: {:1.3f}".
                  format(epoch, num_points, total_correct / total_seen * 1e2,
                         total_loss / len(d_loader)))
            num_points *= 2

            accuracy.append(total_correct / total_seen)
            losses.append(total_loss / len(d_loader))

        return np.mean(accuracy), np.mean(losses)

    def train(self,
              start_epoch,
              n_epochs,
              test_loader,
              train_loader,
              best_prec=0.0):
        for epoch in range(start_epoch, n_epochs + 1):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.bnm_scheduler is not None:
                self.bnm_scheduler.step()

            print("{0}TRAIN{0}".format("-" * 5))
            self._train_epoch(epoch, train_loader)

            print("{0}EVAL{0}".format("-" * 5))
            val_prec, val_loss = self.eval_epoch(epoch, test_loader)
            tb_log.log_value('Validation error', 1.0 - val_prec, epoch)
            tb_log.log_value('Validation loss', val_loss, epoch)

            is_best = val_prec > best_prec
            best_prec = max(val_prec, best_prec)
            save_checkpoint(
                checkpoint_state(self.model, self.optimizer, best_prec, epoch),
                is_best,
                filename=self.checkpoint_name,
                bestname=self.best_name)

        return best_prec
