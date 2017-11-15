import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.function import InplaceFunction
from itertools import repeat
import numpy as np
import tensorboard_logger as tb_log
import shutil, os, progressbar
from natsort import natsorted
from operator import itemgetter
from typing import List, Tuple


class SharedMLP(nn.Sequential):
    def __init__(self,
                 args: List[int],
                 *,
                 bn: bool = False,
                 activation=nn.ReLU(inplace=True)):
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
                 batch_norm=None,
                 bias=True):
        super().__init__()

        bias = bias and (not bn)
        self.add_module('conv',
                        conv(
                            in_size,
                            out_size,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=bias))
        init(self[0].weight)

        if bias:
            nn.init.constant(self[0].bias, 0)

        if bn:
            self.add_module('bn', batch_norm(out_size))
            nn.init.constant(self[1].weight, 1)
            nn.init.constant(self[1].bias, 0)

        if activation is not None:
            self.add_module('activation', activation)


class Conv1d(_ConvBase):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 *,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 activation=nn.ReLU(inplace=True),
                 bn: bool = False,
                 init=nn.init.kaiming_normal,
                 bias: bool = True):
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
            batch_norm=nn.BatchNorm1d,
            bias=bias)


class Conv2d(_ConvBase):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 *,
                 kernel_size: Tuple[int, int] = (1, 1),
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (0, 0),
                 activation=nn.ReLU(inplace=True),
                 bn: bool = False,
                 init=nn.init.kaiming_normal,
                 bias: bool = True):
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
            batch_norm=nn.BatchNorm2d,
            bias=bias)


class Conv3d(_ConvBase):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 warning=None,
                 kernel_size: Tuple[int, int, int] = (1, 1, 1),
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 padding: Tuple[int, int, int] = (0, 0, 0),
                 activation=nn.ReLU(inplace=True),
                 bn: bool = False,
                 init=nn.init.kaiming_normal,
                 bias: bool = True):
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
            batch_norm=nn.BatchNorm3d,
            bias=bias)


class FC(nn.Sequential):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 *,
                 activation=nn.ReLU(inplace=True),
                 bn: bool = False,
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


class _DropoutNoScaling(InplaceFunction):
    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @staticmethod
    def symbolic(g, input, p=0.5, train=False, inplace=False):
        if inplace:
            return None
        n = g.appendNode(
            g.create("Dropout", [input]).f_("ratio", p)
            .i_("is_test", not train))
        real = g.appendNode(g.createSelect(n, 0))
        g.appendNode(g.createSelect(n, 1))
        return real

    @classmethod
    def forward(cls, ctx, input, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = cls._make_noise(input)
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise.expand_as(input)
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(Variable(ctx.noise)), None, None, None
        else:
            return grad_output, None, None, None


dropout_no_scaling = _DropoutNoScaling.apply


class _FeatureDropoutNoScaling(_DropoutNoScaling):
    @staticmethod
    def symbolic(input, p=0.5, train=False, inplace=False):
        return None

    @staticmethod
    def _make_noise(input):
        return input.new().resize_(
            input.size(0), input.size(1), *repeat(1, input.dim() - 2))


feature_dropout_no_scaling = _FeatureDropoutNoScaling.apply


def checkpoint_state(model=None, optimizer=None, best_prec=None, epoch=None):
    return {
        'epoch':
        epoch,
        'best_prec':
        best_prec,
        'model_state':
        model.state_dict() if model is not None else None,
        'optimizer_state':
        optimizer.state_dict() if optimizer is not None else None
    }


def save_checkpoint(state,
                    is_best,
                    filename='checkpoint',
                    bestname='model_best'):
    filename = '{}.pth.tar'.format(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}.pth.tar'.format(bestname))


def load_checkpoint(model=None, optimizer=None, filename='checkpoint'):
    filename = "{}.pth.tar".format(filename)
    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        best_prec = checkpoint['best_prec']
        if model is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        print("==> Done")
    else:
        print("==> Checkpoint '{}' not found".format(filename))

    return epoch, best_prec


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):
    def __init__(self,
                 model,
                 bn_lambda,
                 last_epoch=-1,
                 setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(
                type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


class Trainer(object):
    r"""
        Reasonably generic trainer for pytorch models

    Parameters
    ----------
    model : pytorch model
        Model to be trained
    model_fn : function (model, inputs, labels) -> preds, loss, accuracy
    optimizer : torch.optim
        Optimizer for model
    checkpoint_name : str
        Name of file to save checkpoints to
    best_name : str
        Name of file to save best model to
    lr_scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.  .step() will be called at the start of every epoch
    bnm_scheduler : BNMomentumScheduler
        Batchnorm momentum scheduler.  .step() will be called at the start of every epoch
    eval_frequency : int
        How often to run an eval
    log_name : str
        Name of file to output tensorboard_logger to
    """

    def __init__(self,
                 model,
                 model_fn,
                 optimizer,
                 checkpoint_name="ckpt",
                 best_name="best",
                 lr_scheduler=None,
                 bnm_scheduler=None,
                 eval_frequency=1,
                 log_name=None):
        self.model, self.model_fn, self.optimizer, self.lr_scheduler, self.bnm_scheduler = (
            model, model_fn, optimizer, lr_scheduler, bnm_scheduler)

        self.checkpoint_name, self.best_name = checkpoint_name, best_name
        self.eval_frequency = eval_frequency

        if log_name is not None:
            tb_log.configure(log_name)
            self.logging = True
        else:
            self.logging = False

    @staticmethod
    def _print(mode, epoch, loss, eval_dict, count):
        to_print = "[{:d}] {}\tMean Loss: {:.4e}".format(
            epoch, mode, loss / count)
        for k, v in natsorted(eval_dict.items(), key=itemgetter(0)):
            to_print += "\tMean {}: {:2.3f}%".format(k, (v / count) * 1e2)

        print(to_print)

    def _train_epoch(self, epoch, d_loader):
        self.model.train()
        total_loss = 0.0
        eval_dict = {}
        count = 0.0

        bar = progressbar.ProgressBar()
        for i, data in bar(enumerate(d_loader, 0), max_value=len(d_loader)):
            self.optimizer.zero_grad()
            _, loss, eval_res = self.model_fn(self.model, data, epoch=epoch)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.data[0]
            for k, v in eval_res.items():
                eval_dict[k] = v + eval_dict.get(k, 0.0)

            count += 1.0

            if self.logging:
                idx = (epoch - 1) * len(d_loader) + i
                tb_log.log_value("Training loss", loss.data[0], step=idx)
                for k, v in eval_res.items():
                    tb_log.log_value(
                        "Training {}".format(k), 1.0 - v, step=idx)

            d_loader.dataset.randomize()

        self._print("Train", epoch, total_loss, eval_dict, count)

    def eval_epoch(self, epoch, d_loader):
        if d_loader is None:
            return

        self.model.eval()
        total_loss = 0.0
        eval_dict = {}
        count = 0.0

        bar = progressbar.ProgressBar()
        for i, data in bar(enumerate(d_loader, 0), max_value=len(d_loader)):
            self.optimizer.zero_grad()

            _, loss, eval_res = self.model_fn(
                self.model, data, eval=True, epoch=epoch)

            total_loss += loss.data[0]
            count += 1
            for k, v in eval_res.items():
                eval_dict[k] = v + eval_dict.get(k, 0.0)

            if self.logging:
                idx = (epoch - 1) * len(d_loader) + i
                tb_log.log_value("Eval loss", loss.data[0], step=idx)
                for k, v in eval_res.items():
                    tb_log.log_value("Eval {}".format(k), 1.0 - v, step=idx)

            d_loader.dataset.randomize()

        self._print("Eval", epoch, total_loss, eval_dict, count)

        return total_loss / count, eval_dict

    def train(self,
              start_epoch,
              n_epochs,
              train_loader,
              test_loader=None,
              best_loss=0.0):
        r"""
           Call to begin training the model

        Parameters
        ----------
        start_epoch : int
            Epoch to start at
        n_epochs : int
            Number of epochs to train for
        test_loader : torch.utils.data.DataLoader
            DataLoader of the test_data
        train_loader : torch.utils.data.DataLoader
            DataLoader of training data
        best_prec : float
            Testing accuracy of the best model
        """
        for epoch in range(start_epoch, n_epochs + 1):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.bnm_scheduler is not None:
                self.bnm_scheduler.step()

            print("\n{0} Train Epoch {1:0>3d} {0}\n".format("-" * 5, epoch))
            self._train_epoch(epoch, train_loader)

            if test_loader is not None:
                print("\n{0} Eval Epoch {1:0>3d} {0}\n".format("-" * 5, epoch))
                val_loss, _ = self.eval_epoch(epoch, test_loader)

                is_best = val_loss < best_loss
                best_loss = min(best_loss, val_loss)
                save_checkpoint(
                    checkpoint_state(self.model, self.optimizer, val_loss,
                                     epoch),
                    is_best,
                    filename=self.checkpoint_name,
                    bestname=self.best_name)

        return best_prec
