import torch
import torch.nn as nn
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
                            padding=padding, bias=not bn))
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
