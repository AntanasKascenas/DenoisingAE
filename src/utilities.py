#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
import torch
from torch.nn.modules.utils import _pair, _quadruple
import torch.nn.functional as F


def smaller(x1, x2, min_delta=0):
    return x2 - x1 > min_delta


def bigger(x1, x2, min_delta=0):
    return x1 - x2 > min_delta


class ModelSaver:

    def __init__(self, get=lambda trainer: trainer.state["val_loss"], better=smaller, path=None):

        self.get = get
        self.better = better
        self.path = path

        self.best = None

    def save(self, trainer):
        trainer.save(path=None, value=self.best)

    def check(self, trainer):

        res = self.get(trainer)

        if self.best is None:
            self.best = res
            self.save(trainer)

        else:
            if self.better(res, self.best):
                self.save(trainer)
                self.best = res

    def register(self, callback_dict):
        callback_dict["after_epoch"].append(lambda trainer: self.check(trainer))


def move_to(list, device):
    return [x.to(device) if isinstance(x, torch.Tensor) else x for x in list]


def median_pool(x, kernel_size=3, stride=1, padding=0):
    k = _pair(kernel_size)
    stride = _pair(stride)
    padding = _quadruple(padding)

    x = F.pad(x, padding, mode='reflect')
    x = x.unfold(2, k[0], stride[0]).unfold(3, k[1], stride[1])
    x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]

    return x