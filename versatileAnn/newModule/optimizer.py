import copy

import torch

from utils.typeCheck import argValidator


class _NewWrapper_optimizer:
    def __init__(self, **kwargs):
        pass

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    @argValidator
    def optimizer(self, value: torch.optim.Optimizer):
        self._optimizer = value
