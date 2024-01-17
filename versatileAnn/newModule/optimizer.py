import copy

import torch

from utils.typeCheck import argValidator
from utils.warnings import Warn


class _NewWrapper_optimizer:
    def __init__(self, **kwargs):
        pass

    @property
    def optimizer(self):
        # if the optimizer is not set we set a default one
        if not hasattr(self, '_optimizer'):
            lr = 3e-4
            self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            self.lr = lr
            Warn.info(f'no optimizer was set, a default Adam optimizer with lr={lr} was set')

        return self._optimizer

    @optimizer.setter
    @argValidator
    def optimizer(self, value: torch.optim.Optimizer):
        # kkk
        #  may we assert error when weight decay is True or have a number(don't know yet)
        #  but anyway either force users or don't apply weightDeccay here and apply weightDecay
        #  with our implementation of Regularizations
        # cccDevStruct
        #  this part is designed in order to be able to resetOptimizer but the args passed when
        #  setting it
        optimizerInitArgs_names = list(value['param_groups'][0].keys())
        if 'params' in optimizerInitArgs_names:
            optimizerInitArgs_names.remove('params')

        self._optimizerInitArgs = {'type': type(value),
                                   'args': {par: copy.deepcopy(value['param_groups'][0][par]) for
                                            par in
                                            optimizerInitArgs_names}}

        self._optimizer = value

        # goodToHave3
        #  as I think the `value['param_groups'][0].keys()` are same as `getMethodArgs(type(value))`
        #  but later can add logging so if its not True it would be logged and more precaution
        #  applied in next version


    def resetOptimizer(self, keepLr=True):
        # cccUsage
        #  this is inplace and no need to set, but also compatible with setting
        # cccDevAlgo
        #  this is designed in order to be sure that past accumulated
        #  params like momentum have got reset
        argsToReinit = self._optimizerInitArgs['args'].copy()
        if keepLr:
            argsToReinit['lr'] = self.optimizer.param_groups[0]['lr']

        optimizerClass = self._optimizerInitArgs['type']
        self.optimizer = optimizerClass(self.parameters(), **argsToReinit)

        return self.optimizer  # this for the case that user does'nt know it's inplace


    def changeLearningRate(self, newLr):
        self._lr = newLr
        if not hasattr(self, 'optimizer'):  # prevent error if the optimizer is not set yet
            return
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._lr


    @property
    def lr(self):
        return self._lr


    @lr.setter
    @argValidator
    def lr(self, newLr: float):
        self.changeLearningRate(newLr)


    @argValidator
    def multiplyLr(self, factor: float):
        self.lr = self.lr * factor


    @argValidator
    def divideLr(self, factor: float):
        self.lr = self.lr / factor
