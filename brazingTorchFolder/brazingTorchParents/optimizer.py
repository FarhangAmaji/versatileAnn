import copy
from typing import Optional

import torch
from torch.optim.lr_scheduler import ChainedScheduler

from brazingTorchFolder.utilsFolder.utils import isPytorchLightningScheduler
from projectUtils.misc import _allowOnlyCreationOf_ChildrenInstances
from projectUtils.typeCheck import argValidator
from projectUtils.warnings import Warn


class _BrazingTorch_optimizer:
    def __init__(self, optimizer: Optional[torch.optim.Optimizer] = None,
                 schedulers: Optional[list] = None,
                 lr: Optional[float] = None, **kwargs):

        if optimizer:
            self.optimizer = optimizer
            if lr:
                raise ValueError(
                    "you have passed optimizer and lr together. just pass the optimizer")

        if schedulers:
            self.schedulers = schedulers
        else:
            self.schedulers = []

        if lr is not None:
            self.lr = lr

        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _BrazingTorch_optimizer)

    @property
    def optimizer(self):
        # if the optimizer is not set we set a default one
        if not hasattr(self, '_optimizer'):
            if hasattr(self, 'parameters'):  # to prevent error when parameters are not set
                lr = 3e-4
                self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

                infoMsg = f'no optimizer was set, a default Adam optimizer with lr={lr} was set'
                # note we check some tests with the things printed but Warn.info is not on prints
                # so we do self._printTestPrints(infoMsg)
                self._printTestPrints(infoMsg)
                Warn.info(infoMsg)

        return self._optimizer

    @optimizer.setter
    @argValidator
    def optimizer(self, value: torch.optim.Optimizer):
        # mustHave3
        #  should have 2 optimizers
        # kkk
        #  may we assert error when weight decay is True or have a number(don't know yet)
        #  but anyway either force users or don't apply weightDeccay here and apply weightDecay
        #  with our implementation of Regularizations
        # ccc1
        #  this part is designed in order to be able to resetOptimizer but the args passed when
        #  setting it
        # ccc4
        #  minor: value which is a torch.optim.Optimizer is not subscriptable so we do vars(value)
        optimizerInitArgs_names = list(vars(value)['param_groups'][0].keys())
        # ccc3
        #  params are dynamic and must not be saved
        if 'params' in optimizerInitArgs_names:
            optimizerInitArgs_names.remove('params')

        # it's not allowed to have weight_decay in optimizer and generalRegularization(ofc not None
        # version) together
        # addTest2
        if self._generalRegularization.type != 'None':
            if 'weight_decay' in optimizerInitArgs_names:
                if value.param_groups[0]['weight_decay'] != 0:
                    Warn.warn(
                        f'the model has generalRegularization of {self._generalRegularization}' + \
                        "so can't set weight_decay for optimizer and weight_decay will be set 0.")

        self._optimizerInitArgs = {'type': type(value),
                                   'args': {par: copy.deepcopy(vars(value)['param_groups'][0][par])
                                            for par in optimizerInitArgs_names}}

        self._optimizer = value

        # make sure self.lr is in sync with optimizer lr
        self.lr = self._optimizerInitArgs['args']['lr']

        # goodToHave3
        #  as I think the `value['param_groups'][0].keys()` are same as `getMethodArgs(type(value))`
        #  but later can add logging so if its not True it would be logged and more precaution
        #  applied in next version

    def resetOptimizer(self, keepLr=True):
        # cccUsage
        #  this is inplace and no need to set, but also compatible with setting
        # ccc1
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

    # schedulers
    @property
    def schedulers(self):
        if len(self._schedulers) > 1:
            # for multiple schedulers we use ChainedScheduler
            return ChainedScheduler(self._schedulers)
        elif len(self._schedulers) == 1:
            return self._schedulers[0]
        return self._schedulers

    @schedulers.setter
    @argValidator
    def schedulers(self, value: list):
        if not value:
            self._schedulers = value
            return

        if not hasattr(self, 'optimizer'):
            raise ValueError("you must set optimizer before setting schedulers")

        for scheduler in value:
            if not isPytorchLightningScheduler(scheduler):
                raise ValueError("the scheduler must be a PyTorch Lightning scheduler")
        self._schedulers = value
