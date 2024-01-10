import pytorch_lightning as pl
import torch
import torch.nn as nn

from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import getMethodArgs
from versatileAnn.newModule.newWrapper_loss import _NewWrapper_loss
from versatileAnn.newModule.newWrapper_modelDifferentiator import _NewWrapper_modelDifferentiator
from versatileAnn.newModule.newWrapper_optimizer import _NewWrapper_optimizer
from versatileAnn.newModule.newWrapper_preRunTests import _NewWrapper_preRunTests
from versatileAnn.newModule.newWrapper_saveLoad import _NewWrapper_saveLoad


# kkk2 think about seed later
# kkk2 parent classes must not have instance
# kkk1 if I use kwargsBasedOnMethod then I should check conflicts when 2 methods get some args with same name

class _NewWrapper_properties:
    @argValidator
    def __init__(self, modelName: str = '', devMode: bool = True, lr=3e-4):
        self.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.losses = []
        self._setModelName(modelName)
        self.devMode = devMode  # kkk2 do I need it? if I detected has not run pretests then run them and dont need devMode
        self.lr = lr  # kkk2 make property

        if devMode:
            pass  # kkk?
        else:
            pass  # kkk?

    def _setModelName(self, modelName):
        if not modelName:
            if self.__class__.__name__ == 'NewWrapper':
                raise ValueError('modelName must be provided if not inherited form NewWrapper')
            self.modelName = self.__class__.__name__
        else:
            self.modelName = modelName

    @property
    def devMode(self):
        return self._devMode

    @devMode.setter
    @argValidator
    def devMode(self, value: bool):
        self._devMode = value


# kkk2 rest model: after doing getInitArgs and postInitCaller
"""
class A:
    def __init__(self, a):
        self.a = a
        args=inspect.signature(self.__init__).parameters.keys()
        frame = inspect.currentframe()
        _, _, _, values = inspect.getargvalues(frame)
        self._initArgs = {arg: values[arg] for arg in args}
    def restModel(self):
        self.a=5
        print(self.a)
        self=type(self)(**self._initArgs)
        print(self.a)
        return self
a=A(1)
a.restModel()
"""


class NewWrapper(pl.LightningModule, _NewWrapper_properties, _NewWrapper_loss,
                 _NewWrapper_optimizer, _NewWrapper_preRunTests, _NewWrapper_saveLoad,
                 _NewWrapper_modelDifferentiator):
    def __new__(cls, *args, **kwargs):  # kkk1 think about arg sharing with init
        obj = super().__new__(cls)
        # parent classes init are here
        pl.LightningModule.__init__(obj)
        _NewWrapper_properties.__init__(obj)
        _NewWrapper_loss.__init__(obj)
        _NewWrapper_optimizer.__init__(obj)
        _NewWrapper_preRunTests.__init__(obj)
        _NewWrapper_saveLoad.__init__(obj)
        _NewWrapper_modelDifferentiator.__init__(obj)
        return obj

    @argValidator
    def __init__(self, modelName: str = '', devMode: bool = True):
        # pl.LightningModule.__init__(self)
        # _NewWrapper_properties.__init__(self, modelName, devMode)
        self.dummy = nn.Linear(7, 1)
        # kkk1 iss1 later check how the user should use this class; note dummyLayer is added in order not
        #  to get error of optimizer; note specially inheriting from classes I want

    def forward(self, inputs, targets):
        output = {}
        output['volume'] = self.dummy(
            targets['volume'])  # kkk1 this just to get some output; related to iss1
        return output

    def _tempCommonStep(self, batch, stepPhase):
        # kkk2 with some epochData or stepData which get rest I may add flexibility to pipeline
        inputs, targets = batch
        # kkk1 later make it compatible with outputMask
        outputs = self(inputs, targets)  # kkk2 may add targets if its is model arguments
        # calculate loss
        loss = None
        calculatedLosses, loss = self._calculateLosses(loss, outputs, targets)
        # Log losses
        self._logLosses(calculatedLosses, stepPhase)
        return loss

    def training_step(self, batch, batch_idx):
        stepPhase = 'train'
        return self._tempCommonStep(batch, stepPhase)

    def validation_step(self, batch, batch_idx):
        stepPhase = 'val'
        return self._tempCommonStep(batch, stepPhase)
