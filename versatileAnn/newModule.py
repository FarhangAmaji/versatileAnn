from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn

from dataPrep.dataloader import _NestedDictStruct
from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import snakeToCamel


class NewWrapper_properties:
    @argValidator
    def __init__(self, modelName: str = '', devMode: bool = True):
        self.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.losses = []
        self._setModelName(modelName)
        self.devMode = devMode

        if devMode:
            pass  # kkk
        else:
            pass  # kkk

    def _setModelName(self, modelName):
        if not modelName:
            if self.__class__.__name__ == 'NewWrapper':
                raise ValueError('modelName must be provided if not inherited form NewWrapper')
            self.modelName = self.__class__.__name__
        else:
            self.modelName = modelName

    @property
    def losses(self):
        return self._losses

    @losses.setter
    @argValidator
    def losses(self, value: List[nn.modules.loss._Loss]):
        self._losses = value

    @property
    def devMode(self):
        return self._devMode

    @devMode.setter
    @argValidator
    def devMode(self, value: bool):
        self._devMode = value


class NewWrapper(pl.LightningModule, NewWrapper_properties):
    @argValidator
    def __init__(self, modelName: str = '', devMode: bool = True):
        pl.LightningModule.__init__(self)
        NewWrapper_properties.__init__(self, modelName, devMode)
        self.dummy = nn.Linear(7, 1)
        # kkk later check how the user should use this class; note dummyLayer is added in order not to get
        # error of optimizer
    def forward(self, inputs, targets):
        output = {}
        return output

    def training_step(self, batch, batch_idx):
        loss = None
        return loss

    def configure_optimizers(self):  # kkk change it later
        return torch.optim.Adam(self.parameters(), lr=1e-3)
