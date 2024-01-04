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
        # kkk later check how the user should use this class; note dummyLayer is added in order not
        #  to get error of optimizer; note specially inheriting from classes I want

    def forward(self, inputs, targets):
        output = {}
        output['volume'] = self.dummy(targets['volume'])  # kkk this just to get some output
        return output

    def _calculateLosses(self, loss, outputs, targets):
        calculatedLosses = []
        outputsFlatData = _NestedDictStruct(outputs,
                                            giveFilledStruct=True).toList()  # kkk add to emptyStruct do it one time
        targetsFlatData = _NestedDictStruct(targets, giveFilledStruct=True).toList()
        for i, loss_ in enumerate(self.losses):
            # kkk only for first one should make backwards and for other should do
            assert len(outputsFlatData) == len(
                targetsFlatData), 'mismatch in lens of outputsFlatData and targetsFlatData'
            lossRes = torch.Tensor([0]).to(outputsFlatData[0].device)
            for j in range(len(outputsFlatData)):
                lossRes += loss_(outputsFlatData[j], targetsFlatData[j])

            calculatedLosses.append(lossRes)

            if i == 0:  # only first loss is returned, and therefore done backprop on
                loss = calculatedLosses[0]
        return calculatedLosses, loss

    def _logLosses(self, calculatedLosses, stepPhase):
        for i, loss_ in enumerate(self.losses):  # kkk where does it save the log
            self.log(snakeToCamel(stepPhase + type(loss_).__name__), calculatedLosses[i],
                     on_epoch=False, prog_bar=True)

    def training_step(self, batch, batch_idx):
        stepPhase = 'train'
        inputs, targets = batch
        # kkk later make it compatible with outputMask
        outputs = self(inputs, targets)  # kkk may add targets if its is model arguments

        # calculate loss
        loss = None
        calculatedLosses, loss = self._calculateLosses(loss, outputs, targets)
        # Log losses
        self._logLosses(calculatedLosses, stepPhase)
        return loss

    def configure_optimizers(self):  # kkk change it later
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    @argValidator
    def trainModel(self, trainDataloader, *, losses: List[nn.modules.loss._Loss], maxEpochs=5,
                   savePath, tensorboardPath='', valDataloader=None, externalKwargs=None):
        # cccUsage
        #  only first loss is used for backpropagation and others are just for logging
        if losses:
            # cccUsage
            #  in the case outside of trainModel losses is been set, so if not passed would use them
            self.losses = losses

        if self.devMode:
            trainer = pl.Trainer(
                # kkk add camelCase and snakeCase compatible options of this to passed
                fast_dev_run=True,  # Run only for a small number of epochs for faster development
                log_every_n_steps=1,  # Log every step
                logger=pl.loggers.TensorBoardLogger("Kog2s", name=self.modelName),
                # kkk where does it; SavePath
            )
        else:
            pass  # kkk

        if valDataloader:  # kkk add camelCase and snakeCase compatible options of this to passed
            trainer.fit(self, trainDataloader, valDataloader)
        else:
            trainer.fit(self, trainDataloader)
