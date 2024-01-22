from abc import ABC

import pytorch_lightning as pl
import torch
import torch.nn as nn

from utils.initParentClasses import initParentClasses
from utils.typeCheck import argValidator
from versatileAnn.newModule.loss import _NewWrapper_loss
from versatileAnn.newModule.modelDifferentiator import _NewWrapper_modelDifferentiator
from versatileAnn.newModule.optimizer import _NewWrapper_optimizer
from versatileAnn.newModule.preInitNPostInit_nModelReset import \
    _NewWrapper_preInitNPostInit_nModelReset
from versatileAnn.newModule.preRunTests import _NewWrapper_preRunTests
from versatileAnn.newModule.properties import _NewWrapper_properties
from versatileAnn.newModule.saveLoad import _NewWrapper_saveLoad
from versatileAnn.newModule.temVars import _NewWrapper_tempVars


# kkk2 think about seed later
# kkk2 parent classes must not have instance
# kkk1 if I use kwargsBasedOnMethod then I should check conflicts when 2 methods get some args with same name


class NewWrapper(pl.LightningModule, _NewWrapper_properties,
                 _NewWrapper_tempVars, _NewWrapper_preInitNPostInit_nModelReset,
                 _NewWrapper_loss, _NewWrapper_optimizer,
                 _NewWrapper_preRunTests, _NewWrapper_saveLoad,
                 _NewWrapper_modelDifferentiator):

    @argValidator
    def __init__(self, **kwargs):
        # cccDevStruct
        #  note __init__ and __new__ don't take optional args
        self.dummy = nn.Linear(7, 1) # kkk this is dummy

        # _NewWrapper_optimizer must be initialized at the postInitialization
        initParentClasses(type(self), kwargs, self, just=['_NewWrapper_optimizer'])
        # kkk1 iss1 later check how the user should use this class; note dummyLayer is added in order not
        #  to get error of optimizer; note specially inheriting from classes I want

    def forward(self, inputs, targets):
        output = {}
        output['volume'] = self.dummy(
            targets['volume'])  # kkk1 this just to get some output; related to iss1
        return output

    def _tempCommonStep(self, batch, phase):
        # reset tempVarStep
        self.resetTempVar_step(phase)

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

    # reset tempVar of phases on epoch start
    def on_train_epoch_start(self):
        self.resetTempVar_epoch(self.phases.train)

    def on_validation_epoch_start(self):
        self.resetTempVar_epoch(self.phases.val)

    def on_test_epoch_start(self):
        self.resetTempVar_epoch(self.phases.test)

    def on_predict_epoch_start(self):
        self.resetTempVar_epoch(self.phases.predict)

    # reset tempVar of all phases on run start
    def on_fit_start(self):
        for phase in list(self.phases.keys()):
            self.resetTempVar_run(phase)

        self.resetTempVarRun_allPhases()
