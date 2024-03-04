from typing import List, Union, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from brazingTorchFolder.brazingTorchParents.baseAttributes import _BrazingTorch_baseAttributes
from brazingTorchFolder.brazingTorchParents.lossModule import _BrazingTorch_loss
from brazingTorchFolder.brazingTorchParents.modelDifferentiator import \
    _BrazingTorch_modelDifferentiator
from brazingTorchFolder.brazingTorchParents.modelFitter import _BrazingTorch_modelFitter
from brazingTorchFolder.brazingTorchParents.optimizer import _BrazingTorch_optimizer
from brazingTorchFolder.brazingTorchParents.preInitNPostInit_nModelReset import \
    _BrazingTorch_preInitNPostInit_nModelReset
from brazingTorchFolder.brazingTorchParents.preRunTests import _BrazingTorch_preRunTests
from brazingTorchFolder.brazingTorchParents.regularization import _BrazingTorch_regularization
from brazingTorchFolder.brazingTorchParents.saveLoad import _BrazingTorch_saveLoad
from brazingTorchFolder.brazingTorchParents.specialModes import _BrazingTorch_specialModes
from brazingTorchFolder.brazingTorchParents.temVars import _BrazingTorch_tempVars
from brazingTorchFolder.lossRegulator import LossRegulator
from projectUtils.misc import _allowOnlyCreationOf_ChildrenInstances
from projectUtils.typeCheck import argValidator


# kkk2 think about seed later
# kkk2 parent classes must not have instance
# kkk1 if I use kwargsBasedOnMethod then I should check conflicts when 2 methods get some args with same name


class BrazingTorch(pl.LightningModule,
                   _BrazingTorch_baseAttributes, _BrazingTorch_tempVars,
                   _BrazingTorch_preInitNPostInit_nModelReset, _BrazingTorch_optimizer,
                   _BrazingTorch_loss, _BrazingTorch_regularization,
                   _BrazingTorch_modelFitter, _BrazingTorch_preRunTests,
                   _BrazingTorch_saveLoad, _BrazingTorch_modelDifferentiator,
                   _BrazingTorch_specialModes):
    __version__ = '0.2.0'

    @argValidator
    def __init__(self, modelName: str = '',
                 noAdditionalOptions: bool = False,
                 generalRegularization: Optional[Union[LossRegulator, dict]] = None,
                 lossFuncs: Optional[List[nn.modules.loss._Loss]] = None,
                 keepLr_notReplaceWithBestLr: Optional[bool] = False,
                 dropoutEnsembleMode: bool = False, VAEMode: bool = False,
                 dropoutEnsemble_samplesNum=100,
                 getAllNeededDefinitions=True,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 schedulers: Optional[list] = None, lr: Optional[float] = None,
                 keepBatchSize_notReplaceWithBestBatchSize: Optional[bool] = False,
                 testPrints=False, **kwargs):

        # ccc1
        #  to know what these args are exactly are go to the original parent
        #  class where they are defined:
        #               _BrazingTorch_baseAttributes: modelName, noAdditionalOptions, testPrints
        #               _BrazingTorch_modelDifferentiator: getAllNeededDefinitions
        #               _BrazingTorch_loss: lossFuncs
        #               _BrazingTorch_optimizer: optimizer, schedulers, lr
        #               _BrazingTorch_regularization: generalRegularization
        #               _BrazingTorch_specialModes: dropoutEnsembleMode, VAEMode,
        #                                   dropoutEnsemble_samplesNum
        #               _BrazingTorch_preRunTests: keepLr_notReplaceWithBestLr,
        #                           keepBatchSize_notReplaceWithBestBatchSize

        # ccc1
        #  in order to separate concerns and duties also to make the code more readable
        #  components of BrazingTorch are separated into different parent classes
        #  as u see, almost all of args are not set here; these are the args used here, so the user
        #  can really check all parent classes to see what functionalities does class offer
        # bugPotn1
        #  does putting all args of parent classes here, make problems with __new__ specially
        #  _get_parentClassesOfBrazingTorch in _BrazingTorch_preInitNPostInit_nModelReset
        #  - note # LBTEam1 part is supposed to give error if there are same args in parent classes

        self.printTestPrints('BrazingTorch init')
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, BrazingTorch)

    def forward(self, inputs, targets):
        # cccUsage
        #  if you want to use VAEMode must return normalForwardOutputs, mean, logvar

        # force reimplementing this method
        raise NotImplementedError

    def commonStep(self, batch, phase):
        # bugPotn1
        #  note this method should always be similar to specialModesStep
        #  so check specialModesStep and make similar changes here specially for the comments
        # ccc1
        #  we don't make a baseFunc as the users would get idea of these 2 funcs separately
        # cccUsage
        #  note we may need to reimplement this method
        #  so take a look at this method to get an idea how to reimplement it yourself
        #  - also if you are using variationalAutoEncoderMode or dropoutEnsembleMode you may
        #       want to reimplement the specialModesStep in specialModes.py
        if self.VAEMode or self.dropoutEnsembleMode:
            return self.specialModesStep(batch, phase)

        # reset tempVarStep
        self.resetTempVar_step(phase)

        inputs, targets = batch
        # goodToHave1
        #  later make it compatible with outputMask; also do the change on specialModesStep
        # bugPotn2
        #  also what if the batch has 1 items; may don't allow this one as almost everything depends on targets
        # goodToHave1
        #  must think about this more on how to match batchOutputs and self.forward args can get
        #  matched and related values of batchOutputs get sent to self.forward
        #  - may add targets if its is model arguments
        forwardOutputs = self.forward(inputs, targets)

        # calculate loss
        # bugPotn1
        #  if the loss is not returned from _calculatedLosses because of
        #  not having self.lossFuncs would it make error
        loss, calculatedLosses = self._calculateLosses(forwardOutputs, targets)

        # Log losses
        self._logLosses(calculatedLosses, phase)
        # cccUsage
        #  Please ensure that your `training_step` method in PyTorch Lightning
        #  returns either the loss value directly or a dictionary containing
        #  the loss value under the key "loss". This is essential for the
        #  training process specially back propagation to function correctly.
        return loss

    def training_step(self, batch, batch_idx):
        phase = self.phases.train
        return self.commonStep(batch, phase)

    def validation_step(self, batch, batch_idx):
        phase = self.phases.val
        return self.commonStep(batch, phase)

    def test_step(self, batch, batch_idx):
        phase = self.phases.test
        return self.commonStep(batch, phase)

    def predict_step(self, batch, batch_idx):
        phase = self.phases.predict
        return self.commonStep(batch, phase)

    # ----
    def configure_optimizers(self):
        # ccc1
        #  pytorch lightning expects this method to be here
        if self._schedulers:
            return (self.optimizer, self.schedulers)# addTest2
        return self.optimizer

    # ---- reset tempVar of phases on epoch start
    def on_train_epoch_start(self):
        # ccc1
        #  pytorch lightning expects this method to be here
        self.resetTempVar_epoch(self.phases.train)

    def on_validation_epoch_start(self):
        # ccc1
        #  pytorch lightning expects this method to be here
        self.resetTempVar_epoch(self.phases.val)

    def on_test_epoch_start(self):
        # ccc1
        #  pytorch lightning expects this method to be here
        self.resetTempVar_epoch(self.phases.test)

    def on_predict_epoch_start(self):
        # ccc1
        #  pytorch lightning expects this method to be here
        self.resetTempVar_epoch(self.phases.predict)

    # reset tempVar of all phases on run start
    def on_fit_start(self):
        # ccc1
        #  pytorch lightning expects this method to be here
        for phase in list(self.phases.keys()):
            self.resetTempVar_run(phase)

        self.resetTempVarRun_allPhases()
        self._tempVarRun_allPhases_hidden = {}

    # ----
    @argValidator
    def on_save_checkpoint(self, checkpoint: dict):
        # ccc1
        #  pytorch lightning expects this method to be here
        return self.onSaveCheckpoint(checkpoint)

    @argValidator
    def on_load_checkpoint(self, checkpoint: dict):  # kkk
        # ccc1
        #  pytorch lightning expects this method to be here
        return self.onLoadCheckpoint(checkpoint)

    # ----
    def _isCls_BrazingTorchClass(self, cls_):
        # ccc1
        #  this is a util to be used in parent classes and not get circular import error
        return cls_ is BrazingTorch

    @staticmethod
    def _getBrazingTorch_classObject():
        return BrazingTorch
