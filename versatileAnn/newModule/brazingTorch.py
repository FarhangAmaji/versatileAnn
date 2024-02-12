import pytorch_lightning as pl

from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import _allowOnlyCreationOf_ChildrenInstances
from versatileAnn.newModule.lossModule import _BrazingTorch_loss
from versatileAnn.newModule.modelDifferentiator import _BrazingTorch_modelDifferentiator
from versatileAnn.newModule.modelFitter import _BrazingTorch_modelFitter
from versatileAnn.newModule.optimizer import _BrazingTorch_optimizer
from versatileAnn.newModule.preInitNPostInit_nModelReset import \
    _BrazingTorch_preInitNPostInit_nModelReset
from versatileAnn.newModule.preRunTests import _BrazingTorch_preRunTests
from versatileAnn.newModule.properties import _BrazingTorch_properties
from versatileAnn.newModule.regularization import _BrazingTorch_regularization
from versatileAnn.newModule.saveLoad import _BrazingTorch_saveLoad
from versatileAnn.newModule.specialModes import _BrazingTorch_specialModes
from versatileAnn.newModule.temVars import _BrazingTorch_tempVars


# kkk2 think about seed later
# kkk2 parent classes must not have instance
# kkk1 if I use kwargsBasedOnMethod then I should check conflicts when 2 methods get some args with same name


class BrazingTorch(pl.LightningModule,
                 _BrazingTorch_properties, _BrazingTorch_tempVars,
                 _BrazingTorch_preInitNPostInit_nModelReset, _BrazingTorch_optimizer,
                 _BrazingTorch_loss, _BrazingTorch_regularization,
                 _BrazingTorch_modelFitter, _BrazingTorch_preRunTests,
                 _BrazingTorch_saveLoad, _BrazingTorch_modelDifferentiator,
                 _BrazingTorch_specialModes):

    __version__ = '0.2'
    @argValidator
    def __init__(self, **kwargs):
        # kkk
        #  this init should take all other args which it parent classes take because the user can
        #  really check all parent classes to see what functionalities does class offer
        self.printTestPrints('BrazingTorch init')
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, BrazingTorch)

    def forward(self, inputs, targets):
        # cccUsage
        #  if you want to use VAEMode must return normalForwardOutputs, mean, logvar

        # force reimplementing this method
        raise NotImplementedError

    def commonStep(self, batch, phase):
        # bugPotentialCheck1
        #  note this method should always be similar to specialModesStep
        #  so check specialModesStep and make similar changes here specially for the comments
        # cccDevStruct
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
        # bugPotentialCheck2
        #  also what if the batch has 1 items; may don't allow this one as almost everything depends on targets
        # goodToHave1
        #  must think about this more on how to match batchOutputs and self.forward args can get
        #  matched and related values of batchOutputs get sent to self.forward
        #  - may add targets if its is model arguments
        forwardOutputs = self.forward(inputs, targets)

        # calculate loss
        # bugPotentialCheck1
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

    def configure_optimizers(self):
        # cccDevStruct
        #  pytorch lightning expects this method to be here
        return self.optimizer

    # reset tempVar of phases on epoch start
    def on_train_epoch_start(self):
        # cccDevStruct
        #  pytorch lightning expects this method to be here
        self.resetTempVar_epoch(self.phases.train)

    def on_validation_epoch_start(self):
        # cccDevStruct
        #  pytorch lightning expects this method to be here
        self.resetTempVar_epoch(self.phases.val)

    def on_test_epoch_start(self):
        # cccDevStruct
        #  pytorch lightning expects this method to be here
        self.resetTempVar_epoch(self.phases.test)

    def on_predict_epoch_start(self):
        # cccDevStruct
        #  pytorch lightning expects this method to be here
        self.resetTempVar_epoch(self.phases.predict)

    # reset tempVar of all phases on run start
    def on_fit_start(self):
        # cccDevStruct
        #  pytorch lightning expects this method to be here
        for phase in list(self.phases.keys()):
            self.resetTempVar_run(phase)

        self.resetTempVarRun_allPhases()
        self._tempVarRun_allPhases_hidden = {}

    def _isCls_BrazingTorchClass(self, cls_):
        # cccDevAlgo
        #  this is a util to be used in parent classes and not get circular import error
        return cls_ is BrazingTorch

    @staticmethod
    def _getBrazingTorch_classObject():
        return BrazingTorch
