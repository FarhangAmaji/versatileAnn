import pytorch_lightning as pl

from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import _allowOnlyCreationOf_ChildrenInstances
from versatileAnn.newModule.fit import _NewWrapper_modelFitter
from versatileAnn.newModule.loss import _NewWrapper_lossNRegularization
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
                 _NewWrapper_lossNRegularization, _NewWrapper_optimizer,
                 _NewWrapper_modelFitter, _NewWrapper_preRunTests,
                 _NewWrapper_saveLoad, _NewWrapper_modelDifferentiator):

    @argValidator
    def __init__(self, **kwargs):
        self.printTestPrints('NewWrapper init')
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, NewWrapper)


    def forward(self, inputs, targets):
        # force reimplementing this method
        raise NotImplementedError

    def _tempCommonStep(self, batch, phase):
        # reset tempVarStep
        self.resetTempVar_step(phase)

        inputs, targets = batch
        # goodToHave1
        #  later make it compatible with outputMask
        # bugPotentialCheck2
        #  also what if the batch has 1 items; may don't allow this one as almost everything depends on targets
        # goodToHave1
        #  must think about this more on how to match batchOutputs and self.forward args can get
        #  matched and related values of batchOutputs get sent to self.forward
        #  - may add targets if its is model arguments
        forwardOutputs = self(inputs, targets)

        # calculate loss
        # bugPotentialCheck1
        #  if the loss is not returned from _calculatedLosses because of
        #  not having self.lossFuncs would it make error
        loss = None  # kkk maybe this is related to allowing self.lossFuncs to be empty
        loss, calculatedLosses = self._calculateLosses(loss, forwardOutputs, targets)

        # Log losses
        self._logLosses(calculatedLosses, phase)
        return loss

    def training_step(self, batch, batch_idx):
        phase = self.phases.train
        return self._tempCommonStep(batch, phase)

    def validation_step(self, batch, batch_idx):
        phase = self.phases.val
        return self._tempCommonStep(batch, phase)

    def configure_optimizers(self):
        return self.optimizer

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
