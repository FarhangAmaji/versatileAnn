import os
from inspect import getmro
from inspect import isclass
from typing import List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import Logger
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler
from torch.utils.data import DataLoader

from brazingTorchFolder.utilsFolder.callbacks import StoreEpochData, WarmUpScheduler, \
    SchedulerChanger
from projectUtils.dataTypeUtils.tensor import getTorchDevice
from projectUtils.typeCheck import argValidator
from projectUtils.warnings import Warn


def loadFromCheckpointPath(checkpointPath, modelClassOrInstance):
    # bugPotn1
    #  note a normal checkpoint dictionary has these keys ['epoch', 'global_step',
    #  'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks',
    #  'optimizer_states', 'lr_schedulers']
    #  - 'optimizer_states' and 'lr_schedulers' are used below and 'pytorch-lightning_version'
    #  is not gonna be used but I have doubts about the rest of the keys

    if not os.path.exists(checkpointPath):
        raise ValueError(f"{checkpointPath=} doesn't exist")

    # Load checkpoint
    checkpoint = torch.load(checkpointPath)

    # Load state_dict into model
    if isclass(modelClassOrInstance):
        modelClass = modelClassOrInstance
    else:
        modelClass = type(modelClassOrInstance)
    model = modelClass.load_from_checkpoint(checkpoint_path=checkpointPath)

    # Load optimizer and schedulers states
    if 'optimizer_states' in checkpoint:
        model.optimizer.load_state_dict(checkpoint['optimizer_states'])
    if 'lr_schedulers' in checkpoint:
        model.schedulers = checkpoint['lr_schedulers']

    return model


def isPytorchLightningScheduler(obj):
    """
    Robustly checks if an object can be used as a PyTorch Lightning scheduler.

    This function combines the accuracy and flexibility of both Response A and Response B,
    adding error handling and a more comprehensive check for compatibility.

    Args:
        obj: The object to be checked.

    Returns:
        True if the object is a PyTorch Lightning scheduler, False otherwise.
    """
    if isinstance(obj, (LRScheduler._LRScheduler, LRScheduler.LambdaLR,
                        LRScheduler.OneCycleLR, LearningRateMonitor)):
        # Direct instance of PyTorch Lightning schedulers (preferred)
        return True
    elif isinstance(obj, LightningModule):
        # Check if LightningModule has a compatible custom schedulers implementation
        schedulerMethods = [
            methodName for methodName in dir(obj)
            if methodName in {'onTrainEpochStart', 'onTrainBatchEnd', 'onValidationEpochStart'}
        ]
        customSchedulerMethodFound = len(schedulerMethods) > 0
        lightningModuleHasScheduler = hasattr(obj, 'lrScheduler') or \
                                      'optimizer.lrScheduler' in obj.__dict__
        return customSchedulerMethodFound and lightningModuleHasScheduler
    else:
        # Check if obj is a subclass of a supported base class
        try:
            # Try using getmro() for more reliable inheritance path checks
            for baseClass in getmro(obj):
                if baseClass in (LRScheduler._LRScheduler, LRScheduler.LambdaLR,
                                 LRScheduler.OneCycleLR, LightningModule):
                    return True
        except TypeError:  # Handle cases where getmro() might not be supported
            pass
        return False


@argValidator
def externalFit(self, trainDataloader: DataLoader,
                valDataloader: Optional[DataLoader] = None,
                *, lossFuncs: Optional[List[nn.modules.loss._Loss]] = None,
                seed=None, resume=True, seedSensitive=False,
                addDefaultLogger=True, addDefault_gradientClipping=True,
                warmUp_epochNum=5, addDefault_reduceLROnPlateau=True,
                addDefault_earlyStopping=True,
                preRunTests_force=False, preRunTests_seedSensitive=False,
                preRunTests_lrsToFindBest=None,
                preRunTests_batchSizesToFindBest=None,
                preRunTests_fastDevRunKwargs=None, preRunTests_overfitBatchesKwargs=None,
                preRunTests_profilerKwargs=None, preRunTests_findBestLearningRateKwargs=None,
                preRunTests_findBestBatchSizesKwargs=None,
                **kwargs):
    # ccc1
    #  note this is implementation for .fit method of BrazingTorch class, but implemented here
    #  you may read why it's been implemented here in the .fit method itself

    if not seed:
        seed = self.seed

    lossFuncs = self._lossFuncsNotPassedHere_errorOrUseModels(lossFuncs)
    self._setLossFuncs_ifNot(lossFuncs)

    architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged = self._determineFitRunState(
        seed=seed,
        resume=resume,
        seedSensitive=seedSensitive)

    if fitRunState == "don't run":
        Warn.info("as you decided, the model is not replaced, if you want to resume, pass " +
                  "resume=True to .fit")
        return self, None
    elif fitRunState == "resume":
        answer = self._warnIf_modelIsChanged()
        if answer and answer.lower() == 'yes':
            return self, None

        self = loadFromCheckpointPath(checkpointPath, self)
        self.to(getTorchDevice().type)
        # bugPotn1
        #  why the pycharm coloring, still shows `self` as it is the `self from input arguments`

    self._saveArchitectureDict(loggerPath)

    # add default logger if allowed and no logger is passed
    # because by default we are logging some metrics
    if addDefaultLogger and 'logger' not in kwargs:
        architectureNameNVersion = self.getArchitectureNameNVersion_fromLoggerPath(loggerPath)
        kwargs['logger'] = pl.loggers.TensorBoardLogger(architectureNameNVersion['modelName'],
                                                        name=architectureNameNVersion[
                                                            'architectureName'],
                                                        version=architectureNameNVersion['version'])
        # ccc2
        #  - note the addDefaultLogger is added here also in baseFit
        #  note in the baseFit we don't have access to the architectureName and
        #  the version but here we have
        #  - also note obviously we don't expect addDefaultLogger in baseFit replace
        #  this one as now the logger is filled and addDefaultLogger only adds
        #  when the logger is empty
        #  - addDefaultLogger adds logger to kwargs['trainer'] and not kwargs or kwargs['log']

    # run preRunTests which provide useful sanity checks like fastDevRun, overfitBatches
    # also details like profiler to see bottlenecks of the model, bestLearningRate(sets it by
    # default), bestBatchSize(sets it by default)
    self.preRunTests(trainDataloader=trainDataloader, valDataloader=valDataloader,
                     lossFuncs=lossFuncs,
                     force=preRunTests_force, seedSensitive=preRunTests_seedSensitive,
                     lrsToFindBest=preRunTests_lrsToFindBest,
                     batchSizesToFindBest=preRunTests_batchSizesToFindBest,
                     fastDevRunKwargs=preRunTests_fastDevRunKwargs,
                     overfitBatchesKwargs=preRunTests_overfitBatchesKwargs,
                     profilerKwargs=preRunTests_profilerKwargs,
                     findBestLearningRateKwargs=preRunTests_findBestLearningRateKwargs,
                     findBestBatchSizesKwargs=preRunTests_findBestBatchSizesKwargs, **kwargs)

    # we didn't return runName from _determineFitRunState as a separate variable,
    # but here we are getting its value
    runName = loggerPath.split(os.sep)[-1]

    # goodToHave3
    #  we have some preset kwargsApplied. but I want to let user either use these defaults or
    #  not
    #  - note the user may want to keep/remove them
    #  - note the user won't be able to add to defaults; note as if user wants any feature
    #  he/she can easily add them with **kwargs of .fit method
    #  - so ._fit_defaults should be encapsulated and there should be .fit_defaults which
    #  can remove some of ._fit_defaults

    checkpointCallback = ModelCheckpoint(
        monitor=f"{self._getLossName('val', self.lossFuncs[0])}",
        mode='min',  # Save the model when the monitored quantity is minimized
        save_top_k=1,  # Save the top model based on the monitored quantity
        every_n_epochs=1,  # Checkpoint every 1 epoch
        dirpath=loggerPath,  # Directory to save checkpoints
        filename=f'BrazingTorch',
    )
    callbacks_ = [checkpointCallback, StoreEpochData()]
    kwargsApplied = {
        'logger': pl.loggers.TensorBoardLogger(self.modelName, name=architectureName,
                                               version=runName),
        'callbacks': callbacks_, }

    newSchedulers = _addDefaultSchedulers(self, addDefault_earlyStopping,
                                          addDefault_reduceLROnPlateau, warmUp_epochNum)

    doBaseFit = lambda: self.baseFit(trainDataloader=trainDataloader, valDataloader=valDataloader,
                                     addDefaultLogger=addDefaultLogger,
                                     addDefault_gradientClipping=addDefault_gradientClipping,
                                     listOfKwargs=[kwargsApplied], **kwargs)

    if len(newSchedulers) != len(self._schedulers):
        with SchedulerChanger(self, newSchedulers=newSchedulers):
            return self, doBaseFit()
    else:
        return self, doBaseFit()


def _addDefaultSchedulers(self, addDefault_earlyStopping, addDefault_reduceLROnPlateau,
                          warmUp_epochNum):
    newSchedulers = self._schedulers
    if warmUp_epochNum:
        warmUp = WarmUpScheduler(self.optimizer, warmUpEpochs=warmUp_epochNum)
        newSchedulers = [warmUp] + newSchedulers

    if addDefault_reduceLROnPlateau:
        rlrop = ReduceLROnPlateau(monitor=f"{self._getLossName('val', self.lossFuncs[0])}",
                                  mode='min', factor=0.1, patience=15, verbose=False,
                                  cooldown=0, min_lr=1e-8)

        newSchedulers = newSchedulers + [rlrop]

    if addDefault_earlyStopping:
        earlyStopping = EarlyStopping(monitor=f"{self._getLossName('val', self.lossFuncs[0])}",
                                      patience=5, verbose=False)
        newSchedulers = newSchedulers + [earlyStopping]
    return newSchedulers
