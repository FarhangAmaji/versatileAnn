from typing import List, Union, Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger
from pytorch_lightning.profilers import PyTorchProfiler
from torch import nn
from torch.utils.data import DataLoader

from brazingTorchFolder.brazingTorchParents.innerClassesWithoutPublicMethods.preRunTests_inner import \
    _BrazingTorch_preRunTests_inner
from brazingTorchFolder.utilsFolder.callbacks import StoreEpochData
from projectUtils.misc import morePreciseFloat, _allowOnlyCreationOf_ChildrenInstances
from projectUtils.typeCheck import argValidator
from projectUtils.warnings import Warn


class _BrazingTorch_preRunTests(_BrazingTorch_preRunTests_inner):
    def __init__(self, keepLr_notReplaceWithBestLr: Optional[bool] = False,
                 keepBatchSize_notReplaceWithBestBatchSize: Optional[bool] = False, **kwargs):
        self._outputsStruct = None
        self.keepLr_notReplaceWithBestLr = keepLr_notReplaceWithBestLr
        self.keepBatchSize_notReplaceWithBestBatchSize = keepBatchSize_notReplaceWithBestBatchSize

        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _BrazingTorch_preRunTests)

    # ---- properties
    @property
    def keepLr_notReplaceWithBestLr(self):
        return self._keepLr_notReplaceWithBestLr

    @keepLr_notReplaceWithBestLr.setter
    @argValidator
    def keepLr_notReplaceWithBestLr(self, value: bool):
        self._keepLr_notReplaceWithBestLr = value

    @property
    def keepBatchSize_notReplaceWithBestBatchSize(self):
        return self._keepBatchSize_notReplaceWithBestBatchSize

    @keepBatchSize_notReplaceWithBestBatchSize.setter
    @argValidator
    def keepBatchSize_notReplaceWithBestBatchSize(self, value: bool):
        self._keepBatchSize_notReplaceWithBestBatchSize = value

    # ----
    @argValidator
    def preRunTests(self, trainDataloader,
                    valDataloader: Optional[DataLoader] = None,
                    *, lossFuncs: List[nn.modules.loss._Loss],
                    force=False, seedSensitive=False,
                    lrFinderRange=(1e-6, 5), lrFinderNumSteps=20, lrsToFindBest=None,
                    batchSizesToFindBest=None,
                    fastDevRunKwargs=None, overfitBatchesKwargs=None, profilerKwargs=None,
                    findBestLearningRateKwargs=None, findBestBatchSizesKwargs=None, **kwargs):

        # cccUsage
        #  - seedSensitive: to know what is seedSensitive read _shouldRun_preRunTests_seedSensitivePart docs
        #  - force: by default if the model has run before with the same name and the same structure
        #         its prevented to run but force forces to rerun
        #  - customizing learning rates to search:
        #       either should pass (lrFinderRange and lrFinderNumSteps) or lrsToFindBest
        #  - (*RunKwargs)== fastDevRunKwargs, overfitBatchesKwargs, profilerKwargs,
        #       findBestLearningRateKwargs, findBestBatchSizesKwargs:
        #           - these are the kwargs which are passed to the corresponding run methods
        #           - you can pass kwargs related to pytorch lightning trainer, trainer.fit,
        #               and self.log
        #  - **kwargs same as (*RunKwargs) but are passed to all run methods

        fastDevRunKwargs = fastDevRunKwargs or {}
        overfitBatchesKwargs = overfitBatchesKwargs or {}
        profilerKwargs = profilerKwargs or {}
        findBestLearningRateKwargs = findBestLearningRateKwargs or {}
        findBestBatchSizesKwargs = findBestBatchSizesKwargs or {}

        # goodToHave3
        #  revise logging prints
        self._setLossFuncs_ifNot(lossFuncs)

        # cccWhat
        #  in general check if model with this architecture doesn't exist allow to run.
        #  note but there are some cases depending on force and seedSensitive which
        #  also affect allowing to run. read the _determineShouldRun_preRunTests comments
        architectureName, loggerPath, shouldRun_preRunTests = self._determineShouldRun_preRunTests(
            force, seedSensitive)

        # specify default_root_dir in order to specify model save path
        # note as how runs are defined here(by default) there is no model save
        # but in the case the user specifies it, it would be saved in the loggerPath
        if 'default_root_dir' not in kwargs:
            kwargs['default_root_dir'] = loggerPath
        else:
            Warn.info("with specifying default_root_dir, the automatic feature of " + \
                      "model architecture detection won't work.")

        if not shouldRun_preRunTests:
            return

        # goodToHave3
        #  I tried to create a feature for saving original trainDataloader, valDataloader
        #  in order to keep them untouched but it failed because doing deepcopy, maximum recursion
        #  stack overflow occurred meaning that it has some identical parts repeated in it. so may
        #  be add dataLoader reset later
        runKwargs_ = self._mergeKwargsWith_runKwargs(kwargs, fastDevRunKwargs)
        self.runFastDevRun(trainDataloader, valDataloader, **runKwargs_)

        runKwargs_ = self._mergeKwargsWith_runKwargs(kwargs, overfitBatchesKwargs)
        self.runOverfitBatches(trainDataloader, valDataloader, **runKwargs_)

        runKwargs_ = self._mergeKwargsWith_runKwargs(kwargs, profilerKwargs)
        self.runProfiler(trainDataloader, architectureName,
                         valDataloader, **runKwargs_)

        runKwargs_ = self._mergeKwargsWith_runKwargs(kwargs, findBestLearningRateKwargs)
        self.findBestLearningRate(trainDataloader, valDataloader,
                                  numSteps=lrFinderNumSteps, lrRange=lrFinderRange,
                                  lrsToFindBest=lrsToFindBest, **runKwargs_)

        runKwargs_ = self._mergeKwargsWith_runKwargs(kwargs, findBestBatchSizesKwargs)
        self.findBestBatchSize(trainDataloader, valDataloader,
                               batchSizesToFindBest=batchSizesToFindBest, **runKwargs_)

        # goodToHave3
        #  add finding best shuffle index!!. this may be very useful sometimes

        # message how to use tensorboard
        self._informTensorboardPath(fastDevRunKwargs, findBestBatchSizesKwargs,
                                    findBestLearningRateKwargs, kwargs, overfitBatchesKwargs,
                                    profilerKwargs)

        # save model architecture in a file
        self._saveArchitectureDict(loggerPath)

    # ---- runs
    @argValidator
    def runFastDevRun(self, trainDataloader: DataLoader,
                      valDataloader: Union[DataLoader, None] = None, **kwargs):
        # ccc1
        #  ensures whole pipeline is working correctly by running couple of epochs on a batch
        self._printTestPrints('running fastDevRun')

        kwargsApplied = {'log': {'logger': False, }, 'trainer': {'logger': False, },
                         'trainerFit': {}}
        kwargsApplied = self._plKwargUpdater(kwargsApplied, kwargs)
        # bugPotn2 to have logger equal to false

        # force setting 'fast_dev_run' True
        kwargsApplied['trainer']['fast_dev_run'] = True

        self.baseFit(trainDataloader, valDataloader, addDefaultLogger=False, **kwargsApplied)

    @argValidator
    def runOverfitBatches(self, trainDataloader: DataLoader,
                          valDataloader: Union[DataLoader, None] = None,
                          **kwargs):
        self._printTestPrints('running overfitBatches')

        # ccc1 # bugPotn1
        #  with including 'overfit_batches' option, when the trainer is ran, "make sure you have
        #  set, VAnnTsDataset.indexes to .indexes of sampler". this is an indication of that the
        #  pytorchLighning tries to re__init__ the dataLoader.
        #  even though it seems the dataLoaders passed here are kept unchanged and this
        #  reiniting are applied just internally. because the sampler of trainDataloader is
        #  still is instance of SamplerFor_vAnnTsDataset
        #  but it gives error over .indexes of sampler which is an indicator which the sampler has
        #  been replaced completely. so it's better to not include 'overfit_batches' option and try
        #  to replicate it. so decided to use 'limit_train_batches' option instead. ofc with setting
        #  dataloader shuffle to False temporarily then turn it back to its original
        # bugPotn1
        #  the amount which decreases the loss over 200 epochs is not really sth to be called
        #  overfitting
        #  https://stackoverflow.com/questions/77854815/replicating-overfit-batches-functionality-of-pytorch-lightning

        pastDataloaderShuffle = trainDataloader.shuffle
        trainDataloader.shuffle = False

        callbacks_ = [StoreEpochData()]

        kwargsApplied = {
            'trainer': {'limit_train_batches': 1, 'max_epochs': 100, 'enable_checkpointing': False,
                        'callbacks': callbacks_, 'logger': False}, 'log': {'logger': False, },
            'trainerFit': {}}
        kwargsApplied = self._plKwargUpdater(kwargsApplied, kwargs)

        if 'max_epochs' in kwargsApplied['trainer'] and kwargsApplied['trainer']['max_epochs'] < 50:
            kwargsApplied['trainer']['max_epochs'] = 50

        self.baseFit(trainDataloader, valDataloader, addDefaultLogger=False, **kwargsApplied)

        self._printFirstNLast_valLossChanges(callbacks_)

        trainDataloader.shuffle = pastDataloaderShuffle

    @argValidator
    def runProfiler(self, trainDataloader: DataLoader, architectureName,
                    valDataloader: Union[DataLoader, None] = None,
                    **kwargs):
        self._printTestPrints('running profiler')
        kwargsApplied = {'trainer': {'max_epochs': 4, 'enable_checkpointing': False,
                                     'profiler': PyTorchProfiler(),
                                     'logger': pl.loggers.TensorBoardLogger(self.modelName,
                                                                            name=architectureName,
                                                                            version='preRunTests')},
                         'log': {}, 'trainerFit': {}}
        kwargsApplied = self._plKwargUpdater(kwargsApplied, kwargs)

        trainer = self.baseFit(trainDataloader, valDataloader, **kwargsApplied)

    @argValidator
    def findBestLearningRate(self, trainDataloader: DataLoader,
                             valDataloader: Union[DataLoader, None] = None,
                             *, lrRange=(1e-6, 5), numSteps=20, lrsToFindBest=None,
                             **kwargs):

        kwargsApplied = {'max_epochs': 4, 'enable_checkpointing': False, 'logger': False}
        kwargsApplied = self._plKwargUpdater(kwargsApplied, kwargs)

        pastLr = self.lr
        lossRatioDecrease = {}

        # cccUsage
        #  takes lr ranges either with (lrRange, numSteps) or with (lrsToFindBest)
        if lrsToFindBest:
            lrs = lrsToFindBest
        else:
            lrUpdateStep = (lrRange[1] / lrRange[0]) ** (1 / numSteps)
            # confines lrs to a precision of 6 digits, also with using set, only unique values are kept
            lrs = {morePreciseFloat(lrRange[0] * (lrUpdateStep ** step)) for step in
                   range(numSteps)}

        mainValLossName = self._getLossName('val', self.lossFuncs[0])
        for thisLr in lrs:
            self = self.resetModel()
            self.resetOptimizer()  # to be sure that past accumulated params like momentum have got reset
            self.changeLearningRate(thisLr)

            kwargsAppliedCopy = kwargsApplied.copy()
            callbacks_ = [StoreEpochData()]
            # its wrong to 'kwargsApplied['callbacks'] = callbacks_'
            callbacks_Kwargs = {'callbacks': callbacks_}
            kwargsAppliedCopy = self._plKwargUpdater(kwargsAppliedCopy, callbacks_Kwargs)

            self.baseFit(trainDataloader, valDataloader, addDefaultLogger=False,
                         **kwargsAppliedCopy)
            self._collectBestValScores_ofMetrics(callbacks_, lossRatioDecrease,
                                                 mainValLossName, thisLr)

        bestLearningRate = min(lossRatioDecrease, key=lambda k: lossRatioDecrease[k]['score'])
        worstLearningRate = max(lossRatioDecrease, key=lambda k: lossRatioDecrease[k]['score'])
        Warn.info(f'bestLearningRate is {bestLearningRate} and the worst being {worstLearningRate}')

        # set back lr or not
        if not self.keepLr_notReplaceWithBestLr:
            self.lr = bestLearningRate
            Warn.info(f"changed learningRate to bestLearningRate." + \
                      "\nif you want to keep ur learningRate pass" + \
                      "keepLr_notReplaceWithBestLr=True to model.")
        else:
            self.lr = pastLr
        # goodToHave2
        #  add ploting in tensorboard with a message that plot is in tensorboard

    @argValidator
    def findBestBatchSize(self, trainDataloader: DataLoader,
                          valDataloader: Union[DataLoader, None] = None,
                          *, batchSizesToFindBest: Union[None, List],
                          **kwargs):
        # goodToHave3
        #  check if the batchSizes are power of 2 are they slower or faster or
        #  doesnt make any difference
        batchSizesToFindBest = batchSizesToFindBest or [8, 16, 32, 64, 128]

        kwargsApplied = {'max_epochs': 4, 'enable_checkpointing': False, 'logger': False}
        kwargsApplied = self._plKwargUpdater(kwargsApplied, kwargs)

        pastBatchSize = trainDataloader.batch_size
        lossRatioDecrease = {}

        mainValLossName = self._getLossName('val', self.lossFuncs[0])
        for thisBatchSize in batchSizesToFindBest:
            self = self.resetModel()
            self.resetOptimizer()
            trainDataloader = trainDataloader.changeBatchSize(thisBatchSize)

            kwargsAppliedCopy = kwargsApplied.copy()
            callbacks_ = [StoreEpochData()]
            callbacks_Kwargs = {'callbacks': callbacks_}
            kwargsAppliedCopy = self._plKwargUpdater(kwargsAppliedCopy, callbacks_Kwargs)

            self.baseFit(trainDataloader, valDataloader, addDefaultLogger=False,
                         **kwargsAppliedCopy)
            self._collectBestValScores_ofMetrics(callbacks_, lossRatioDecrease,
                                                 mainValLossName, thisBatchSize)

        bestBatchSize = min(lossRatioDecrease, key=lambda k: lossRatioDecrease[k]['score'])
        worstBatchSize = max(lossRatioDecrease, key=lambda k: lossRatioDecrease[k]['score'])
        Warn.info(f'bestBatchSize is {bestBatchSize} and the worst being {worstBatchSize}')

        # set back batchSize or not
        if not self.keepBatchSize_notReplaceWithBestBatchSize:
            trainDataloader = trainDataloader.changeBatchSize(bestBatchSize)
            Warn.info(f"changed batchSize to bestBatchSize." + \
                      "\nif you want to keep ur batchSize pass" + \
                      "keepBatchSize_notReplaceWithBestBatchSize=True to model.")
        else:
            trainDataloader = trainDataloader.changeBatchSize(pastBatchSize)
        # goodToHave2
        #  add ploting in tensorboard with a message that plot is in tensorboard
