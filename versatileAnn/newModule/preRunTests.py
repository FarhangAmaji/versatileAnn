import os
import pickle
from typing import List, Union, Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger
from pytorch_lightning.profilers import PyTorchProfiler
from torch import nn
from torch.utils.data import DataLoader

from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import morePreciseFloat, nFoldersBack, stringValuedDictsEqual
from utils.warnings import Warn
from versatileAnn.newModule.callbacks import StoreEpochData


class _NewWrapper_preRunTests:
    def __init__(self, keepLr_notReplaceWithBestLr: Optional[bool] = False,
                 keepBatchSize_notReplaceWithBestBatchSize: Optional[bool] = False, **kwargs):
        self._outputsStruct = None
        self.keepLr_notReplaceWithBestLr = keepLr_notReplaceWithBestLr
        self.keepBatchSize_notReplaceWithBestBatchSize = keepBatchSize_notReplaceWithBestBatchSize

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
                    *, lossFuncs: List[nn.modules.loss._Loss],
                    valDataloader=None,
                    force=False, seedSensitive=False,
                    lrFinderRange=(1e-6, 5), lrFinderNumSteps=20, lrsToFindBest=None,
                    batchSizesToFindBest=None,
                    fastDevRunKwargs=None, overfitBatchesKwargs=None, profilerKwargs=None,
                    findBestLearningRateKwargs=None, findBestBatchSizesKwargs=None, **kwargs):
        # cccUsage
        #  - seedSensitive: to know what is seedSensitive read _determineSeedSensitive_shouldRun docs
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
        # mustHave3
        #  revise logging prints
        if lossFuncs:
            self.lossFuncs = lossFuncs
            # cccUsage
            #  only first loss is used for backpropagation and others are just for logging
            # cccDevStruct
            #  in the case outside of trainModel lossFuncs is been set, so if not passed would use them
        # anyway self.lossFuncs must be set
        if not self.lossFuncs:
            raise ValueError('lossFuncs must have set self.lossFuncs before running ' + \
                             'preRunTests or pass them to it')

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
        # cccDevAlgo
        #  ensures whole pipeline is working correctly by running couple of epochs on a batch
        self.printTestPrints('running fastDevRun')

        kwargsApplied = {'logger': False, }
        self._plKwargUpdater(kwargsApplied, kwargs)

        # force setting 'fast_dev_run' True
        kwargsApplied['fast_dev_run'] = True

        self.fit(trainDataloader, valDataloader, addDefaultLogger=False, **kwargsApplied)

    @argValidator
    def runOverfitBatches(self, trainDataloader: DataLoader,
                          valDataloader: Union[DataLoader, None] = None,
                          **kwargs):
        self.printTestPrints('running overfitBatches')

        # cccDevStruct # bugPotentialCheck1
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
        # bugPotentialCheck1
        #  the amount which decreases the loss over 200 epochs is not really sth to be called
        #  overfitting
        #  https://stackoverflow.com/questions/77854815/replicating-overfit-batches-functionality-of-pytorch-lightning

        pastDataloaderShuffle = trainDataloader.shuffle
        trainDataloader.shuffle = False

        callbacks_ = [StoreEpochData()]

        kwargsApplied = {'limit_train_batches': 1, 'max_epochs': 100,
                         'enable_checkpointing': False, 'logger': False,
                         'callbacks': callbacks_, }
        self._plKwargUpdater(kwargsApplied, kwargs)

        if 'max_epochs' in kwargsApplied and kwargsApplied['max_epochs'] < 50:
            kwargsApplied['max_epochs'] = 50

        self.fit(trainDataloader, valDataloader, addDefaultLogger=False, **kwargsApplied)

        self._printFirstNLast_valLossChanges(callbacks_)

        trainDataloader.shuffle = pastDataloaderShuffle

    @argValidator
    def runProfiler(self, trainDataloader: DataLoader, architectureName,
                    valDataloader: Union[DataLoader, None] = None,
                    **kwargs):
        self.printTestPrints('running profiler')

        kwargsApplied = {'max_epochs': 4, 'enable_checkpointing': False,
                         'profiler': PyTorchProfiler(),
                         'logger': pl.loggers.TensorBoardLogger(self.modelName,
                                                                name=architectureName,
                                                                version='preRunTests'), }
        self._plKwargUpdater(kwargsApplied, kwargs)

        trainer = self.fit(trainDataloader, valDataloader, **kwargsApplied)

    @argValidator
    def findBestLearningRate(self, trainDataloader: DataLoader,
                             valDataloader: Union[DataLoader, None] = None,
                             *, lrRange=(1e-6, 5), numSteps=20, lrsToFindBest=None,
                             **kwargs):

        kwargsApplied = {'max_epochs': 4, 'enable_checkpointing': False, 'logger': False}
        self._plKwargUpdater(kwargsApplied, kwargs)

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
            self._plKwargUpdater(kwargsAppliedCopy, callbacks_Kwargs)

            self.fit(trainDataloader, valDataloader, addDefaultLogger=False, **kwargsAppliedCopy)
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
        self._plKwargUpdater(kwargsApplied, kwargs)

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
            self._plKwargUpdater(kwargsAppliedCopy, callbacks_Kwargs)

            self.fit(trainDataloader, valDataloader, addDefaultLogger=False, **kwargsAppliedCopy)
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

    # ---- runs utils
    @staticmethod
    def _collectBestValScores_ofMetrics(callbacks_, lossRatioDecrease,
                                        mainValLossName, changingParam):
        firstEpochLoss = callbacks_[0].epochData['val'][0][mainValLossName]

        lastEpochNum = list(callbacks_[0].epochData['val'].keys())
        lastEpochNum.sort()
        lastEpochNum = lastEpochNum[-1]
        lastEpochLoss = callbacks_[0].epochData['val'][lastEpochNum][mainValLossName]
        lossRatioDecrease.update({changingParam: {'ratio': lastEpochLoss / firstEpochLoss,
                                                  '1st': firstEpochLoss, 'last': lastEpochLoss,
                                                  'score': lastEpochLoss / firstEpochLoss * lastEpochLoss}})

    def _mergeKwargsWith_runKwargs(self, mainKwargsOfPreRunTests, runKwargs):
        result = mainKwargsOfPreRunTests.copy()
        self._plKwargUpdater(result, runKwargs)
        return result

    # ---- _determineShouldRun_preRunTests
    def _determineShouldRun_preRunTests(self, force, seedSensitive):
        # addTest1

        # by default these values are assumed
        shouldRun_preRunTests = True
        architectureName = 'arch1'

        dummyLogger = pl.loggers.TensorBoardLogger(self.modelName)
        loggerPath = os.path.abspath(dummyLogger.log_dir)
        # loggerPath is fullPath including 'modelName/preRunTests/version_0'

        if os.path.exists(nFoldersBack(loggerPath, n=2)):
            # there is a model run before with the name of this model
            architectureDicts = self._collectArchDicts(loggerPath)
            architectureDicts_withMatchedAllDefinitions = self._getArchitectureDicts_withMatchedAllDefinitions(
                architectureDicts)
            # matchedAllDefinitions means the exact same model structure as all layers and their definitions are exactly the same
            # note architectureDicts matches _saveArchitectureDict
            if architectureDicts_withMatchedAllDefinitions:
                if force:
                    # the force is True so the user wants replace model's previous results therefore
                    # we have to find architectureName, so to know where are the past results
                    acw = architectureDicts_withMatchedAllDefinitions[0]
                    filePath = acw.keys()[0]
                    architectureName = os.path.basename(filePath)
                else:
                    architectureName, shouldRun_preRunTests = self._determineSeedSensitive_shouldRun(
                        architectureDicts_withMatchedAllDefinitions, architectureName, loggerPath,
                        seedSensitive, shouldRun_preRunTests)

            else:
                # there are models with the name of this model but with different structures
                pass  # so default shouldRun_preRunTests and architectureName are applied

        else:
            # no model with this name in directory has never run
            pass  # so default shouldRun_preRunTests and architectureName are applied

        dummyLogger = pl.loggers.TensorBoardLogger(self.modelName,
                                                   name=architectureName,
                                                   version='preRunTests')
        loggerPath = os.path.abspath(dummyLogger.log_dir)

        return architectureName, loggerPath, shouldRun_preRunTests

    def _getArchitectureDicts_withMatchedAllDefinitions(self, architectureDicts):
        # cccWhat
        # this func checks the match between self.allDefinitions and allDefinitions
        # in architectureDicts and brings back 'architectureDicts_withMatchedAllDefinitions' which
        # is a list of architectureDicts
        # cccWhy
        # 1. self.allDefinitions is a list of some dicts which have 'func or class names'
        # as key and there string definition, sth like
        #   [{'class1Parent': 'class class1Parent:\n    def __init__(self):\n        self.var1 = 1\n'},
        #   {'func1': "def func1():\n    print('func1')\n"}]
        # 2. architectureDicts is a list of dicts like
        #   {filePath:{'allDefinitions': allDefinitions, '__plSeed__': someNumber}}

        # Convert list of dicts to a single dict
        toDictConvertor = lambda list_: {k: v for d in list_ for k, v in d.items()}

        mainAllDefinitions_dict = toDictConvertor(self.allDefinitions)

        architectureDicts_withMatchedAllDefinitions = []

        for archDict in architectureDicts:
            for filePath, fileDict in archDict.items():
                allDefinitions = toDictConvertor(fileDict['allDefinitions'])

                if stringValuedDictsEqual(mainAllDefinitions_dict, allDefinitions):
                    architectureDicts_withMatchedAllDefinitions.append(archDict)

        return architectureDicts_withMatchedAllDefinitions

    def _determineSeedSensitive_shouldRun(self, architectureDicts_withMatchedAllDefinitions,
                                          architectureName, loggerPath, seedSensitive,
                                          shouldRun_preRunTests):
        if seedSensitive:
            # seedSensitive True means the user wants:
            # seedCase1:
            #       even if there is a model with same structure but its seed
            #       differs, so run the model with the new seed (the seed
            #       passed to this run)
            # seedCase2:
            #       but if the seed passed to this run has run before so no need to run
            foundSeedMatch = False
            thisModelSeed = self._initArgs['__plSeed__']
            for acw in architectureDicts_withMatchedAllDefinitions:
                filePath = acw.keys()[0]
                if thisModelSeed == acw[filePath]['__plSeed__']:
                    # seedCase2
                    foundSeedMatch = True
                    shouldRun_preRunTests = False  # just for clarity but may change
                    if not os.path.join(filePath, 'preRunTests').exists():
                        # this exact model even with this seed has run before but
                        # its 'preRunTests' has not
                        shouldRun_preRunTests = True
                        architectureName = os.path.basename(filePath)

                    if not shouldRun_preRunTests:
                        Warn.info(
                            'skipping preRunTests: this model with same structure and same seed has run before')
                    break

            if not foundSeedMatch:  # seedCase1
                # we have to find architectureName which doesn't exist,
                # in order not to overwrite the previous results
                architectureName = self.findAvailableArchName(
                    nFoldersBack(loggerPath, n=1))
        else:
            # there are models with the name of this model
            # also same structure, and the seed is not important factor
            # so there is no need to run
            shouldRun_preRunTests = False  # just for clarity but may change
            acw = architectureDicts_withMatchedAllDefinitions[0]
            filePath = acw.keys()[0]
            if not os.path.join(filePath, 'preRunTests').exists():
                # this exact model has run before but its 'preRunTests' has not
                architectureName = os.path.basename(filePath)
                shouldRun_preRunTests = True

            if not shouldRun_preRunTests:
                Warn.info('skipping preRunTests: this model with same structure has run before')
        return architectureName, shouldRun_preRunTests

    def findAvailableArchName(self, folderToSearch):
        """
        Find the first available 'arch' folder within the specified parent folder.
        """
        i = 0
        while True:
            i += 1
            archName = f'arch{i}'
            folderPath = os.path.join(folderToSearch, archName)

            if os.path.exists(folderPath) and os.path.isdir(folderPath):
                continue
            else:
                return archName

    def _collectArchDicts(self, loggerPath):
        pickleFiles = []

        path = nFoldersBack(loggerPath, n=2)
        for file in os.listdir(path):
            if file == 'architecture.pkl':
                pickleFiles.append(os.path.join(path, file))

        architectureDicts = []
        for pickleFile in pickleFiles:
            with open(pickleFile, 'rb') as f:
                architectureDict = pickle.load(f)
                architectureDict = {pickleFile: architectureDict}
                architectureDicts.append(architectureDict)

        return architectureDicts

    # ----
    def _informTensorboardPath(self, fastDevRunKwargs, findBestBatchSizesKwargs,
                               findBestLearningRateKwargs, kwargs, overfitBatchesKwargs,
                               profilerKwargs):
        loggingPaths = self._getPreRunLoggingPaths(fastDevRunKwargs, overfitBatchesKwargs,
                                                   profilerKwargs, findBestLearningRateKwargs,
                                                   findBestBatchSizesKwargs, kwargs)
        for loggingPath in loggingPaths:
            Warn.info(
                "to see tensorboard in terminal execute: 'python -m tensorboard.main --logdir " +
                f'"{loggingPath}"' + "'")

    def _getPreRunLoggingPaths(self, fastDevRunKwargs, overfitBatchesKwargs, profilerKwargs,
                               findBestLearningRateKwargs, findBestBatchSizesKwargs, kwargs):
        loggingPaths = []
        for kwarg in [kwargs, fastDevRunKwargs, overfitBatchesKwargs, profilerKwargs,
                      findBestLearningRateKwargs, findBestBatchSizesKwargs]:
            if 'logger' in kwarg:
                logger = kwarg['logger']

                if isinstance(logger, Logger):
                    self._addLogDirTOLoggingPaths(loggingPaths, logger)
                else:  # logger may be a logger instance and be a list of Loggers
                    for logger in kwarg['logger']:
                        self._addLogDirTOLoggingPaths(loggingPaths, logger)

        if not loggingPaths:
            dummyLogger = pl.loggers.TensorBoardLogger(self.modelName, name='a')
            loggingPaths.append(nFoldersBack(os.path.abspath(dummyLogger.log_dir), n=2))
        return loggingPaths

    def _addLogDirTOLoggingPaths(self, loggingPaths, logger):
        logDir = os.path.abspath(logger.log_dir)
        if logDir not in loggingPaths:
            loggingPaths.append(logDir)

    # ----

    def _saveArchitectureDict(self, loggerPath):
        architectureDict = {'allDefinitions': self.allDefinitions,
                            'seed': self._initArgs['__plSeed__']}

        with open(os.path.join(loggerPath, 'architecture.pkl'), 'wb') as f:
            pickle.dump(architectureDict, f)
