import os
from typing import List, Union

import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from torch import nn
from torch.utils.data import DataLoader

from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import morePreciseFloat
from utils.warnings import Warn
from versatileAnn.newModule.callbacks import StoreEpochData


class _NewWrapper_preRunTests:
    def __init__(self, **kwargs):
        self._outputsStruct = None
        self.keepLr_notReplaceWithBestLr = kwargs.get('keepLr_notReplaceWithBestLr', False)
        self.keepBatchSize_notReplaceWithBestBatchSize = kwargs.get(
            'keepBatchSize_notReplaceWithBestBatchSize', False)

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
                    lrFinderRange=(1e-6, 5), lrFinderNumSteps=20, lrsToFindBest=None,
                    batchSizesToFindBest=None,
                    fastDevRunKwargs=None, overfitBatchesKwargs=None, profilerKwargs=None,
                    findBestLearningRateKwargs=None, findBestBatchSizesKwargs=None, **kwargs):

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

        # mustHave2
        #  check if model with this architecture doesnt exist allow to run.
        #  - add force option also

        # goodToHave3
        #  I tried to create a freature for saving trainDataloader, valDataloader originals
        #  in order to keep them untouched but it failed because doing deepcopy, maximum recursion
        #  stack overflow occurred meaning that it has some identical parts repeated in it. so may
        #  be add dataLoader reset later
        runKwargs_ = self._mergeKwargsWith_runKwargs(fastDevRunKwargs, kwargs)
        self.runFastDevRun(trainDataloader, valDataloader, **runKwargs_)

        runKwargs_ = self._mergeKwargsWith_runKwargs(overfitBatchesKwargs, kwargs)
        self.runOverfitBatches(trainDataloader, valDataloader, **runKwargs_)

        runKwargs_ = self._mergeKwargsWith_runKwargs(profilerKwargs, kwargs)
        trainer = self.runProfiler(trainDataloader, valDataloader, **runKwargs_)

        runKwargs_ = self._mergeKwargsWith_runKwargs(findBestLearningRateKwargs, kwargs)
        self.findBestLearningRate(trainDataloader, valDataloader,
                                  numSteps=lrFinderNumSteps, lrRange=lrFinderRange,
                                  lrsToFindBest=lrsToFindBest, **runKwargs_)

        runKwargs_ = self._mergeKwargsWith_runKwargs(findBestBatchSizesKwargs, kwargs)
        self.findBestBatchSize(trainDataloader, valDataloader,
                               batchSizesToFindBest=batchSizesToFindBest, **runKwargs_)

        # goodToHave3
        #  add finding best shuffle index!!. this may be very useful sometimes

        # message how to use tensorboard
        self._printTensorboardPath(trainer)

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

        self.fit(trainDataloader, valDataloader, **kwargsApplied)

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

        self.fit(trainDataloader, valDataloader, **kwargsApplied)

        self._printFirstNLast_valLossChanges(callbacks_)

        trainDataloader.shuffle = pastDataloaderShuffle

    @argValidator
    def runProfiler(self, trainDataloader: DataLoader,
                    valDataloader: Union[DataLoader, None] = None,
                    **kwargs):
        self.printTestPrints('running profiler')

        kwargsApplied = {'max_epochs': 4, 'enable_checkpointing': False,
                         'profiler': PyTorchProfiler(),
                         'logger': pl.loggers.TensorBoardLogger(self.modelName, name='profiler'), }
        self._plKwargUpdater(kwargsApplied, kwargs)

        trainer = self.fit(trainDataloader, valDataloader, **kwargsApplied)

        # trainer is returned to be able to print(in _printTensorboardPath) where users can use tensorboard
        return trainer

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

            self.fit(trainDataloader, valDataloader, **kwargsAppliedCopy)
            self._collectBestValScores_ofMetrics(callbacks_, lossRatioDecrease,
                                                 mainValLossName, thisLr)

        bestLearningRate = min(lossRatioDecrease, key=lambda k: lossRatioDecrease[k]['score'])
        worstLearningRate = max(lossRatioDecrease, key=lambda k: lossRatioDecrease[k]['score'])
        Warn.info(f'bestLearningRate is {bestLearningRate} and the worst being {worstLearningRate}')

        # set back lr or not
        if not self.keepLr_notReplaceWithBestLr:
            self.lr = bestLearningRate
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

            self.fit(trainDataloader, valDataloader, **kwargsAppliedCopy)
            self._collectBestValScores_ofMetrics(callbacks_, lossRatioDecrease,
                                                 mainValLossName, thisBatchSize)

        bestBatchSize = min(lossRatioDecrease, key=lambda k: lossRatioDecrease[k]['score'])
        worstBatchSize = max(lossRatioDecrease, key=lambda k: lossRatioDecrease[k]['score'])
        Warn.info(f'bestBatchSize is {bestBatchSize} and the worst being {worstBatchSize}')

        # set back batchSize or not
        if not self.keepBatchSize_notReplaceWithBestBatchSize:
            trainDataloader = trainDataloader.changeBatchSize(bestBatchSize)
        else:
            trainDataloader = trainDataloader.changeBatchSize(pastBatchSize)
        # goodToHave2
        #  add ploting in tensorboard with a message that plot is in tensorboard

    # ---- utils
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

    @staticmethod
    def _printTensorboardPath(trainer):
        # the last 2 folders are not included
        tensorboardDir = os.path.abspath(trainer.logger.log_dir)
        tensorboardDir = os.path.split(os.path.split(tensorboardDir)[0])[0]
        Warn.info("to see tensorboard in terminal execute: 'python -m tensorboard.main --logdir " +
                  f'"{tensorboardDir}"' + "'")

    def _mergeKwargsWith_runKwargs(self, runKwargs, mainKwargsOfPreRunTests):
        result = mainKwargsOfPreRunTests.copy()
        self.self._plKwargUpdater(result, runKwargs)
        return result
