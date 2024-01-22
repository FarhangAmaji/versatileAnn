import os
from typing import List, Union

import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from torch import nn

from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import getMethodRelatedKwargs, morePreciseFloat, varPasser
from utils.warnings import Warn
from versatileAnn.newModule.callbacks import StoreEpochData


class _NewWrapper_preRunTests:
    def __init__(self, **kwargs):
        self._outputsStruct = None
        self.keepLr_notReplaceWithBestLr = kwargs.get('keepLr_notReplaceWithBestLr', False)
        self.keepBatchSize_notReplaceWithBestBatchSize = kwargs.get(
            'keepBatchSize_notReplaceWithBestBatchSize', False)

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

    @argValidator
    def preRunTests(self, trainDataloader,
                    *, lossFuncs: List[nn.modules.loss._Loss],
                    valDataloader=None,
                    lrFinderRange=(1e-6, 5), lrFinderNumSteps=20, lrsToFindBest=None,
                    batchSizesToFindBest=None,
                    fastDevRunKwargs=None, overfitBatchesKwargs=None, profilerKwargs=None,
                    findBestLearningRateKwargs=None, findBestBatchSizesKwargs=None, **kwargs):

        # mustHave3
        #  revise logging prints
        # cccUsage
        #  only first loss is used for backpropagation and others are just for logging
        if lossFuncs:
            # cccDevStruct
            #  in the case outside of trainModel lossFuncs is been set, so if not passed would use them
            self.lossFuncs = lossFuncs
        # anyway self.lossFuncs must be set
        if not self.lossFuncs:
            raise ValueError('lossFuncs must have set self.lossFuncs before running ' + \
                             'preRunTests or pass them to it')

        # mustHave2
        #  check if model with this architecture doesnt exist allow to run.
        #  - add force option also
        # find kwargs can be passed to pl.Trainer
        kwargsRelatedToTrainer = getMethodRelatedKwargs(pl.Trainer, kwargs, delAfter=True)

        # goodToHave3
        #  I tried to create a freature for saving trainDataloader, valDataloader originals
        #  in order to keep them untouched but it failed because doing deepcopy, maximum recursion
        #  stack overflow occurred meaning that it has some identical parts repeated in it. so may
        #  be add dataLoader reset later

        self.runFastDevRun(trainDataloader, valDataloader,
                           fastDevRunKwargs, kwargsRelatedToTrainer)
        self.runOverfitBatches(trainDataloader, valDataloader,
                               overfitBatchesKwargs, kwargsRelatedToTrainer)
        trainer = self.runProfiler(trainDataloader, valDataloader,
                                   profilerKwargs, kwargsRelatedToTrainer)

        kwargs_ = varPasser(localArgNames=['lrsToFindBest', 'findBestLearningRateKwargs',
                                           'kwargsRelatedToTrainer'])
        self.findBestLearningRate(trainDataloader, valDataloader,
                                  numSteps=lrFinderNumSteps, lrRange=lrFinderRange, **kwargs_)

        kwargs_ = varPasser(localArgNames=['batchSizesToFindBest', 'findBestBatchSizesKwargs',
                                           'kwargsRelatedToTrainer'])
        self.findBestBatchSize(trainDataloader, valDataloader,
                               **kwargs_)

        # goodToHave3
        #  add finding best shuffle index!!. this may be very useful sometimes

        # message how to use tensorboard
        self._printTensorboardPath(trainer)

    def runFastDevRun(self, trainDataloader, valDataloader=None,
                      fastDevRunKwargs=None, kwargsRelatedToTrainer=None):
        fastDevRunKwargs = fastDevRunKwargs or {}
        kwargsRelatedToTrainer = kwargsRelatedToTrainer or {}
        # cccDevAlgo
        #  ensures whole pipeline is working correctly by running couple of epochs on a batch
        self.printTestPrints('running fastDevRun')

        kwargsApplied = {'logger': False, }

        self._getKwargsApplied_forRelatedRun(kwargsRelatedToTrainer, fastDevRunKwargs,
                                             kwargsApplied)
        # disallow changing 'fast_dev_run' option
        if 'fast_dev_run' in kwargsApplied:
            del kwargsApplied['fast_dev_run']

        trainer = pl.Trainer(
            fast_dev_run=True,  # Run only for a small number of epochs for faster development
            **kwargsApplied)
        trainer.fit(self, train_dataloaders=trainDataloader, val_dataloaders=valDataloader)

    def runOverfitBatches(self, trainDataloader, valDataloader=None,
                          overfitBatchesKwargs=None, kwargsRelatedToTrainer=None):
        overfitBatchesKwargs = overfitBatchesKwargs or {}
        kwargsRelatedToTrainer = kwargsRelatedToTrainer or {}
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
        pastDataloaderShuffle = trainDataloader.shuffle
        trainDataloader.shuffle = False

        mainValLossName = self._getLossName('val', self.lossFuncs[0])
        callbacks_ = [StoreEpochData()]

        kwargsApplied = {'limit_train_batches': 1, 'max_epochs': 100,
                         'enable_checkpointing': False, 'logger': False, 'callbacks': callbacks_, }

        self._getKwargsApplied_forRelatedRun(kwargsRelatedToTrainer, overfitBatchesKwargs,
                                             kwargsApplied)

        if 'max_epochs' in kwargsApplied and kwargsApplied['max_epochs'] < 50:
            kwargsApplied['max_epochs'] = 50

        trainer = pl.Trainer(**kwargsApplied)
        trainer.fit(self, trainDataloader, valDataloader)

        self._printFirstNLast_valLossChanges(callbacks_)

        trainDataloader.shuffle = pastDataloaderShuffle

    def runProfiler(self, trainDataloader, valDataloader=None,
                    profilerKwargs=None, kwargsRelatedToTrainer=None):
        profilerKwargs = profilerKwargs or {}
        kwargsRelatedToTrainer = kwargsRelatedToTrainer or {}
        self.printTestPrints('running profiler')

        kwargsApplied = {'max_epochs': 4, 'enable_checkpointing': False,
                         'profiler': PyTorchProfiler(),
                         'logger': pl.loggers.TensorBoardLogger(self.modelName,
                                                                name='profiler'), }
        self._getKwargsApplied_forRelatedRun(kwargsRelatedToTrainer, profilerKwargs,
                                             kwargsApplied)

        trainer = pl.Trainer(**kwargsApplied)
        trainer.fit(self, trainDataloader, valDataloader)

        # trainer is returned to be able to print(in _printTensorboardPath) where users can use tensorboard
        return trainer

    def findBestLearningRate(self, trainDataloader, valDataloader=None,
                             *, lrRange=(1e-6, 5), numSteps=20, lrsToFindBest=None,
                             findBestLearningRateKwargs=None, kwargsRelatedToTrainer=None):
        findBestLearningRateKwargs = findBestLearningRateKwargs or {}
        kwargsRelatedToTrainer = kwargsRelatedToTrainer or {}

        kwargsApplied = {'max_epochs': 4, 'enable_checkpointing': False, 'logger': False}
        self._getKwargsApplied_forRelatedRun(kwargsRelatedToTrainer, findBestLearningRateKwargs,
                                             kwargsApplied)

        # cccUsage
        #  takes lr ranges either with (lrRange, numSteps) or with (lrsToFindBest)
        if lrsToFindBest:
            lrs = lrsToFindBest
        else:
            lrUpdateStep = (lrRange[1] / lrRange[0]) ** (1 / numSteps)
            # confines lrs to a precision of 6 digits, also with using set, only unique values are kept
            lrs = {morePreciseFloat(lrRange[0] * (lrUpdateStep ** step)) for step in
                   range(numSteps)}
        pastLr = self.lr
        lossRatioDecrease = {}

        mainValLossName = self._getLossName('val', self.lossFuncs[0])
        for thisLr in lrs:
            self = self.resetModel()
            self.resetOptimizer()  # to be sure that past accumulated params like momentum have got reset
            self.changeLearningRate(thisLr)

            callbacks_ = [StoreEpochData()]
            kwargsApplied['callbacks'] = callbacks_
            trainer = pl.Trainer(**kwargsApplied)
            # goodToHave1
            #  contextManger to disable progressBar temporarily
            trainer.fit(self, trainDataloader, valDataloader)
            self._collectBestValScores_ofMetrics(callbacks_, lossRatioDecrease,
                                                 mainValLossName, thisLr)

        bestLearningRate = min(lossRatioDecrease, key=lambda k: lossRatioDecrease[k]['score'])
        worstLearningRate = max(lossRatioDecrease, key=lambda k: lossRatioDecrease[k]['score'])
        Warn.info(f'bestLearningRate is {bestLearningRate} and the worst being {worstLearningRate}')

        if not self.keepLr_notReplaceWithBestLr:
            self.lr = bestLearningRate
        else:
            self.lr = pastLr
        # goodToHave2
        #  add ploting in tensorboard with a message that plot is in tensorboard

    @argValidator
    def findBestBatchSize(self, trainDataloader, valDataloader=None,
                          *, batchSizesToFindBest: Union[None, List],
                          findBestBatchSizesKwargs=None, kwargsRelatedToTrainer=None):
        findBestBatchSizesKwargs = findBestBatchSizesKwargs or {}
        kwargsRelatedToTrainer = kwargsRelatedToTrainer or {}
        batchSizesToFindBest = batchSizesToFindBest or [8, 16, 32, 64, 128]

        kwargsApplied = {'max_epochs': 4, 'enable_checkpointing': False, 'logger': False}
        self._getKwargsApplied_forRelatedRun(kwargsRelatedToTrainer, findBestBatchSizesKwargs,
                                             kwargsApplied)

        pastBatchSize = trainDataloader.batch_size
        lossRatioDecrease = {}

        mainValLossName = self._getLossName('val', self.lossFuncs[0])
        for thisBatchSize in batchSizesToFindBest:
            self = self.resetModel()
            self.resetOptimizer()
            trainDataloader = trainDataloader.changeBatchSize(thisBatchSize)
            callbacks_ = [StoreEpochData()]
            kwargsApplied['callbacks'] = callbacks_
            trainer = pl.Trainer(**kwargsApplied)
            # goodToHave1
            #  contextManger to disable progressBar temporarily
            trainer.fit(self, trainDataloader, valDataloader)
            self._collectBestValScores_ofMetrics(callbacks_, lossRatioDecrease,
                                                 mainValLossName, thisBatchSize)

        bestBatchSize = min(lossRatioDecrease, key=lambda k: lossRatioDecrease[k]['score'])
        worstBatchSize = max(lossRatioDecrease, key=lambda k: lossRatioDecrease[k]['score'])
        Warn.info(f'bestBatchSize is {bestBatchSize} and the worst being {worstBatchSize}')

        if not self.keepBatchSize_notReplaceWithBestBatchSize:
            trainDataloader = trainDataloader.changeBatchSize(bestBatchSize)
        else:
            trainDataloader = trainDataloader.changeBatchSize(pastBatchSize)
        # goodToHave2
        #  add ploting in tensorboard with a message that plot is in tensorboard

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
    def _getKwargsApplied_forRelatedRun(kwargsRelatedToTrainer, runKwargs,
                                        kwargsApplied):
        # cccUsage
        #  the runKwargs which is kwargs passed by user for specific run, have higher priority
        # cccAlgo
        #  in finds kwargs of trainModel or runKwargs(for i.e. fastDevRunKwargs) related to pl.Trainer
        runKwargs = getMethodRelatedKwargs(pl.Trainer, updater=runKwargs,
                                           updatee=runKwargs, delAfter=True)
        kwargsApplied.update(kwargsRelatedToTrainer)
        kwargsApplied.update(runKwargs)

    @staticmethod
    def _printTensorboardPath(trainer):
        # the last 2 folders are not included
        tensorboardDir = os.path.abspath(trainer.logger.log_dir)
        tensorboardDir = os.path.split(os.path.split(tensorboardDir)[0])[0]
        Warn.info("to see tensorboard in terminal execute: 'python -m tensorboard.main --logdir " +
                  f'"{tensorboardDir}"' + "'")
