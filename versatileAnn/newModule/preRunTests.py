import os
from typing import List

import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from torch import nn

from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import getMethodRelatedKwargs, morePreciseFloat
from utils.warnings import Warn
from versatileAnn.newModule.callbacks import StoreEpochData


class _NewWrapper_preRunTests:
    def __init__(self, **kwargs):
        self._outputsStruct = None

    @argValidator
    def preRunTests(self, trainDataloader, *, losses: List[nn.modules.loss._Loss], maxEpochs=5,
                    savePath, tensorboardPath='', valDataloader=None, lrRange=(1e-6, 5),
                    lrNumSteps=20, lrsToFindBest=None, fastDevRunKwargs=None,
                    overfitBatchesKwargs=None, profilerKwargs=None, findBestLearningRateKwargs=None,
                    findBestBatchSizesKwargs=None, **kwargs):
        # kkk correct this args
        fastDevRunKwargs = fastDevRunKwargs or {}
        overfitBatchesKwargs = overfitBatchesKwargs or {}
        profilerKwargs = profilerKwargs or {}
        findBestLearningRateKwargs = findBestLearningRateKwargs or {}
        kwargsRelatedToTrainer = getMethodRelatedKwargs(pl.Trainer, kwargs, delAfter=True)

        self.runFastDevRun(fastDevRunKwargs, kwargsRelatedToTrainer,
                           trainDataloader, valDataloader)
        self.runOverfitBatches(kwargsRelatedToTrainer, overfitBatchesKwargs,
                               trainDataloader, valDataloader)
        trainer = self.runProfiler(kwargsRelatedToTrainer, profilerKwargs,
                                   trainDataloader, valDataloader)

        self.findBestLearningRate(trainDataloader, valDataloader,
                                  lrRange=lrRange, numSteps=lrNumSteps,
                                  lrsToFindBest=lrsToFindBest,
                                  findBestLearningRateKwargs=findBestLearningRateKwargs,
                                  kwargsRelatedToTrainer=kwargsRelatedToTrainer)  # kkk2 add kwargs also for this
    def runFastDevRun(self, fastDevRunKwargs, kwargsRelatedToTrainer, trainDataloader,
                      valDataloader):
        # cccDevAlgo
        #  ensures whole pipeline is working correctly by running couple of epochs on a batch
        self.printTestPrints('running fastDevRun')

        kwargsApplied = {'min_epochs': 2, 'logger': False, }

        self._getKwargsApplied_forRelatedRun(kwargsRelatedToTrainer, fastDevRunKwargs,
                                             kwargsApplied)
        # disallow changing 'fast_dev_run' option
        if 'fast_dev_run' in kwargsApplied:
            del kwargsApplied['fast_dev_run']

        trainer = pl.Trainer(
            fast_dev_run=True,  # Run only for a small number of epochs for faster development
            **kwargsApplied)
        trainer.fit(self, train_dataloaders=trainDataloader, val_dataloaders=valDataloader)

    def runOverfitBatches(self, kwargsRelatedToTrainer, overfitBatchesKwargs, trainDataloader,
                          valDataloader):
        self.printTestPrints('running overfitBatches')
        # bugPotentialCheck2
        #  with including 'overfit_batches' option, when the trainer is ran, "make sure you have
        #  set, VAnnTsDataset.indexes to .indexes of sampler". this is an indication of that the
        #  pytorchLighning tries to re__init__ the dataLoader.
        #  but apparently the dataLoaders passed here are kept unchanged and this reiniting are
        #  applied just internally. because the sampler of trainDataloader is still is instance of
        #  SamplerFor_vAnnTsDataset

        mainValLossName = self._getLossName('val', self.losses[0])
        lossRatioDecrease = {}
        callbacks_ = [StoreEpochData()]

        kwargsApplied = {'overfit_batches': True, 'max_epochs': 100,#kkk
                         'enable_checkpointing': False, 'logger': False, 'callbacks': callbacks_, }

        self._getKwargsApplied_forRelatedRun(kwargsRelatedToTrainer, overfitBatchesKwargs,
                                             kwargsApplied)

        if 'max_epochs' in kwargsApplied and kwargsApplied['max_epochs'] < 50:
            kwargsApplied['max_epochs'] = 50

        trainer = pl.Trainer(**kwargsApplied)
        trainer.fit(self, trainDataloader, valDataloader)

        self._printLossChanges(callbacks_)

    def runProfiler(self, kwargsRelatedToTrainer, profilerKwargs, trainDataloader,
                    valDataloader):
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

    def findBestLearningRate(self, trainDataloader, valDataloader, lrRange=(1e-6, 5), numSteps=20,
                             lrsToFindBest=None, findBestLearningRateKwargs=None,
                             kwargsRelatedToTrainer=None):
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

        kwargsApplied = {'max_epochs': 4, 'enable_checkpointing': False, 'logger': False, }
        self._getKwargsApplied_forRelatedRun(kwargsRelatedToTrainer, findBestLearningRateKwargs,
                                             kwargsApplied)

        mainValLossName = self._getLossName('val', self.losses[0])
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
            self._collectMetricsOfRun(callbacks_, lossRatioDecrease, mainValLossName,
                                      thisLr)

        bestLearningRate = min(lossRatioDecrease, key=lambda k: lossRatioDecrease[k]['score'])
        worstLearningRate = max(lossRatioDecrease, key=lambda k: lossRatioDecrease[k]['score'])
        Warn.info(f'bestLearningRate is {bestLearningRate} and the worst being {worstLearningRate}')

        if not self.keepLr_notReplaceWithBestLr:
            self.lr = bestLearningRate
        else:
            self.lr = pastLr
        # goodToHave2
        #  add ploting in tensorboard with a message that plot is in tensorboard

