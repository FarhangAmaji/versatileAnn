from abc import ABC
from typing import List, Union

class _NewWrapper_preRunTests:
    def __init__(self, **kwargs):
        self._outputsStruct = None

    @argValidator
    def preRunTests(self, trainDataloader, *, losses: List[nn.modules.loss._Loss], maxEpochs=5,
                    savePath, tensorboardPath='', valDataloader=None, lrRange=(1e-6, 5),
                    lrNumSteps=20, lrsToFindBest=None, externalKwargs=None, fastDevRunKwargs=None,
                    overfitBatchesKwargs=None, profilerKwargs=None,
                    **kwargs):
        # kkk correct this args
        fastDevRunKwargs = fastDevRunKwargs or {}
        overfitBatchesKwargs = overfitBatchesKwargs or {}
        profilerKwargs = profilerKwargs or {}
        kwargsRelatedToTrainer = getMethodRelatedKwargs(pl.Trainer, kwargs, delAfter=True)

        self.runFastDevRun(fastDevRunKwargs, kwargsRelatedToTrainer,
                           trainDataloader, valDataloader)
        self.runOverfitBatches(kwargsRelatedToTrainer, overfitBatchesKwargs,
                               trainDataloader, valDataloader)
        trainer = self.runProfiler(kwargsRelatedToTrainer, profilerKwargs,
                                   trainDataloader, valDataloader)

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

