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
        kwargsRelatedToTrainer = getMethodRelatedKwargs(pl.Trainer, kwargs, delAfter=True)

        self.runFastDevRun(fastDevRunKwargs, kwargsRelatedToTrainer,
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

