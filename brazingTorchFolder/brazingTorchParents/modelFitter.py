from typing import List, Union, Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger
from torch import nn
from torch.utils.data import DataLoader

from brazingTorchFolder.brazingTorchParents.innerClassesWithoutPublicMethods.modelFitter_inner import \
    _BrazingTorch_modelFitter_inner
from brazingTorchFolder.utilsFolder.utils import externalFit
from projectUtils.misc import _allowOnlyCreationOf_ChildrenInstances, varPasser
from projectUtils.typeCheck import argValidator
from projectUtils.warnings import Warn


class _BrazingTorch_modelFitter(_BrazingTorch_modelFitter_inner):
    _phaseBasedLoggingTypes = ['train', 'val', 'test', 'predict', 'else']
    _methodsFitCanHandle_names = ['trainer', 'trainerFit', 'log']

    def __init__(self):
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _BrazingTorch_modelFitter)

    @argValidator
    def fit(self, trainDataloader: DataLoader,
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

        # cccUsage
        #  - **kwargs are any argument related to pytorch lightning trainer, trainer.fit,
        #       and self.log
        #  - note there are many args related to preRunTests; you may want to run preRunTests separately

        # case the lossFuncs is not passed here
        lossFuncs = self._setLossFuncs_ifNot(lossFuncs)

        # ccc1
        #  note this method in some cases is loading another instance and runs on that
        #  but changing self with methods of an instance is not possible so take a look
        #  at devDocs\codeClarifier\replaceAnotherInstance_withSelf.py
        #  therefore this method is implemented as an external method in brazingTorchFolder/utils.py

        kwargs_ = varPasser(
            localArgNames=['trainDataloader', 'valDataloader', 'lossFuncs', 'seed', 'resume',
                           'seedSensitive', 'addDefaultLogger', 'addDefault_gradientClipping',
                           'warmUp_epochNum', 'addDefault_reduceLROnPlateau',
                           'addDefault_earlyStopping',
                           'preRunTests_force', 'preRunTests_seedSensitive',
                           'preRunTests_lrsToFindBest', 'preRunTests_batchSizesToFindBest',
                           'preRunTests_fastDevRunKwargs', 'preRunTests_overfitBatchesKwargs',
                           'preRunTests_profilerKwargs', 'preRunTests_findBestLearningRateKwargs',
                           'preRunTests_findBestBatchSizesKwargs'])

        return externalFit(self, **kwargs_, **kwargs)

    @argValidator
    def baseFit(self, trainDataloader: DataLoader,
                valDataloader: Optional[Union[DataLoader]] = None,
                addDefaultLogger=True, addDefault_gradientClipping=True,
                listOfKwargs: List[dict] = None,
                **kwargs):

        # cccUsage
        #  - **kwargs or listOfKwargs are any argument related to pytorch lightning trainer, trainer.fit,
        #       and self.log
        #  - the order in listOfKwargs is important:
        #       - the later ones overwrite the earliers
        #       - kwargs are always the last (may overwrite others)
        #  - logging kwargs: kwargs related to logging may be specified with phase:
        #       - args related to self.log may be a dict with these keys 'train', 'val', 'test',
        #                   'predict' or 'else'
        #       - this way u can specify what exact phase use what values and what values the rest
        #               of phases (not specified ones) 'else' use

        # put together all kwargs user wants to pass to trainer, trainer.fit, and self.log
        appliedKwargs_byMethod = self._getBaseFit_appliedKwargs(kwargs, listOfKwargs)

        # add default logger if allowed and no logger is passed
        # because by default we are logging some metrics
        if addDefaultLogger and 'logger' not in appliedKwargs_byMethod['trainer']:
            appliedKwargs_byMethod['trainer']['logger'] = {
                pl.loggers.TensorBoardLogger(self.modelName)}

        notAllowedArgs = ['self', 'overfit_batches', 'name', 'value']
        # ccc3
        #  - 'name','value' can be used in logging and are not allowed as the
        #       _logLosses in _BrazingTorch_loss module sets them itself
        #  - overfit_batches is not compatible with this project
        #       for more info take look at 'ccc1' at runOverfitBatches
        self._removeNotAllowedArgs(appliedKwargs_byMethod, notAllowedArgs)

        # add gradient clipping by default
        if not self.noAdditionalOptions and addDefault_gradientClipping \
                and 'gradient_clip_val' not in appliedKwargs_byMethod['trainer']:
            appliedKwargs_byMethod['trainer']['gradient_clip_val'] = 0.1
            Warn.info('gradient_clip_val is not provided to fit;' + \
                      ' so by default it is set to default "0.1"' + \
                      '\nto cancel it, you may either pass noAdditionalOptions=True to model or ' + \
                      'pass addDefault_gradientClipping=False to fit method.' + \
                      '\nor set another value to "gradient_clip_val" in kwargs passed to fit method.')

        trainer = pl.Trainer(**appliedKwargs_byMethod['trainer'])

        self._logOptions = appliedKwargs_byMethod['log']

        # ccc3
        #  the user may have passed the train_dataloaders and val_dataloaders in kwargs
        #  note they are not going to be applied and those arguments should be passed to this method
        #  directly
        if 'train_dataloaders' in appliedKwargs_byMethod['trainerFit']:
            del appliedKwargs_byMethod['trainerFit']['train_dataloaders']
        if 'val_dataloaders' in appliedKwargs_byMethod['trainerFit']:
            del appliedKwargs_byMethod['trainerFit']['val_dataloaders']
        trainer.fit(self, trainDataloader, valDataloader, **appliedKwargs_byMethod['trainerFit'])

        self._logOptions = {}
        return trainer
