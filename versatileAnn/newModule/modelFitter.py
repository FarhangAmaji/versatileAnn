from typing import List, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from utils.typeCheck import argValidator
from utils.vAnnGeneralUtils import giveOnlyKwargsRelated_toMethod, snakeToCamel, \
    _allowOnlyCreationOf_ChildrenInstances
from utils.warnings import Warn


class _NewWrapper_modelFitter:
    def __init__(self):
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _NewWrapper_modelFitter)

    @argValidator
    def train(self, trainDataLoader: DataLoader,
              valDataLoader: Union[DataLoader, None] = None,
              listOfKwargs: List[dict] = None, **kwargs):
        # kkk support log kwargs to have phases
        # addTest1
        # cccUsage
        #  - this method accepts kwargs related to trainer, trainer.fit, and model.log and
        #  pass them accordingly
        #  - the order in listOfKwargs is important
        # put together all kwargs user wants to pass to trainer, trainer.fit, and model.log
        listOfKwargs = listOfKwargs or []
        listOfKwargs.append(kwargs)
        allUserKwargs = {}
        for kw in listOfKwargs:
            allUserKwargs.update(kw)

        appliedKwargs = self._getArgsRelated_toEachMethodSeparately(allUserKwargs)

        notAllowedArgs = ['self', 'overfit_batches', 'name', 'value']
        # cccDevStruct
        #  - 'name','value' can be used in logging and are not allowed as the
        #       _logLosses in _NewWrapper_lossNRegularization module sets them itself
        #  - overfit_batches is not compatible with this project
        #       for more info take look at 'cccDevStruct' at runOverfitBatches
        self._removeNotAllowedArgs(allUserKwargs, appliedKwargs, notAllowedArgs)

        self._warnForNotUsedArgs(allUserKwargs, appliedKwargs)

        trainer = pl.Trainer(**appliedKwargs['trainer'])

        self._logOptions = appliedKwargs['log']

        if 'train_dataloaders' in appliedKwargs['trainerFit']:
            del appliedKwargs['trainerFit']['train_dataloaders']
        if 'val_dataloaders' in appliedKwargs['trainerFit']:
            del appliedKwargs['trainerFit']['val_dataloaders']
        trainer.fit(self, trainDataLoader, valDataLoader, **appliedKwargs['trainerFit'])

        self._logOptions = {}
        return trainer

    def _getArgsRelated_toEachMethodSeparately(self, allUserKwargs):
        appliedKwargs = {}
        for meth, methName in zip([pl.Trainer, pl.Trainer.fit, pl.LightningModule.log],
                                  ['trainer', 'trainerFit', 'log']):
            # cccDevAlgo
            #  - note I have checked these 3 methods and they don't have args with mutual names
            #  except pl.Trainer and pl.LightningModule.log which take 'logger' arg which is ok
            #  to be mutual
            #  - note giveOnlyKwargsRelated_toMethod has camelCase compatibility
            #  for i.e. if the method takes `my_arg` but updater has
            #  `myArg`, includes `my_arg` as 'myArg'
            appliedKwargs[methName] = {}
            giveOnlyKwargsRelated_toMethod(meth, updater=allUserKwargs,
                                           updatee=appliedKwargs[methName])
        return appliedKwargs

    def _removeNotAllowedArgs(self, allUserKwargs, appliedKwargs, notAllowedArgs):
        for naa in notAllowedArgs:
            for ak in appliedKwargs.keys():
                for ak2 in appliedKwargs[ak].keys():
                    if naa == ak2 or naa == snakeToCamel(ak2):
                        Warn.warn(f'you have include {naa} in kwargs you passed, but' + \
                                  'this is not allowed and omitted')
                        del appliedKwargs[ak][ak2]
                        del allUserKwargs[ak2]  # in order to give warning again in leftOvers part

    def _warnForNotUsedArgs(self, allUserKwargs, appliedKwargs):
        # warn for left over kwargs, the kwargs which user has passed but don't
        # fit in pl.Trainer, pl.Trainer.fit or pl.LightningModule.log
        for auk in allUserKwargs:
            isInVolved = False
            for ak in appliedKwargs.keys():
                if isInVolved:
                    break
                for ak2 in appliedKwargs[ak].keys():
                    if auk == ak2 or auk == snakeToCamel(ak2):
                        isInVolved = True
                        break
            if not isInVolved:
                Warn.warn(f"you have included {auk} but it doesn't match with args " + \
                          "can be passed to pl.Trainer, pl.Trainer.fit or " + \
                          "pl.LightningModule.log; even their camelCase names")
