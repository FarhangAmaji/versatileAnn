from typing import List, Union, Iterable, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger
from torch.utils.data import DataLoader

from utils.dataTypeUtils.dict import giveOnlyKwargsRelated_toMethod
from utils.dataTypeUtils.str import snakeToCamel
from utils.generalUtils import _allowOnlyCreationOf_ChildrenInstances
from utils.typeCheck import argValidator
from utils.warnings import Warn


# kkk
#  does pl set trainer to model after training once? if so then in continuation
#  (for i.e. after loading model) we may not use .fit of this class

class _BrazingTorch_modelFitter:
    def __init__(self):
        # not allowing this class to have direct instance
        _allowOnlyCreationOf_ChildrenInstances(self, _BrazingTorch_modelFitter)

    @property
    def _logOptions(self):
        return self.__logOptions

    @_logOptions.setter
    @argValidator
    def _logOptions(self, value: dict):
        self._assertPhaseBased_logOptions(value)
        self.__logOptions = value

    @argValidator
    def fit(self, trainDataloader: DataLoader,
            valDataloader: Union[DataLoader, None] = None,
            addDefault_gradientClipping=True, **kwargs):
            listOfKwargs: List[dict] = None, addDefaultLogger=True,
        # kkk support log kwargs to have phases
        # addTest1
        # cccUsage
        #  - this method accepts kwargs related to trainer, trainer.fit, and self.log and
        #  pass them accordingly
        #  - the order in listOfKwargs is important
        #  - _logOptions phase based values feature:
        #           - args related to self.log may be a dict with these keys 'train', 'val', 'test',
        #                   'predict' or 'else'
        #           - this way u can specify what phase use what values and if not specified with
        #               'else' it's gonna know

        # put together all kwargs user wants to pass to trainer, trainer.fit, and self.log
        listOfKwargs = listOfKwargs or []
        listOfKwargs.append(kwargs)
        allUserKwargs = {}
        for kw in listOfKwargs:
            self._plKwargUpdater(allUserKwargs, kw)

        # add default logger if allowed and no logger is passes
        # because by default we are logging some metrics
        if addDefaultLogger and 'logger' not in allUserKwargs:
            allUserKwargs['logger'] = pl.loggers.TensorBoardLogger(self.modelName)
            # bugPotentialCheck1
            #  shouldn't this default logger have architectureName

        appliedKwargs = self._getArgsRelated_toEachMethodSeparately(allUserKwargs)

        notAllowedArgs = ['self', 'overfit_batches', 'name', 'value']
        # cccDevStruct
        #  - 'name','value' can be used in logging and are not allowed as the
        #       _logLosses in _BrazingTorch_loss module sets them itself
        #  - overfit_batches is not compatible with this project
        #       for more info take look at 'cccDevStruct' at runOverfitBatches
        self._removeNotAllowedArgs(allUserKwargs, appliedKwargs, notAllowedArgs)

        self._warnForNotUsedArgs(allUserKwargs, appliedKwargs)

        # add gradient clipping by default
        if not self.noAdditionalOptions and addDefault_gradientClipping \
                and 'gradient_clip_val' not in appliedKwargs['trainer']:
            appliedKwargs['trainer']['gradient_clip_val'] = 0.1
            Warn.info('gradient_clip_val is not provided to fit;' + \
                      ' so by default it is set to default "0.1"' + \
                      '\nto cancel it, you may either pass noAdditionalOptions=True to model or ' + \
                      'pass addDefault_gradientClipping=False to fit method.' + \
                      '\nor set another value to "gradient_clip_val" in kwargs passed to fit method.')

        trainer = pl.Trainer(**appliedKwargs['trainer'])

        self._logOptions = appliedKwargs['log']

        if 'train_dataloaders' in appliedKwargs['trainerFit']:
            del appliedKwargs['trainerFit']['train_dataloaders']
        if 'val_dataloaders' in appliedKwargs['trainerFit']:
            del appliedKwargs['trainerFit']['val_dataloaders']
        trainer.fit(self, trainDataloader, valDataloader, **appliedKwargs['trainerFit'])

        self._logOptions = {}
        return trainer

    def _getArgsRelated_toEachMethodSeparately(self, allUserKwargs):
        appliedKwargs = {}
        for meth, methName in zip([pl.Trainer, pl.Trainer.fit, pl.LightningModule.log],
                                  ['trainer', 'trainerFit', 'log']):
            # cccDevAlgo
            #  - note I have checked these 3 methods and they don't have args with mutual names
            #           except pl.Trainer and pl.LightningModule.log which take 'logger' arg
            #           which is ok, even though logger in self.log is just a bool or None but in
            #           pl.Trainer is a Logger object or a list of Logger objects or bool or None
            #  to be mutual
            #  - note giveOnlyKwargsRelated_toMethod has camelCase compatibility
            #  for i.e. if the method takes `my_arg` but updater has
            #  `myArg`, includes `my_arg` as 'myArg'
            appliedKwargs[methName] = {}
            appliedKwargs[methName] = giveOnlyKwargsRelated_toMethod(meth, updater=allUserKwargs,
                                                                     updatee=appliedKwargs[
                                                                         methName])
        return appliedKwargs

    def _removeNotAllowedArgs(self, allUserKwargs, appliedKwargs, notAllowedArgs):
        # addTest2
        for naa in notAllowedArgs:
            for ak in appliedKwargs.keys():
                for ak2 in appliedKwargs[ak].keys():
                    if naa == ak2 or naa == snakeToCamel(ak2):
                        Warn.warn(f'you have include {naa} in kwargs you passed, but' + \
                                  'this is not allowed and omitted')
                        del appliedKwargs[ak][ak2]
                        del allUserKwargs[ak2]  # in order to give warning again in leftOvers part

    def _warnForNotUsedArgs(self, allUserKwargs, appliedKwargs):
        # addTest2
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

    def _assertPhaseBased_logOptions(self, _logOptions):
        # assert phaseBased _logOptions to fit in the format it should have
        for akl, aklV in _logOptions.items():
            if isinstance(aklV, dict):
                if akl not in ['train', 'val', 'test', 'predict', 'else']:
                    raise ValueError("it seems you are trying to use phaseBased logOptions" + \
                                     f"\nthe key '{akl}' must be " + \
                                     "'train', 'val', 'test', 'predict', 'else'")

    # ---- _plKwargUpdater and inners
    def _plKwargUpdater(self, allUserKwargs, kw):
        # cccDevStruct
        #  pytorch lightning for some args may get different type
        #  this methods makes sure those options are correctly applied
        #  for i.e. logger can be a Logger object or a list/
        kw_ = kw.copy()
        correctArgs = {}
        if 'logger' in allUserKwargs and 'logger' in kw:
            correctArgs['logger'] = self._putTogether_plLoggers(allUserKwargs['logger'],
                                                                kw['logger'])
            del kw_['logger']
        # 'plugins' is also like callbacks. but I don't take care of as this option is rare
        if 'callbacks' in allUserKwargs and 'callbacks' in kw:
            correctArgs['callbacks'] = self._putTogether_plCallbacks(allUserKwargs['callbacks'],
                                                                     kw['callbacks'])
            del kw_['callbacks']
        allUserKwargs.update(kw_)
        allUserKwargs.update(correctArgs)
        return allUserKwargs

    @argValidator
    def _putTogether_plLoggers(self,
                               var1: Optional[Union[Logger, Iterable[Logger], bool]],
                               var2: Optional[Union[Logger, Iterable[Logger], bool]]) \
            -> Union[Logger, List[Logger], None, bool]:
        # addTest2
        # cccDevALgo
        #  - each pytorch lightning arg may get a Logger object or a list of Logger
        #       objects or None Or bool
        #  - note have higher importance in setting Logger, Iterable[Logger] than None or bool

        # Check if either var1 or var2 is None or bool
        if var1 is None:
            return var2
        elif isinstance(var1, bool):
            if var2 is not None:
                return var2
            else:
                return var1

        if var2 is None:
            return var1
        elif isinstance(var2, bool):
            # we know var1 is not None now (because of previous if block)
            if isinstance(var1, bool):
                # when both are bool var2 has higher importance
                return var2
            else:
                # when var1 is not None or bool means it's Logger or Iterable[Logger] so
                # it's returned
                return var1

        result = []
        if isinstance(var1, Iterable):
            # goodToHave3
            #  check does argValidator check for Iterable[Logger] or not
            # Check if all elements of var1 are Logger
            if not all(isinstance(logger, Logger) for logger in var1):
                raise ValueError('var1 has some elements which are not Logger objects')
            # bugPotentialCheck1
            #  I checked and found that list(here result) can extend tuples and sets as well
            #  but I don't know what happens for other iterables
            result.extend(var1)
        else:
            result.append(var1)

        if isinstance(var2, Iterable):
            # Check if all elements of var2 are Logger
            if not all(isinstance(logger, Logger) for logger in var2):
                raise ValueError('var2 has some elements which are not Logger objects')
            result.extend(var2)
        else:
            result.append(var2)
        return result

    @argValidator
    def _putTogether_plCallbacks(self,
                                 var1: Optional[Union[List[Callback], Callback]],
                                 var2: Optional[Union[List[Callback], Callback]]) \
            -> Union[Callback, List[Callback], None]:
        # cccDevALgo
        #  - each pytorch lightning arg may get a Callback object or a list of Callback or None
        #  - note have higher importance in setting Callback, Iterable[Callback] than None

        # Check if either var1 or var2 is None
        if var1 is None:
            return var2
        if var2 is None:
            return var1

        result = []
        if isinstance(var1, Iterable):
            # goodToHave3 like above
            if not all(isinstance(callback, Callback) for callback in var1):
                raise ValueError('var1 has some elements which are not Callback objects')
            # bugPotentialCheck1 like above
            result.extend(var1)
        else:
            result.append(var1)

        if isinstance(var2, Iterable):
            # goodToHave3 like above
            if not all(isinstance(callback, Callback) for callback in var2):
                raise ValueError('var2 has some elements which are not Callback objects')
            # bugPotentialCheck1 like above
            result.extend(var2)
        else:
            result.append(var2)
        return result
