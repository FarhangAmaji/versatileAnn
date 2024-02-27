"""
has 2 main methods fit and baseFit
"""
import os
from typing import Iterable
from typing import List, Union, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers import Logger
from torch import nn
from torch.utils.data import DataLoader

from brazingTorchFolder.utils import externalFit
from projectUtils.dataTypeUtils.dict import giveOnlyKwargsRelated_toMethod
from projectUtils.dataTypeUtils.str import snakeToCamel
from projectUtils.misc import _allowOnlyCreationOf_ChildrenInstances, inputTimeout, nFoldersBack, \
    varPasser
from projectUtils.typeCheck import argValidator
from projectUtils.warnings import Warn


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
            valDataloader: Optional[DataLoader] = None,
            *, lossFuncs: List[nn.modules.loss._Loss],
            seed=None, resume=True, seedSensitive=False,
            addDefaultLogger=True, addDefault_gradientClipping=True,
            warmUp_epochNum=5, addDefault_reduceLROnPlateau=True,
            preRunTests_force=False, preRunTests_seedSensitive=False,
            preRunTests_lrsToFindBest=None,
            preRunTests_batchSizesToFindBest=None,
            preRunTests_fastDevRunKwargs=None, preRunTests_overfitBatchesKwargs=None,
            preRunTests_profilerKwargs=None, preRunTests_findBestLearningRateKwargs=None,
            preRunTests_findBestBatchSizesKwargs=None,
            **kwargs):

        # cccUsage
        #  note there are many args related to preRunTests; you may also run preRunTests separately

        # cccDevStruct
        #  note this method in some cases is loading another instance and runs on that
        #  but changing self with methods of an instance is not possible so take a look
        #  at devDocs\codeClarifier\replaceAnotherInstance_withSelf.py
        #  therefore this method is implemented as an external method in brazingTorchFolder/utils.py

        kwargs_ = varPasser(
            localArgNames=['trainDataloader', 'valDataloader', 'lossFuncs', 'seed', 'resume',
                           'seedSensitive', 'addDefaultLogger', 'addDefault_gradientClipping',
                           'warmUp_epochNum', 'addDefault_reduceLROnPlateau',
                           'preRunTests_force', 'preRunTests_seedSensitive',
                           'preRunTests_lrsToFindBest', 'preRunTests_batchSizesToFindBest',
                           'preRunTests_fastDevRunKwargs', 'preRunTests_overfitBatchesKwargs',
                           'preRunTests_profilerKwargs', 'preRunTests_findBestLearningRateKwargs',
                           'preRunTests_findBestBatchSizesKwargs'])

        return externalFit(**kwargs_, **kwargs)

    @argValidator
    def baseFit(self, trainDataloader: DataLoader,
                valDataloader: Union[DataLoader, None] = None,
                addDefaultLogger=True, addDefault_gradientClipping=True,
                listOfKwargs: List[dict] = None,
                **kwargs):

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

    # ----
    def _determineFitRunState(self, seed, resume=True, seedSensitive=False):
        # goodToHave2
        #  this is similar to _determineShouldRun_preRunTests
        # addTest1

        # cccDevStruct(same as _determineShouldRun_preRunTests)
        #  there was a idea about ""architectureName should be figured out in postInit"" but it may
        #  cause problems with loggerPath in preRunTests and .fit method

        # by default these values are assumed and will be
        # changed depending on the case
        fitRunState = 'beginning'  # fitRunState can have these: "beginning", "resume", "don't run"
        architectureName = 'arch1'
        checkpointPath = ''
        runName = f'mainRun_seed{seed}'
        isModelChanged = False

        dummyLogger = pl.loggers.TensorBoardLogger(self.modelName)
        loggerPath = os.path.abspath(dummyLogger.log_dir)
        # loggerPath is fullPath including 'modelName/someName/version_0'

        if os.path.exists(nFoldersBack(loggerPath, n=2)):
            # there is a model run before with the name of this model

            architectureDicts = self._collectArchDicts(loggerPath)
            # update loggerPath with now found archDicts
            if architectureDicts:
                loggerPath = self._updateLoggerPath_withExistingArchName(architectureDicts, runName)
            architectureDicts_withMatchedAllDefinitions = self._getArchitectureDicts_withMatchedAllDefinitions(
                architectureDicts)
            # matchedAllDefinitions means the exact same model structure as all
            # layers and their definitions are exactly the same
            # note architectureDicts matches _saveArchitectureDict

            # remove "architecture.pkl" which is in the "preRunTests" folder
            architectureDicts_withMatchedAllDefinitions = self._remove_architectureDicts_fromPreRunTestsFolder(
                architectureDicts_withMatchedAllDefinitions)

            if architectureDicts_withMatchedAllDefinitions:
                # note being here means some models with the exact same
                # structure as this model has run before

                # Check if there is a model with the same seed
                # note this is gonna be used in various cases below
                matchedSeedDict, matchedSeedDict_filePath = self._findSeedMatch_inArchitectureDicts(
                    architectureDicts_withMatchedAllDefinitions, seed, returnCheckPointPath=True)

                checkpointPath, fitRunState, runName, isModelChanged = self._fitRunState_conditions(
                    checkpointPath, matchedSeedDict, matchedSeedDict_filePath, resume,
                    runName, seedSensitive, isModelChanged)
            else:
                # there are models with the name of this model but with different structures

                # finds new architectureName
                architectureName = self._findAvailableArchName(nFoldersBack(loggerPath, n=2))

        else:
            # no model with this name in directory has never run
            pass  # so default fitRunState and architectureName are applied

        dummyLogger = pl.loggers.TensorBoardLogger(self.modelName,
                                                   name=architectureName,
                                                   version=runName)
        loggerPath = os.path.abspath(dummyLogger.log_dir)

        return architectureName, loggerPath, fitRunState, checkpointPath, isModelChanged

    def _fitRunState_conditions(self, checkpointPath,
                                matchedSeedDict, matchedSeedDict_filePath,
                                resume, runName, seedSensitive, isModelChanged):
        # cccDevStruct
        #  - note being here means some models with the exact same
        #  structure as this model has run before
        #  - note there is no option to have duplicate models with the
        #  same seed as the result would be the same anyway, so having
        #  duplicate models with the same seed is useless
        #  - 'no resume' means tend to run from beginning
        if seedSensitive:
            if matchedSeedDict:
                if resume:
                    # Resume the model with the same seed
                    fitRunState = 'resume'
                    checkpointPath = matchedSeedDict_filePath
                else:
                    # 'no resume' means tend to run from beginning

                    # Ask the user whether to run the model with the same
                    # seed from beginning and replace the old save
                    if self._askUserToReplaceModel():
                        fitRunState = 'beginning'
                    else:
                        fitRunState = "don't run"
            else:
                # cccDevStruct
                #  there is no matched seed and it's seedSensitive(meaning that
                #  doesn't prefer to resume or start from the beginning models
                #  saved with other seeds)
                #  - thus only option is to run from the beginning
                fitRunState = 'beginning'
        else:
            if resume:  # not seedSensitive and resume
                # cccDevStruct
                #  even it's no seedSensitive if there is a model with the same
                #  seed, so resumes that because it is probably more desired
                if matchedSeedDict:
                    # Resume the model which has the same seed
                    fitRunState = 'resume'
                    checkpointPath = matchedSeedDict_filePath
                else:
                    # Resumes any other seed
                    # cccDevStruct
                    #  note as the seeds are not the same, this option only wants to resume
                    #  sth, later in .fit when the model(with different seed) is loaded
                    #  there would be a warning for 'seed is changing'
                    fitRunState = 'resume'
                    # cccDevStruct
                    #  note as you have guessed matchedSeedDict_filePath is checkPointPath
                    #  for a model with different seed
                    checkpointPath = matchedSeedDict_filePath
                    runName = matchedSeedDict_filePath.split(os.sep)[-2]
                    isModelChanged = True
                    Warn.info("loading a model with another seed")
            else:  # no resume and no seedSensitive
                # cccDevStruct
                #  even it's no seedSensitive if there is a model with the same seed,
                #  thats more prefered; also note duplicate models with same seeds
                #  are not allowed
                if matchedSeedDict:
                    # Ask the user whether to run the model with the same
                    # seed from beginning and replace the old save
                    if self._askUserToReplaceModel():
                        fitRunState = 'beginning'
                    else:
                        fitRunState = "don't run"
                else:  # not seedSensitive and not resume and no matchedSeedDict
                    # here the seed differs and 'no resume' means tend to run from
                    # beginning with its own seed
                    fitRunState = 'beginning'
        return checkpointPath, fitRunState, runName, isModelChanged

    def _askUserToReplaceModel(self):
        # Use inputTimeout to ask the user whether to replace the model
        answer = inputTimeout("Do you want to replace the model? (yes/no)", timeout=30)
        if answer:
            return answer.lower() == 'yes'
        return False

    def _remove_architectureDicts_fromPreRunTestsFolder(self, listOfDicts):
        """
        removes architectureDict which is "architectureDict" folder
        """
        newListOfDicts = []

        for currentDict in listOfDicts:
            for key, value in currentDict.items():
                pathComponents = key.split(os.sep)
                # Check if "architecture.pkl" is not in "preRunTests" folder
                if pathComponents[-2] != "preRunTests":
                    newListOfDicts.append(currentDict)
        return newListOfDicts

    @staticmethod
    def _warnIf_modelIsChanged():
        Warn.error("WARNING: note .fit method returns instance(or self) and also the trainer." + \
                   "\nspecially in this case, which the instance is replaced, make sure" + \
                   "that you are catching new instance like 'instance, trainer=instance.fit(...)'")
        answer = inputTimeout("do you want to stop the code to follow the pattern? (yes/no)",
                              timeout=30)
        return answer
